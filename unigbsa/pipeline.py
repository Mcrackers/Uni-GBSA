import os
import shutil
import argparse
import traceback
import multiprocessing
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from unigbsa.version import __version__
from unigbsa.gbsa.gbsarun import GBSA
from unigbsa.utils import generate_index_file, load_configue_file
from unigbsa.simulation.mdrun import GMXEngine
from unigbsa.simulation.topology import build_topol, build_protein
from unigbsa.simulation.utils import ligand_validate
from unigbsa.settings import (
    logging,
    DEFAULT_CONFIGURE_FILE,
    GMXEXE,
    set_OMP_NUM_THREADS,
    PathManager,
)


KEY = [
    'ligandName',
    'Frames',
    'mode',
    'complex',
    'receptor',
    'ligand',
    'Internal',
    'Van der Waals',
    'Electrostatic',
    'Polar Solvation',
    'Non-Polar Solvation',
    'Gas',
    'Solvation',
    'TOTAL',
    'status',
]


def _empty_energy_frame(modes):
    """Placeholder row used when topology or GBSA fails."""
    return pd.DataFrame(
        {
            'Frames': 1,
            'mode': modes,
            'complex': 0.0,
            'receptor': 0.0,
            'ligand': 0.0,
            'Internal': 0.0,
            'Van der Waals': 0.0,
            'Electrostatic': 0,
            'Polar Solvation': 0.0,
            'Non-Polar Solvation': 0.0,
            'Gas': 0.0,
            'Solvation': 0.0,
            'TOTAL': 0.0,
        },
        index=[1],
    )


def _reres(infile, outfile):
    """Renumber residues from 1 via gmx editconf."""
    cmd = (
        '%s editconf -f %s -o %s -resnr 1 >/dev/null 2>&1'
        % (GMXEXE, infile, outfile)
    )
    if os.system(cmd) != 0:
        raise Exception('Error convert %s to %s' % (infile, outfile))
    return outfile


def _enter_ligand_dir(ligandfile, validate=False):
    """Create the ligand directory, chdir into it, optionally validate."""
    ligandfile = os.path.abspath(ligandfile)
    ligandName = os.path.split(ligandfile)[-1][:-4]
    if not os.path.exists(ligandName):
        os.mkdir(ligandName)
    os.chdir(ligandName)
    if validate:
        ligandfile = ligand_validate(ligandfile, ligandName + '.mol')
    return ligandName, ligandfile


def _build_complex(receptor, ligandfile, simParas, nt):
    """Build complex.pdb / complex.top; return those names and the index file."""
    grofile, topfile = 'complex.pdb', 'complex.top'
    indexfile = build_topol(
        receptor,
        ligandfile,
        outpdb=grofile,
        outtop=topfile,
        ligandforce=simParas['ligandforcefield'],
        charge_method=simParas['ligandCharge'],
        nt=nt,
    )
    return grofile, topfile, indexfile


def _sim_box(simParas, nt):
    """Box and ion kwargs shared by GMXEngine.run_to_minim / run_to_md."""
    return dict(
        boxtype=simParas['boxtype'],
        boxsize=simParas['boxsize'],
        conc=simParas['conc'],
        nt=nt,
    )


def _write_energy_csv(df, outfile):
    """Write the standard energy columns to CSV."""
    df[KEY].to_csv(outfile, index=False)


# Numeric energy columns that are averaged over frames for the SDF output.
_ENERGY_COLS = [
    'Frames', 'complex', 'receptor', 'ligand',
    'Internal', 'Van der Waals', 'Electrostatic',
    'Polar Solvation', 'Non-Polar Solvation',
    'Gas', 'Solvation', 'TOTAL',
]


def _write_energy_sdf(df, ligandfiles, outfile):
    """Write a new SDF with mmGBSA results as molecule properties.

    One record per ligand.  Energy values are averaged over all successful
    frames; failed ligands (status != 'S') get their status written but
    energy properties are omitted.  Property names mirror the CSV columns
    with spaces replaced by underscores and prefixed with ``mmGBSA_``.
    """
    from rdkit import Chem
    from rdkit.Chem import SDWriter

    # Build lookup: ligandName → original SDF/MOL path
    name_to_file = {}
    for lf in ligandfiles:
        lf = os.path.abspath(lf)
        name = os.path.split(lf)[-1][:-4]
        name_to_file[name] = lf

    with SDWriter(outfile) as writer:
        for ligandName, group in df.groupby('ligandName', sort=False):
            lf = name_to_file.get(ligandName)
            if lf is None:
                logging.warning(
                    'SDF write: no input file found for %s, skipping.'
                    % ligandName
                )
                continue

            # Try to read the original molecule
            mol = None
            if lf.endswith('.sdf') or lf.endswith('.mol'):
                mol = Chem.MolFromMolFile(lf, removeHs=False)
            if mol is None:
                # Fall back to a molecule with no structure
                mol = Chem.MolFromSmiles('')
                logging.warning(
                    'SDF write: could not read structure from %s,'
                    ' writing properties only.' % lf
                )

            mol.SetProp('_Name', ligandName)

            # Status: 'S' if all frames succeeded, else the failure code
            statuses = group['status'].unique()
            status = 'S' if list(statuses) == ['S'] else statuses[0]
            mol.SetProp('mmGBSA_status', status)

            # Averaged mode string (e.g. 'gb')
            if 'mode' in group.columns:
                mol.SetProp('mmGBSA_mode', str(group['mode'].iloc[0]))

            # Average numeric columns over successful frames only
            ok = group[group['status'] == 'S']
            if not ok.empty:
                for col in _ENERGY_COLS:
                    if col in ok.columns:
                        val = ok[col].mean()
                        prop = 'mmGBSA_' + col.replace(' ', '_')
                        mol.SetDoubleProp(prop, float(val))

            writer.write(mol)

    logging.info('SDF results written to %s' % outfile)


def _tag_result(df, ligandName, status):
    """Copy a GBSA frame and set ligandName / status columns."""
    df = df.copy()
    df['ligandName'] = ligandName
    df['status'] = status
    return df


def _gbsa_job(
    ligandName,
    grofile,
    trajfile,
    topfile,
    indexfile,
    pbsaParas,
    mmpbsafile,
    verbose,
    receptorfile,
    ligandfile,
    modes,
    clean=False,
):
    """Picklable job dict for one gmx_MMPBSA instance."""
    return dict(
        ligandName=ligandName,
        grofile=grofile,
        trajfile=trajfile,
        topfile=topfile,
        indexfile=indexfile,
        pbsaParas=pbsaParas,
        mmpbsafile=mmpbsafile,
        verbose=verbose,
        receptorfile=receptorfile,
        ligandfile=ligandfile,
        modes=modes,
        clean=clean,
    )


def _run_gbsa_jobs(jobs, nt):
    """Run gmx_MMPBSA: 1 thread per instance, up to ``nt`` instances."""
    if not jobs:
        return None
    nworker = min(len(jobs), max(int(nt), 1))
    set_OMP_NUM_THREADS(1)
    logging.info(
        'GBSA: %d ligand(s), %d instance(s) x 1 thread.'
        % (len(jobs), nworker)
    )
    if len(jobs) == 1:
        frames = [single(jobs[0])]
    else:
        with Pool(nworker) as pool:
            frames = list(
                tqdm(pool.imap(single, jobs), total=len(jobs))
            )
    return pd.concat(frames)


def _combine_and_write(failed, gbsa_df, outfile, ligandfiles=None):
    """Write failed placeholders and GBSA frames to CSV and (optionally) SDF.

    When ``ligandfiles`` is provided and rdkit is available, also writes
    ``<outfile>.sdf`` with per-molecule mmGBSA properties.
    """
    frames = list(failed)
    if gbsa_df is not None:
        frames.append(gbsa_df)
    if not frames:
        raise Exception('No GBSA results to write.')
    df = pd.concat(frames)
    _write_energy_csv(df, outfile)

    if ligandfiles:
        sdf_out = outfile.replace('.csv', '') + '.sdf'
        try:
            _write_energy_sdf(df, ligandfiles, sdf_out)
        except ImportError:
            logging.warning(
                'rdkit not available — SDF output skipped.'
                ' Install with: conda install -c conda-forge rdkit'
            )


def traj_pipeline(
    complexfile,
    trajfile,
    topolfile,
    indexfile,
    pbsaParas=None,
    mmpbsafile=None,
    nt=1,
    verbose=False,
    input_rec_file=None,
    input_lig_file=None,
):
    """
    A pipeline for calculate GBSA/PBSA for trajectory.

    Args:
      complexfile: The name of the PDB file containing the protein-ligand
        complex.
      trajfile: the trajectory file, in pdb format
      topolfile: The topology file of the complex.
      indexfile: the index file for the complex
      mode: gb or tm. Defaults to gb
      dec: whether to decompose the free energy into its components.
        Defaults to False
      debug: if True, will print out all the debug messages. Defaults to False

    Returns:
      delta_G is a dictionary, the key is the mode, the value is a list, the
      first element is the average value, the second element is the standard
      deviation.
    """
    reresfile = _reres(complexfile, complexfile[:-4] + '_reres.pdb')

    pbsa = GBSA()
    pbsa.complex = os.path.abspath(reresfile)
    if input_rec_file:
        pbsa.input_rec_file = os.path.abspath(input_rec_file)
    if input_lig_file:
        pbsa.input_lig_file = os.path.abspath(input_lig_file)

    mmpbsafile = pbsa.set_paras(
        complexfile=reresfile,
        trajectoryfile=trajfile,
        topolfile=topolfile,
        indexfile=indexfile,
        pbsaParas=pbsaParas,
        mmpbsafile=mmpbsafile,
        nt=nt,
    )
    pbsa.run(verbose=verbose)
    delta_G = pbsa.extract_result()

    print("Frames    mode    delta_G(kcal/mole)")
    for i, irow in delta_G.iterrows():
        print(
            '%6d    %4s    %18.4f  '
            % (irow['Frames'], irow['mode'], irow['TOTAL'])
        )
    return delta_G


def base_pipeline(
    receptorfile,
    ligandfiles,
    paras,
    nt=1,
    mmpbsafile=None,
    outfile='BindingEnergy.csv',
    validate=False,
    verbose=False,
):
    """
    This function takes a receptorfile and ligandfile, and build a
    complex.pdb and complex.top file.

    Args:
      receptorfile: the file name of the receptor pdb file
      ligandfile: the name of the ligand file
      paras: a dictionary of parameters for the pipeline.
    """
    simParas = paras['simulation']
    pbsaParas = paras['GBSA']

    receptorfile = os.path.abspath(receptorfile)
    logging.info('Build protein topology.')
    receptor = build_protein(
        receptorfile, forcefield=simParas['proteinforcefield']
    )

    cwd = os.getcwd()
    jobs, failed = [], []
    d = _empty_energy_frame(pbsaParas['modes'])

    for ligandfile in ligandfiles:
        ligandName, ligandfile = _enter_ligand_dir(
            ligandfile, validate=validate
        )
        logging.info('Build ligand topology: %s' % ligandName)
        try:
            grofile, topfile, indexfile = _build_complex(
                receptor, ligandfile, simParas, nt
            )
            if not os.path.exists(indexfile):
                indexfile = generate_index_file(grofile)
        except Exception as e:
            if len(ligandfiles) == 1:
                traceback.print_exc()
            statu = 'F_top'
            logging.warning(
                'Failed to generate forcefield for ligand: %s' % ligandName
            )
            failed.append(_tag_result(d, ligandName, statu))
            os.chdir(cwd)
            continue

        jobs.append(_gbsa_job(
            ligandName,
            grofile,
            grofile,
            topfile,
            indexfile,
            pbsaParas,
            mmpbsafile,
            verbose,
            receptorfile,
            ligandfile,
            pbsaParas['modes'],
            clean=False,
        ))
        os.chdir(cwd)

    _combine_and_write(
        failed, _run_gbsa_jobs(jobs, nt), outfile,
        ligandfiles=ligandfiles,
    )


def single(arg):
    """Pool worker: one gmx_MMPBSA instance (1 thread) in the ligand directory."""
    set_OMP_NUM_THREADS(1)
    cwd = os.getcwd()
    os.chdir(arg['ligandName'])
    ligandName = arg['ligandName']
    statu = 'S'
    try:
        d1 = traj_pipeline(
            arg['grofile'],
            trajfile=arg['trajfile'],
            topolfile=arg['topfile'],
            indexfile=arg['indexfile'],
            pbsaParas=arg['pbsaParas'],
            mmpbsafile=arg['mmpbsafile'],
            verbose=arg['verbose'],
            nt=1,
            input_rec_file=arg['receptorfile'],
            input_lig_file=arg['ligandfile'],
        )
        if arg['clean'] and not arg['verbose']:
            GMXEngine().clean(pdbfile=arg['grofile'])
    except:
        traceback.print_exc()
        logging.warning(
            'Failed to run GBSA for ligand: %s' % ligandName
        )
        d1 = _empty_energy_frame(arg['modes'])
        statu = 'F_GBSA'
    os.chdir(cwd)
    return _tag_result(d1, ligandName, statu)


def minim_pipeline(
    receptorfile,
    ligandfiles,
    paras,
    mmpbsafile=None,
    nt=1,
    outfile='BindingEnergy.csv',
    validate=False,
    verbose=False,
):
    """
    It runs the simulation pipeline for each ligand.

    Args:
      receptorfile: The name of the receptor file.
      ligandfiles: a list of ligand files
      paras: a dictionary of parameters
      outfile: the output file name. Defaults to BindingEnergy.csv
    """
    simParas = paras['simulation']
    pbsaParas = paras['GBSA']

    receptorfile = os.path.abspath(receptorfile)
    logging.info('Build protein topology.')
    receptor = build_protein(
        receptorfile, forcefield=simParas['proteinforcefield']
    )

    ligandfiles = sorted(ligandfiles)
    logging.info(
        'EM: %d ligand(s), sequential gmx with %d thread(s).'
        % (len(ligandfiles), nt)
    )
    set_OMP_NUM_THREADS(nt)

    cwd = os.getcwd()
    jobs, failed = [], []
    d = _empty_energy_frame(pbsaParas['modes'])

    for ligandfile in ligandfiles:
        print('=' * 80)
        ligandName, ligandfile = _enter_ligand_dir(
            ligandfile, validate=validate
        )
        logging.info('Build ligand topology: %s' % ligandName)
        try:
            grofile, topfile, indexfile = _build_complex(
                receptor, ligandfile, simParas, nt
            )
        except Exception as e:
            traceback.print_exc()
            logging.warning(
                'Failed to generate forcefield for ligand: %s' % ligandName
            )
            failed.append(_tag_result(d, ligandName, 'F_top'))
            os.chdir(cwd)
            continue

        logging.info('Running energy minimization: %s' % ligandName)
        engine = GMXEngine()
        try:
            minimgro, outtop = engine.run_to_minim(
                grofile,
                topfile,
                maxsol=simParas['maxsol'],
                **_sim_box(simParas, nt),
            )
            _reres(minimgro, grofile)
            shutil.copy(topfile, outtop)
            if not os.path.exists(indexfile):
                indexfile = generate_index_file(grofile)
        except Exception as e:
            traceback.print_exc()
            logging.warning(
                'Failed to run simulation for ligand: %s' % ligandName
            )
            failed.append(_tag_result(d, ligandName, 'F_md'))
            os.chdir(cwd)
            continue

        jobs.append(_gbsa_job(
            ligandName,
            grofile,
            grofile,
            topfile,
            indexfile,
            pbsaParas,
            mmpbsafile,
            verbose,
            receptorfile,
            ligandfile,
            pbsaParas['modes'],
            clean=True,
        ))
        os.chdir(cwd)

    _combine_and_write(
        failed, _run_gbsa_jobs(jobs, nt), outfile,
        ligandfiles=ligandfiles,
    )


def md_pipeline(
    receptorfile,
    ligandfiles,
    paras,
    mmpbsafile=None,
    nt=1,
    outfile='BindingEnergy.csv',
    verbose=False,
):
    """
    The main function of this script.

    Args:
      receptorfile: the protein file
      ligandfiles: a list of ligand files
      paras: a dictionary of parameters
      outfile: the output file name. Defaults to BindingEnergy.csv
    """
    simParas = paras['simulation']
    pbsaParas = paras['GBSA']

    receptorfile = os.path.abspath(receptorfile)
    logging.info('Build protein topology.')
    receptor = build_protein(
        receptorfile, forcefield=simParas['proteinforcefield']
    )

    logging.info(
        'MD: %d ligand(s), sequential gmx with %d thread(s).'
        % (len(ligandfiles), nt)
    )
    set_OMP_NUM_THREADS(nt)
    if 'startframe' not in pbsaParas:
        pbsaParas['startframe'] = 2

    cwd = os.getcwd()
    jobs, failed = [], []
    d = _empty_energy_frame(pbsaParas['modes'])

    for ligandfile in ligandfiles:
        print('=' * 80)
        ligandName, ligandfile = _enter_ligand_dir(ligandfile)
        xtcfile = 'traj_com.xtc'
        logging.info('Build ligand topology: %s' % ligandName)
        try:
            grofile, topfile, indexfile = _build_complex(
                receptor, ligandfile, simParas, nt
            )
        except Exception as e:
            traceback.print_exc()
            logging.warning(
                'Failed to generate forcefield for ligand: %s' % ligandName
            )
            failed.append(_tag_result(d, ligandName, 'F_top'))
            os.chdir(cwd)
            continue

        logging.info('Running simulation: %s' % ligandName)
        engine = GMXEngine()
        try:
            mdgro, mdxtc, outtop = engine.run_to_md(
                grofile,
                topfile,
                nsteps=simParas['nsteps'],
                nframe=simParas['nframe'],
                eqsteps=simParas['eqsteps'],
                **_sim_box(simParas, nt),
            )
            _reres(mdgro, grofile)
            shutil.copy(topfile, outtop)
            shutil.copy(mdxtc, xtcfile)
            if not os.path.exists(indexfile):
                indexfile = generate_index_file(grofile)
        except Exception as e:
            traceback.print_exc()
            logging.warning(
                'Failed to run simulation for ligand: %s' % ligandName
            )
            failed.append(_tag_result(d, ligandName, 'F_md'))
            os.chdir(cwd)
            continue

        jobs.append(_gbsa_job(
            ligandName,
            grofile,
            xtcfile,
            topfile,
            indexfile,
            pbsaParas,
            mmpbsafile,
            verbose,
            receptorfile,
            ligandfile,
            pbsaParas['modes'],
            clean=True,
        ))
        os.chdir(cwd)

    _combine_and_write(
        failed, _run_gbsa_jobs(jobs, nt), outfile,
        ligandfiles=ligandfiles,
    )


def main(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'MM/GB(PB)SA Calculation.  Version: %s' % __version__
        )
    )
    parser.add_argument(
        '-i',
        dest='receptor',
        help='Input protein file in pdb format.',
        required=True,
    )
    parser.add_argument(
        '-l',
        dest='ligand',
        help=(
            'Ligand files to calculate binding energy. '
            'For small molecular, please use format of sdf or mol, '
            'for protein ligand, please use format of pdb.'
        ),
        nargs='+',
        default=None,
    )
    parser.add_argument(
        '-c',
        dest='config',
        help='Config file, default: %s' % DEFAULT_CONFIGURE_FILE,
        default=DEFAULT_CONFIGURE_FILE,
    )
    parser.add_argument(
        '-d',
        dest='ligdir',
        help=(
            'Directory containing many ligand files. '
            'file format: .mol or .sdf'
        ),
        default=None,
    )
    parser.add_argument(
        '-f',
        dest='pbsafile',
        help='gmx_MMPBSA input file. default=None',
        default=None,
    )
    parser.add_argument(
        '-o',
        dest='outfile',
        help='Output CSV file name (written inside --outdir).',
        default='BindingEnergy.csv',
    )
    parser.add_argument(
        '--outdir',
        dest='outdir',
        help='Output directory (like unigbsa-scan -o). default: unigbsa.pipeline',
        default='unigbsa.pipeline',
    )
    parser.add_argument(
        '-validate',
        help='Validate the ligand file. default: False',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '-nt',
        dest='threads',
        help=(
            'GMX uses all threads on one ligand at a time; '
            'then up to this many gmx_MMPBSA instances run in parallel '
            '(1 thread each).'
        ),
        type=int,
        default=multiprocessing.cpu_count(),
    )
    parser.add_argument(
        '--decomp',
        help='Decompose the free energy. default:False',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--verbose',
        help='Keep all the files.',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version='{prog}s ({version})'.format(
            prog='%(prog)', version=__version__
        ),
    )

    args = parser.parse_args(args)
    receptor = os.path.abspath(args.receptor)
    ligands = args.ligand
    conf = os.path.abspath(args.config)
    ligdir = args.ligdir
    outfile = args.outfile
    decomposition = args.decomp
    nt = args.threads
    verbose = args.verbose

    if ligands is None:
        ligands = []
    if ligdir:
        ligdir = os.path.abspath(ligdir)
        for fileName in os.listdir(ligdir):
            if fileName.endswith(('mol', 'sdf')):
                ligands.append(os.path.join(ligdir, fileName))
    ligands = [os.path.abspath(l) for l in ligands]
    if len(ligands) == 0:
        raise Exception('No ligand files found.')

    if not os.path.exists(conf):
        raise Exception('Could not find the config file! %s' % conf)

    mmpbsafile = (
        os.path.abspath(args.pbsafile) if args.pbsafile else args.pbsafile
    )
    paras = load_configue_file(conf)
    gbsa_modes = paras['GBSA']['modes']

    if decomposition:
        paras['GBSA']['modes'] += ',decomposition'
    if '-' in gbsa_modes:
        tmplist = gbsa_modes.split(',')[0].split('-')
        gbtype = tmplist[0]
        gbnum = tmplist[1]
        if gbtype.upper() == 'GB':
            paras['GBSA']['igb'] = gbnum
        elif gbtype.upper() == 'PB':
            paras['GBSA']['ipb'] = gbnum
        if 'decomposition' in gbsa_modes:
            gbtype += ',decomposition'
        paras['GBSA']['modes'] = gbtype

    mode = paras['simulation']['mode']
    runners = {
        'em': minim_pipeline,
        'md': md_pipeline,
        'input': base_pipeline,
    }
    runner = runners.get(mode)
    if runner is None:
        raise Exception('Unknown simulation mode: %s' % mode)
    kw = dict(
        receptorfile=receptor,
        ligandfiles=ligands,
        paras=paras,
        outfile=outfile,
        mmpbsafile=mmpbsafile,
        verbose=verbose,
        nt=nt,
    )
    if runner is not md_pipeline:
        kw['validate'] = args.validate
    logging.info('Output directory: %s' % os.path.abspath(args.outdir))
    with PathManager(args.outdir):
        runner(**kw)


if __name__ == '__main__':
    main()
