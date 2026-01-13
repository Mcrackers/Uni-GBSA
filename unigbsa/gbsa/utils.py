import os
from unigbsa.settings import GMXEXE
def set_amber_home(proc):
    """
    Find the directory containing the executable for a command
  
    Args:
      proc: The name of the executable to find.
  
    Returns:
      the path to the amber home directory.
    """
    cmd = 'which %s '%proc
    f = os.popen(cmd)
    text = f.read().strip()
    if not text:
        raise Exception("Command not found: %s "%proc)
    bindir = os.path.split(text)[0]
    amberhome = os.path.split(bindir)[0]
    return amberhome


def obtain_num_of_frame(trajfile):
    """
    Get the number of frames in a trajectory file

    Args:
      trajfile: the trajectory file, in .xtc or .trr format.

    Returns:
      the number of frames in the trajectory file.
    """
    cmd = '%s check -f %s 2>&1 |grep Coords'% (GMXEXE, trajfile)
    fr = os.popen(cmd)
    text = fr.read().strip()
    if not text:
        print(cmd)
        raise Exception("ERROR obtain %s's frame number.")
    nframe = int(text.split()[1])
    return nframe

def mapping_resname(recfile, ligfile, complexfile):
    reskey0 = []
    records = ('ATOM', 'HETATOM')
    
    if ligfile.lower().endswith('.pdb'):
        prev_reskey = None
        with open(ligfile) as fl:
            for line in fl:
                if line.startswith(records):
                    resname = line[17:20].strip()
                    resid = line[22:27].strip()
                    chainID = line[21].strip()
                    key = f'L:{chainID}:{resname}:{resid}'
                    if key != prev_reskey:
                        reskey0.append(key)
                        prev_reskey = key
    else:
        reskey0.append("L::MOL:1")

    prev_reskey = None
    with open(recfile) as fr:
        for line in fr:
            if line.startswith(records):
                resname = line[17:20].strip()
                resid = line[22:27].strip()
                chainID = line[21].strip()
                key = f'R:{chainID}:{resname}:{resid}'
                if key != prev_reskey:
                    reskey0.append(key)
                    prev_reskey = key

    reskeydic = {}
    index = 0
    prev_reskey = None
    
    with open(complexfile) as fr:
        for line in fr:
            if line.startswith(records):
                resname = line[17:20].strip()
                resid = line[22:27].strip()
                chainID = line[21].strip()
                
                current_key = f'C:{chainID}:{resname}:{resid}'
                

                if current_key != prev_reskey:
                    if index >= len(reskey0):
                        print(f"Warning: Complex file has more residues than Ligand + Receptor combined.\n"
                                         f"Stopped at Complex residue: {resname} {resid} (Index {index})")
                        break

                    source_key = reskey0[index]
                    
                    key_type = source_key[0]
                    dict_key = f'{key_type}:{chainID}:{resname}:{resid}'
                    source_parts = source_key.split(':')
                    source_resname = source_parts[2]
                    
                    if source_resname != resname:
                        raise ValueError(f"Residue Mismatch at index {index}!\n"
                                         f"  Expected (Source): {source_resname} (from {source_parts[0]} chain {source_parts[1]})\n"
                                         f"  Found (Complex):   {resname} (chain {chainID}, resid {resid})\n"
                                         f"Possible causes:\n"
                                         f"  1. Ligand/Receptor order is reversed.\n"
                                         f"  2. Non-PDB ligand name ('MOL') does not match complex ('{resname}').\n"
                                         f"  3. Files are not derived from each other.")

                    reskeydic[dict_key] = source_key
                    index += 1
                    prev_reskey = current_key
    
    if index < len(reskey0):
        print(f"Warning: Complex file has fewer residues ({index}) than source files ({len(reskey0)}). "
              f"Truncated?")

    return reskeydic
