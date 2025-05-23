from torch.utils.data import Dataset
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import torch

from mdopt import generate_molecule_from_smiles, parse_molecules
from model import featurize_mol


def featurize_low_quality_mols(smiles_str : str):
    mol = generate_molecule_from_smiles(smiles_str)
    if mol is not None:
        mol = Chem.RemoveHs(mol)
        try:
            node_features, adj_matrix, dist_matrix, positions, symbols = featurize_mol(mol, True)
            return smiles_str, node_features, adj_matrix, dist_matrix, positions, symbols
        except Exception as e:
            return smiles_str, None, None, None, None, None
    else:
        return smiles_str, None, None, None, None, None

def main():

    quantum_datapath = "/mnt/mntsdb/genai/10_623_final_project/QCDGE.xyz"
    max_workers = 4

    dft_molecules = parse_molecules(quantum_datapath)
    smiles_strs = dft_molecules.keys()

    low_quality_mol_data = {}
    fails = 0

    torch.multiprocessing.set_sharing_strategy('file_system')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(featurize_low_quality_mols, ss) for ss in smiles_strs]

        for f in tqdm(as_completed(futures), total = len(futures), desc = "Generating Low Quality Molecules"):
            res = f.result()
            smiles_str, node_features, adj_matrix, dist_matrix, positions, symbols = res
            if node_features is not None:
                data = {
                    "node_features" : node_features, 
                    "adj_matrix" : adj_matrix,
                    "dist_matrix": dist_matrix,
                    "positions" : positions,
                    "symbols" : symbols
                }
                low_quality_mol_data[smiles_str] = data
            else:
                fails += 1

    outpath = "/mnt/mntsdb/genai/10_623_final_project/low_quality_features.npz"
    np.savez(outpath, **low_quality_mol_data)
    
    print(f"Failed to generate features for {fails} molecules")

if __name__ == "__main__":
    main()