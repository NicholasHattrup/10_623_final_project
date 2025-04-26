from torch.utils.data import Dataset
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from rdkit import Chem
import numpy as np

from mdopt import generate_molecule_from_smiles, parse_molecules
from model import featurize_mol


def featurize_low_quality_mols(smiles_str : str):
    mol = generate_molecule_from_smiles(smiles_str)
    if mol is not None:
        node_features, adj_matrix, dist_matrix, positions, symbols = featurize_mol(mol, True)
        return smiles_str, node_features, adj_matrix, dist_matrix, positions, symbols
    else:
        return smiles_str, None, None, None, None, None

def main():

    quantum_datapath = "/mnt/mntsdb/genai/10_623_final_project/QCDGE.xyz"
    max_workers = 40

    dft_molecules = parse_molecules(quantum_datapath)
    smiles_strs = dft_molecules.keys()

    low_quality_mol_data = {}
    fails = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(featurize_low_quality_mols, ss) for ss in smiles_strs]

        for future in tqdm(as_completed(futures), total = len(futures), desc = "Generating Low Quality Molecules"):
            smiles_str, node_features, adj_matrix, dist_matrix, positions, symbols = future.result()
            if node_features is not None:
                data = {
                    "node_features" : node_features, 
                    "adj_matrix" : adj_matrix,
                    "dist_matrix": dist_matrix,
                    "positions" : positions,
                    "symbols" : symbols
                }
                low_quality_mol_data[ss] = data
            else:
                fails += 1

    outpath = os.path.base
    np.savez(os.path.join(outpath, "low_quality_features.npz"), **low_quality_mol_data)
    
    print(f"Failed to generate features for {fails} molecules")

if __name__ == "__main__":
    main()