from rdkit import Chem 
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import LBFGS
from fairchem.core import OCPCalculator
from tqdm import tqdm 
import os
import argparse

def rdkit_mol_to_ase_atoms(mol : "Mol", calculator : OCPCalculator):
    symbols = []
    coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = mol.GetConformer().GetAtomPosition(i)
        symbols.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])
    
    ase_mol = Atoms(symbols=symbols, positions=coords, calculator = calculator)
    return ase_mol

def generate_molecule_from_smiles(smiles_str : str):
    params = AllChem.ETKDGv3()
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles_str))
    AllChem.EmbedMolecule(mol, params)
    return mol

def ase_optimize_molecule(mol : Atoms, outpath : os.PathLike, mol_name : str):
    dyn = LBFGS(mol, trajectory = os.path.join(outpath, mol_name + ".traj"))
    dyn.run(fmax = 0.05, steps = 500)

def optimize_molecules(smiles_strs : dict[str,str], outpath : os.PathLike):

    calc = OCPCalculator(
        model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
        local_cache="pretrained_models",
        cpu=False,
    )

    for name, smiles_str in tqdm(smiles_strs.items()):
        rdkit_molecule = generate_molecule_from_smiles(smiles_str)
        ase_atoms = rdkit_mol_to_ase_atoms(rdkit_molecule, calc)
        ase_optimize_molecule(ase_atoms, outpath, name)
    

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--outpath", "-o", type = str, required = True)

    args = parser.parse_args()


    smiles_strs = {"methylcyanide" : "CC#N"}

    optimize_molecules(smiles_strs, args.outpath)

if __name__ == "__main__":
    main()