from rdkit import Chem 
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import LBFGS
from ase.calculators.emt import EMT
from mace.calculators import mace_off
from tqdm import tqdm 
import numpy as np
import os
import argparse

def rdkit_mol_to_ase_atoms(mol : "Mol", calculator):
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

def ase_optimize_molecule(mol : Atoms, outpath : os.PathLike, mol_name : str, fmax : float):
    dyn = LBFGS(mol, trajectory = os.path.join(outpath, mol_name + ".traj"))
    return dyn.run(fmax = fmax, steps = 500)

def optimize_molecules(smiles_strs : dict[str,str], outpath : os.PathLike, tol : float):

    calc = mace_off(model="large", device='cuda')

    converge_flags = np.zeros(len(smiles_strs.keys()))

    for i, (name, smiles_str) in tqdm(enumerate(smiles_strs.items())):
        rdkit_molecule = generate_molecule_from_smiles(smiles_str)
        ase_atoms = rdkit_mol_to_ase_atoms(rdkit_molecule, calc)
        converge_flags[i] = ase_optimize_molecule(ase_atoms, outpath, name, tol)

    print(f"{int(len(converge_flags) - sum(converge_flags))} did NOT converge")

    return converge_flags
    

def parse_molecules(path : os.PathLike):
    ## Format:
    ## <Number-of-Atoms>
    ## <Smiles-String>
    ## <Atom-Type> <Coords>
    ## ....

    entries = {}
    
    with open(path, 'r') as f:
        lines = f.read().strip().splitlines()
    
    i = 0
    while i < len(lines):
        # Read number of atoms
        num_atoms = int(lines[i].strip())
        i += 1
        
        # Read SMILES string
        smiles = lines[i].strip()
        i += 1
        
        # Read atom coordinates
        atoms = []
        for _ in range(num_atoms):
            parts = lines[i].split()
            i += 1
            
            element = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            
            atoms.append((element, x, y, z))
        i += 1 # skip empty line
        
        entries[smiles] = atoms
    
    return entries



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", "-d", type = str, required = True)
    parser.add_argument("--outpath", "-o", type = str, required = True)
    parser.add_argument("--tol", "-t", type = float, required = False, default = 0.02)

    args = parser.parse_args()

    # smiles_strs = parse_molecules(args.datapath)

    smiles_strs = {"methyl": "CC#N"}

    converge_flags = optimize_molecules(smiles_strs, args.outpath, args.tol)

    np.savetxt(os.path.join(args.outpath, "convergence_flags.txt"),
                np.column_stack((list(smiles_strs.keys()), converge_flags)),
                fmt="%s", delimiter = ",")

if __name__ == "__main__":
    main()