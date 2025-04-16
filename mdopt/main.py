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

class MyMolecule:

    def __init__(self, symbols : list[str], positions : np.ndarray):
        self.symbols = symbols
        self.positions = positions
    
    def rmse(self, other : "MyMolecule"):
        return np.sqrt(np.mean(np.square(self.position - other.position)))
    
    def to_ase(self, calculator):
        return Atoms(symbols=self.symbols, positions=self.positions, calculator = calculator)


def rdkit_mol_to_ase_atoms(mol, calculator):
    symbols = []
    coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = mol.GetConformer().GetAtomPosition(i)
        symbols.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])
    
    ase_mol = Atoms(symbols=symbols, positions=coords, calculator = calculator)
    my_mol = MyMolecule(symbols, np.array(coords))
    return ase_mol, my_mol

def generate_molecule_from_smiles(smiles_str : str):
    params = AllChem.ETKDGv3()
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles_str))
    AllChem.EmbedMolecule(mol, params)
    return mol

def ase_optimize_molecule(
        mol : Atoms,
        outpath : os.PathLike,
        mol_name : str, 
        fmax : float, 
        maxsteps : int
    ):
    dyn = LBFGS(mol, trajectory = os.path.join(outpath, mol_name + ".traj"))
    return dyn.run(fmax = fmax, steps = maxsteps), mol

def optimize_molecules(
        dft_molecules : dict[str,str],
        outpath : os.PathLike, 
        tol : float, 
        maxsteps : int
    ):

    calc = mace_off(model="large", device='cuda')

    # <Converged?> <Initial RMSE from DFT> <Final RMSE from DFT>
    out_data = np.zeros((len(dft_molecules.keys()), 3))

    for i, smiles_str in tqdm(enumerate(dft_molecules.keys())):
        rdkit_molecule = generate_molecule_from_smiles(smiles_str)
        ase_atoms, my_mol = rdkit_mol_to_ase_atoms(rdkit_molecule, calc)
        initial_rmse = my_mol.rmse(dft_molecules[smiles_str])
        converged, optmized_mol = ase_optimize_molecule(ase_atoms, outpath, smiles_str, tol, maxsteps)
        final_rmse = optmized_mol.rmse(dft_molecules[smiles_str])

        out_data[i,:] = [converged, initial_rmse, final_rmse]

    print(f"{int(len(out_data[:,0]) - sum(out_data[:,0]))} did NOT converge")

    return out_data
    

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
        symbols = []
        coords = []
        for _ in range(num_atoms):
            parts = lines[i].split()
            i += 1
            
            element = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            symbols.append(element)
            coords.append([x,y,z])
        i += 1 # skip empty line
        
        entries[smiles] = MyMolecule(symbols, np.array(coords))
    
    return entries



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", "-d", type = str, required = True)
    parser.add_argument("--outpath", "-o", type = str, required = True)
    parser.add_argument("--tol", "-t", type = float, required = False, default = 1e-3)
    parser.add_argument("--maxsteps", "-ms", type = int, required = False, default = 1000)

    args = parser.parse_args()

    dft_molecules = parse_molecules(args.datapath) # {smiles : structure}

    out_data = optimize_molecules(dft_molecules, args.outpath, args.tol, args.maxsteps)

    np.savetxt(
        os.path.join(args.outpath, "optimization_stats.txt"),
        np.column_stack((list(dft_molecules.keys()), out_data)),
        fmt="%s", delimiter = ",", header = "Converged?, Initial RMSE, Final RMSE"
    )

if __name__ == "__main__":
    main()