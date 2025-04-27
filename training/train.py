import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from lightning.fabric import Fabric
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
from dataclasses import dataclass, asdict, field
import torch.nn.functional as F
from tqdm import tqdm
import os
import yaml, pickle
import numpy as np
import wandb
from datetime import date
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from sklearn.metrics import pairwise_distances
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import train_test_split


from wandb.integration.lightning.fabric import WandbLogger

from mdopt import generate_molecule_from_smiles, parse_molecules
from model import make_model, construct_loader, featurize_mol, Diffusion
from mdopt.align import kabsch_weighted_fit


@dataclass
class ModelConfig:
    n_transformer_blocks : int = field(
        default = 8,
        metadata={"description" : "Number of transformer layers"}
    )
    n_heads : int = field(
        default = 16,
        metadata={"description" : "Number of heads in multi-headed attention"}
    )
    d_atom : int = field(
        default = 27, 
        metadata = {"description" : "Atom embed dim, 27 unless you add more atom types"}
    )
    max_atoms : int = field(
        default = 32,
        metadata = {"description" : "max number of atoms model can handle"}
    )
    d_model : int = field(
        default = 256
    )
    dropout : float = field(
        default = 0.1
    )
    timesteps : int = field(
        default = 1000,
        metadata = {"description" : "Number of diffusion timesteps"}
    )


@dataclass 
class TrainConfig:
    quantum_datapath : str = field(
        metadata = {'description' : "Path to xyz file of quantum structures"}
    )
    outpath : str = field(
        metadata = {'description' : "Folder where output directory will be created to save results."}
    )
    features_path : str = field(
        default = None,
        metadata = {'description' : "Path to pre-computed features for dataset"}
    )
    n_epochs : int = field(
        default = 5
    )
    batch_size : int = field(
        default = 8,
    )
    lr : float = field(
        default = 4e-4
    )
    val_size : float = field(
        default = 0.1
    )
    test_size : float = field(
        default = 0.1
    )
    data_loader_workers : int = field(
        default = 8
    )
    checkpoint_interval : int = field(
        default = 5,
        metadata = {'description' : "Should be <= n_epochs"}
    )
    n_devs : int = field(
        default = 1,
        metadata = {'description' : "Number of GPUs to train on"}
    )
    wandb_project : str = field(
        default = '',
        metadata = {'description' : "Name of wandb project, empty string will cause wandb to not be initialized."}
    )
    model_config : ModelConfig = field(
        default_factory = ModelConfig,
    )

    def __post_init__(self) -> None:

        self.model_config = ModelConfig(**self.model_config)

        if self.checkpoint_interval > self.n_epochs:
            self.checkpoint_interval = self.n_epochs

def pos_sym_to_rdkit(symbols, positions):
    rw = Chem.RWMol()
    for s in symbols:
        rw.AddAtom(Chem.Atom(s))          # returns new atom index, not needed here

    # ---- 3.  add a conformer with 3‑D coordinates -------------
    conf = Chem.Conformer(len(symbols))
    for i, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    rw.AddConformer(conf, assignId=True)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol

# Takes two rdkit molecules and calculates difference in their positions
def get_mol_delta(positions, symbols, rdkit_mol):
    rdMolAlign.AlignMol(mol1, mol2)

    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()

    # Check same number of atoms
    if conf1.GetNumAtoms() != conf2.GetNumAtoms():
        raise ValueError("Molecules must have the same number of atoms.")

    # Compute delta
    positions1 = np.array([list(conf1.GetAtomPosition(i)) for i in range(conf1.GetNumAtoms())])
    positions2 = np.array([list(conf2.GetAtomPosition(i)) for i in range(conf2.GetNumAtoms())])

    delta = positions1 - positions2

    return delta


def get_mol_delta_vnick(src_xyz, tgt_xyz):
        src_xyz, rmsd_align = kabsch_weighted_fit(src_xyz, tgt_xyz, return_rmsd=True)
        return src_xyz - tgt_xyz, rmsd_align



def save_split(out_dir, tr_dataset, val_dataset, te_dataset = None):
    with open(os.path.join(out_dir, "training_set.txt"), "w") as f:
        for ss in tr_dataset:
            f.write(f"{ss}\n")
    with open(os.path.join(out_dir, "validation_set.txt"), "w") as f:
        for ss in val_dataset:
            f.write(f"{ss}\n")
    if te_dataset is not None:
        with open(os.path.join(out_dir, "testing_set.txt"), "w") as f:
            for ss in te_dataset:
                f.write(f"{ss}\n")

def train_one_epoch(dataloader, model, optimizer, fabric, cfg, epoch):
    with tqdm(dataloader, unit = "batch", total = len(dataloader)) as bar:
        bar.set_description(f"Epoch [{epoch+1}/{cfg.n_epochs}]")

        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(bar):
            
            # Unpack batch
            *other_features, delta = batch
            # Sample noise
            noise = torch.randn(delta.shape, device = fabric.device)
            # Evaluate Loss
            training_loss = model(delta, other_features, noise)

            fabric.backward(training_loss)

            wandb.log({"training_loss" : training_loss})
            bar.set_postfix(loss=training_loss.item())


def evaluate(cfg, dataloader, model, fabric, dataset : str):

    model.eval()
    loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc = "Evaluation"):
            *other_features, delta = batch
            noise = torch.randn(delta.shape, device = delta.device)
            loss += model(delta, other_features, noise)

    return loss/len(dataloader)

def split_data(x : list, test_size, val_size, seed = 42):

    if not (0.0 < test_size < 1.0 and 0.0 < val_size < 1.0):
        raise ValueError("test_size and val_size must be floats in (0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.")

    train_val, test = train_test_split(
        x,
        test_size=test_size,
        shuffle=True,
        random_state=seed,
    )

    residual_fraction = 1.0 - test_size
    val_fraction_of_residual = val_size / residual_fraction

    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_residual,
        shuffle=True,
        random_state=seed,
    )

    return train, val, test

def load_model(model_config : ModelConfig):

    #* match the keys from MAT with what
    #* I called things in the config
    model_params = {
        "d_atom" : model_config.d_atom,
        "n_output" : model_config.max_atoms**2, # will reshape to distance matrix
        "h" : model_config.n_heads,
        "d_model": model_config.d_model,
        "N_encoder_layers" : model_config.n_transformer_blocks,
        "dropout": model_config.dropout
    }

    return make_model(**model_params)

def work(v):
    return v.to_rdkit_no_H()

def load_quantum_dataset(dft_mol_path : os.PathLike, quantum_datapath : os.PathLike, max_workers = 40):

    if not os.path.isfile(dft_mol_path):
        dft_molecules = parse_molecules(quantum_datapath)

        fails = 0; successes = 0
        dft_molecules_rdkit = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(work, v) : ss for ss, v in dft_molecules.items()}

            for f in tqdm(as_completed(futures), desc = "Quantum --> RDKit", total = len(futures)):
                ss = futures[f]
                res = f.result()
                if res is not None:
                    dft_molecules_rdkit[ss] = res
                    successes += 1
                else:
                    fails += 1

        print(f"Failed to construct geometries for {fails}, succeded for {successes}")

        with open(dft_mol_path, "wb") as f:
            pickle.dump(dft_molecules_rdkit, f)

        return dft_molecules_rdkit
    else:
        with open(dft_mol_path, "rb") as f:
            dft_molecules_rdkit = pickle.load(f)
        return dft_molecules_rdkit


def filter_hydrogen_atoms(feature_tuple):
    """Filter out atoms that have a 1 in the 11th position (index 10)."""
    node_features, adj_matrix, dist_matrix = feature_tuple
    # Find indices where the 11th element (index 10) is NOT 1
    indices_to_keep = [i for i, row in enumerate(node_features) if row[10] != 1] #* HYDROGEN EMBEDDED AS 11th index
    # Filter features
    filtered_features = [node_features[i] for i in indices_to_keep]
    # Filter adjacency matrix
    filtered_adj = []
    for i in indices_to_keep:
        filtered_row = [adj_matrix[i][j] for j in indices_to_keep]
        filtered_adj.append(filtered_row)
    # Filter distance matrix
    filtered_dist = []
    for i in indices_to_keep:
        filtered_row = [dist_matrix[i][j] for j in indices_to_keep]
        filtered_dist.append(filtered_row)
    return (filtered_features, filtered_adj, filtered_dist)


def train(fabric, cfg: TrainConfig, out_dir : str, padding_label = -1, max_workers = 40):

    datapath = os.path.dirname(cfg.features_path)

    # Load Quantum Dataset and convert to RDKit Molecules
    print("LOADING QUANTUM MOLECULES")
    # dft_mol_path = os.path.join(datapath, "dft_molecules.pkl")
    # dft_molecules = load_quantum_dataset(dft_mol_path, cfg.quantum_datapath, max_workers)
    # dft_molecules = parse_molecules(cfg.quantum_datapath)
    with open(cfg.quantum_datapath, "rb") as f:
        dft_dist_matricies = pickle.load(f)
    smiles_strs = dft_dist_matricies.keys()
    print("LOADED QUANTUM MOLECULES")

    print("LOADING LOW-QUALITY MOLECULES")
    with open(cfg.features_path, "rb") as f:
        low_quality_features_with_H = pickle.load(f)
    print("LOADED LOw-QUALITY MOLECULES")

    low_quality_features_noH = {ss : filter_hydrogen_atoms(low_quality_features_with_H[ss]) for ss in tqdm(smiles_strs, desc = "Filtering Hs")}
    low_quality_features = low_quality_features_noH


    #! NO GURANTEE ATOMS IN SAME ORDER ACROSS DATASETS I DONT THINK
    #! SMILES DO WHATEVER THE HELL THEY WANT
    bar = tqdm(smiles_strs, desc = "Calculating Deltas", total = len(smiles_strs))
    deltas = {ss : low_quality_features[ss][-1] - dft_dist_matricies[ss] for ss in bar}

    total_sum   = sum(v.sum()   for v in deltas.values())
    total_count = sum(v.size    for v in deltas.values())
    total_sq_sum   = sum((v**2).sum()    for v in deltas.values())
    delta_mean = total_sum / total_count
    delta_std = np.sqrt((total_sq_sum / total_count) - delta_mean**2)

    print(f"Mean : {delta_mean}")
    print(f"Std : {delta_std}")

    deltas_standardized = {ss : (deltas[ss] - delta_mean) / delta_std for ss in bar}

    # Take intersection of the low quality and quantum molecules
    # smiles_strs = deltas.keys()

    #! some fake data while nick makes the correct data
    # smiles_strs = ["CO", "CC", "CCC", "CCCC"]
    # deltas = {ss : np.random.randn(len(ss), len(ss)) for ss in smiles_strs} #* fake distance matricies

    # Test-Train-Val Split
    train_smiles, val_smiles, test_smiles = split_data(list(smiles_strs), cfg.test_size, cfg.val_size, seed = 42)
    save_split(out_dir, train_smiles, val_smiles, test_smiles)
    print(f"There are training clusters: {len(train_smiles)}")
    print(f"There are validation clusters: {len(val_smiles)}")
    print(f"There are testing clusters: {len(test_smiles)}")

    # Setup data in format MAT expects
    X_train = [low_quality_features[ss]  for ss in train_smiles]
    X_val = [low_quality_features[ss] for ss in val_smiles]
    X_test = [low_quality_features[ss] for ss in test_smiles]
    Y_train = [deltas_standardized[ss] for ss in train_smiles]
    Y_val = [deltas_standardized[ss] for ss in val_smiles]
    Y_test = [deltas_standardized[ss] for ss in test_smiles]

    train_dl = construct_loader(X_train, Y_train, cfg.batch_size)
    val_dl = construct_loader(X_val, Y_val, 1, shuffle = False)
    test_dl = construct_loader(X_test, Y_test, 1, shuffle = False)
    train_dl, val_dl, test_dl = fabric.setup_dataloaders(train_dl, val_dl, test_dl)


    noise_model = load_model(cfg.model_config)
    noise_model.to(fabric.device)
    diffusion_model = Diffusion(
                            noise_model,
                            max_atoms = cfg.model_config.d_atom,
                            timesteps = cfg.model_config.timesteps
                        )
    diffusion_model.to(fabric.device)

    # optimizer = torch.optim.SGD(model.parameters(), lr = cfg.lr)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr = cfg.lr)
    model, optimizer = fabric.setup(diffusion_model, optimizer)


    # for name, p in diffusion_model.named_parameters():
    #     print(name, p.shape)

    for epoch in range(cfg.n_epochs):
        train_one_epoch(train_dl, model, optimizer, fabric, cfg, epoch)
        avg_val_loss = evaluate(cfg, val_dl, model, fabric, "val")
        avg_train_loss = evaluate(cfg, train_dl, model, fabric, "train")

        fabric.log_dict({"avg_val_loss" : avg_val_loss, "avg_train_loss" : avg_train_loss, "epoch" : epoch})

        if epoch > 0 and epoch % (cfg.checkpoint_interval-1) == 0:
            state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss' : avg_val_loss,
                    'dataset' : cfg.quantum_datapath
                    }
            savepath = os.path.join(out_dir, f"checkpoint_{epoch}.pt")
            fabric.save(savepath, state)

def main():
    
    parser = argparse.ArgumentParser(description="Train contact probability model")
    parser.add_argument("-c", "--config", type=str, help="Path to config file", required = True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = TrainConfig(**yaml.safe_load(f))

    today = date.today()
    model_name = f"model-{today.strftime('%Y-%m-%d')}"
    out_dir = os.path.join(config.outpath, model_name)
    os.makedirs(out_dir, exist_ok = True)

    wandb_logger = WandbLogger(log_model="all", save_dir=out_dir, project=config.wandb_project)
    checkpoint_callback = ModelCheckpoint(
                                        monitor="avg_val_loss",
                                        mode="min", 
                                        dirpath = out_dir,
                                        every_n_epochs = config.checkpoint_interval
                                    )

    fabric = Fabric(
                accelerator = "cuda",
                devices = config.n_devs,
                precision = "32",
                loggers=[wandb_logger], 
                callbacks=[checkpoint_callback]
            )

    fabric.launch()
    fabric.log_dict(asdict(config))

    with open(os.path.join(out_dir, "config.yml"), "w") as f:
        yaml.dump(asdict(config), f)

    try:
        train(fabric, config, out_dir)
    except Exception as e:
        os.system(f"rm -rf {out_dir}")
        raise e

if __name__ == "__main__":
    main()


