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
import yaml
import numpy as np
import wandb
from datetime import date
from rdkit import Chem
from sklearn.model_selection import train_test_split


from wandb.integration.lightning.fabric import WandbLogger

from mdopt import generate_molecule_from_smiles, parse_molecules
from model import make_model, construct_loader, featurize_mol


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
        default = 64, 
        metadata = {"description" : "Maximum number of atoms in a molecule model can accept"}
    )
    d_model : int = field(
        default = 256
    )
    dropout : float = field(
        default = 0.1
    )


@dataclass 
class TrainConfig:
    datapath : str = field(
        metadata = {'description' : "Output path of create_dataset.py"}
    )
    outpath : str = field(
        metadata = {'description' : "Folder where output directory will be created to save results."}
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

def train_one_epoch(dataloader, criterion, model, optimizer, fabric, cfg, epoch):
    with tqdm(dataloader, unit = "batch", total = len(dataloader)) as bar:
        bar.set_description(f"Epoch [{epoch+1}/{cfg.n_epochs}]")

        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(bar):
            adjacency_matrix, node_features, distance_matrix, _ = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

            pred_delta = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)

            #* Need to update this to mimic the HW2 training loop with noise etc.
            training_loss = criterion(...)

            fabric.backward(training_loss)

            wandb.log({"training_loss" : training_loss})

def evaluate(cfg, dataloader, criterion, model, fabric, dataset : str):


    model.eval()
    loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc = "Evaluation"):
            adjacency_matrix, node_features, distance_matrix, _ = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

            #* ALSO NEED TO UPDATE THIS TO BE LIKE DIFFUSION IN HW2
            pred_delta = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
            

            loss += criterion(predicted_coords, quantum_coords)

            #* CALCULATE RMSE FOR WHOLE TRAJECTORY??


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

    model_params = {
        "d_atom" : model_config.d_atom,
        "n_output" : 3*model_config.d_atom, # x, y, z coords for each
        "h" : model_config.n_heads,
        "d_model": model_config.d_model,
        "N" : model_config.n_transformer_blocks,
        "dropout": model_config.dropout
    }

    return make_model(**model_params)

def train(fabric, cfg: TrainConfig, out_dir : str, padding_label = -1):

    print("LOADING MOLECULES")

    # Load Quantum Dataset
    dft_molecules = parse_molecules(cfg.datapath)
    smiles_strs = dft_molecules.keys()

    print("LOADED MOLECULES")

    # Generate RDKit Molecules with ETKDGv3
    low_quality_mols = {ss : Chem.RemoveHs(generate_molecule_from_smiles(ss)) for ss in tqdm(smiles_strs, desc = "Low-Quality Structures")}


    # Calculate Molecule Features (atomic number, number of neighbors, number of hydrogens, formal charge)
    # featurize_mol returns tuple of (node_features, adj_matrix, distance_matrix)
    low_quality_mol_features = {ss : featurize_mol(lqm, False, True) for ss,lqm in tqdm(low_quality_mols.items(), desc = "Featurizing")}

    # Test-Train-Val Split
    train_smiles, val_smiles, test_smiles = split_data(smiles_strs, cfg.test_size, cfg.val_size, seed = 42)
    save_split(out_dir, train_smiles, val_smiles, test_smiles)
    print(f"There are training clusters: {len(train_smiles)}")
    print(f"There are validation clusters: {len(val_smiles)}")
    print(f"There are testing clusters: {len(test_smiles)}")

    # Setup data in format MAT expects
    X_train = [low_quality_mol_features[ss] for ss in train_smiles]
    X_val = [low_quality_mol_features[ss] for ss in val_smiles]
    X_test = [low_quality_mol_features[ss] for ss in test_smiles]
    Y_train = [dft_molecules[ss] for ss in train_smiles]
    Y_val = [dft_molecules[ss] for ss in val_smiles]
    Y_test = [dft_molecules[ss] for ss in test_smiles]

    train_dl = construct_loader(X_train, Y_train, cfg.batch_size)
    val_dl = construct_loader(X_val, Y_val, 1, shuffle = False)
    train_dl = construct_loader(X_test, Y_test, 1, shuffle = False)
    train_dl, val_dl, test_dl = fabric.setup_dataloaders(train_dl, val_dl, test_dl)


    model = load_model(cfg.model_config)

    criterion = None #TODO IMPLEMENT DIFFUSION LOSS

    # optimizer = torch.optim.SGD(model.parameters(), lr = cfg.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    model, optimizer = fabric.setup(model, optimizer)


    for epoch in range(cfg.n_epochs):
        train_one_epoch(train_dl, criterion, model, optimizer, fabric, cfg, epoch)
        avg_val_loss, avg_metrics = evaluate(cfg, val_dl, criterion, model, fabric, "val")
        avg_train_loss, avg_train_metrics = evaluate(cfg, train_dl, criterion, model, fabric, "train")

        fabric.log_dict({"avg_val_loss" : avg_val_loss, "avg_train_loss" : avg_train_loss, "epoch" : epoch})
        fabric.log_dict(avg_metrics)
        fabric.log_dict(avg_train_metrics)

        if epoch > 0 and epoch % (cfg.checkpoint_interval-1) == 0:
            state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss' : avg_val_loss,
                    'dataset' : cfg.datapath
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


