import torch
from torch.utils.data import DataLoader, Dataset
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
import pickle
from functools import partial
from typing import List, Dict
from pathlib import Path
import warnings
from datetime import date
import h5py
from sklearn.model_selection import train_test_split


from wandb.integration.lightning.fabric import WandbLogger


@dataclass
class ModelConfig:
    n_layers : int = field(
        metadata={"description" : "Number of transformer layers"}
    ),
    n_heads : int = field(
        metadata={"description" : "Number of heads in multi-headed attention"}
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
        default = 1,
        metadata = {'description' : "Model uses lots of RAM, recomended to use gradient accumulation to increase effective batch size."}
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
    context_length : int = field(
        default = 1024,
    )
    model_config : ModelConfig = field(
        default_factory = ModelConfig,
    )

    def __post_init__(self) -> None:

        self.model_config = ModelConfig(**self.model_config)

        if self.checkpoint_interval > self.n_epochs:
            self.checkpoint_interval = self.n_epochs

class ClusteredHDF5Dataset(Dataset):
    def __init__(
        self,
        file_path: Path | str,
        clusters: list[SequenceCluster],
        label_type : str
    ) -> None:
        """A PyTorch Dataset for loading data from an HDF5 file according to any sampling strategy.
           This returns the untokenized protein and its associated label

        Parameters
        ----------
        file_path : Path | str
            The path to the HDF5 file containing the data (keys are protein tags)
        clusters : list[SequenceCluster]
            A list of sequences clustered with MMSeqs. Contains the tag and sequence data
        label_type : str
            Which label to expect in the dataset (e.g. distogram)
        """
        self.file_path = file_path
        self.clusters = clusters
        self.label_type = label_type

    @property
    def h5_data(self) -> h5py.File:
        """Lazy load the h5 file in the dataloader worker process."""
        if not hasattr(self, "_h5_data"):
            self._h5_data = h5py.File(self.file_path, "r") # should be closed by upon garbage collection
        return self._h5_data

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        # Get random sample from the idx'th cluster
        seq_like = np.random.choice(self.clusters[idx])

        group = self._h5_data[seq_like.tag]
        # seq = group["sequence"][()] # should be same as seq_like.sequence

        label = torch.from_numpy(group[self.label_type][:])

        prot = Protein(tag = seq_like.tag, sequence = seq_like.sequence, 
                        disorder = group["iupred3_disorder_score"][:],
                        disorder_binding = group["anchor2_disorder_binding_score"][:], 
                        hydropathy = group["hydropathy"][:]
                    )

        return prot, label

class BatchConverter:

    """
    Converts a batch from the DataLoader to the proper dimensions by padding
    or truncating existing data. At train time, if a sequence is too long a random subsequence
    is taken instead of just truncating or ignoring that sequence. At inference time,
    if a sequence is too long it is simply truncated to fit the context window.
    """ 

    def __init__(self, tokenizer, tracks : List[str], padding_label = -1):
        self.tokenizer = tokenizer
        self.tracks = tracks
        self.padding_label = padding_label

    @property
    def append_eos(self):
        return self.tokenizer.append_eos
    
    @property
    def prepend_bos(self):
        return self.tokenizer.prepend_bos


    def _maybe_get_slices(self, data_len, sample_length):

        if data_len > sample_length:
            start_index = np.random.randint(0, high = data_len - sample_length)
            end_index = start_index + sample_length
            return slice(start_index, end_index)

        return slice(0, data_len)


    # Raw batch as returned from DataLoader
    # training indicates if training or inference
    def __call__(self, raw_batch, training : bool = False):

        proteins, labels = zip(*raw_batch)

        max_prot_len = max([len(p) for p in proteins])
        padded_len = max_prot_len + self.append_eos + self.prepend_bos

        if padded_len > self.tokenizer.max_tokenized_length:
            padded_len = self.tokenizer.max_tokenized_length


        # Sample a smaller sequence, ensuring there is always room
        # for the EOS/BOS tokens if needed. This technically loses you 
        # 2 tokens if you sample from the middle but its easier than
        # handling all the edge cases
        sub_sample_length = padded_len - self.append_eos - self.prepend_bos

        # During training sample a random subsequence
        if training:
            slices = [self._maybe_get_slices(len(p), sub_sample_length) for p in proteins]
        else:
            slices = [slice(0, sub_sample_length) for p in proteins]

        prots_slices = zip(proteins, slices)
        label_slices = zip(labels, slices)

        result = []
        for l, s in label_slices:
            slice_tensor = l[s,s]
            padded = F.pad(slice_tensor, 
                        (0, sub_sample_length - slice_tensor.shape[1], 
                        0, sub_sample_length - slice_tensor.shape[0]), 
                        value = self.padding_label)
            result.append(padded)
        labels_padded = torch.stack(result)

        features = {}
        for track in self.tracks:

            if track in self.tokenizer.keys():
                t = self.tokenizer[track]
                toks = [t.encode(p.get_track(track)[s], prepend_bos = s.start == 0 and self.prepend_bos, \
                                                        append_eos = s.stop >= len(p) and self.append_eos) for p,s in prots_slices]

                features[track+"_tokens"] = pad_sequence(toks, batch_first = True, padding_value = self.tokenizer.pad_idx)
            else:
                # Add dummy spacing for bos/eos to align with other tracks
                data_with_dummy = [F.pad(p.get_track(track)[s],(s.start == 0 and self.prepend_bos, s.stop == len(p) and self.append_eos), value = self.tokenizer.pad_idx)
                                        for p,s in prot_slices]
                # Pad to same lengths
                features[track] = pad_sequence(data_with_dummy, batch_first = True, padding_value = self.tokenizer.pad_idx)

        return features, labels_padded

def collate_batch(batch, bc : BatchConverter):
    return bc(batch, training = True)

def collate_single(batch, bc : BatchConverter):
    features, labels = collate_batch(batch, bc)
    return features.squeeze(0), labels.squeeze(0)

def save_split(out_dir, tr_dataset, val_dataset, te_dataset = None):
    with open(os.path.join(out_dir, "training_set.txt"), "w") as f:
        for cluster in tr_dataset.clusters:
            f.write("\t".join([p.tag for p in cluster.data]) + "\n")
    with open(os.path.join(out_dir, "validation_set.txt"), "w") as f:
        for cluster in val_dataset.clusters:
            f.write("\t".join([p.tag for p in cluster.data]) + "\n")
    if te_dataset is not None:
        with open(os.path.join(out_dir, "testing_set.txt"), "w") as f:
            for cluster in te_dataset.clusters:
                f.write("\t".join([p.tag for p in cluster.data]) + "\n")

def train_one_epoch(dataloader, criterion, model, optimizer, fabric, cfg, epoch):
    with tqdm(dataloader, unit = "batch", total = len(dataloader)) as bar:
        bar.set_description(f"Epoch [{epoch+1}/{cfg.n_epochs}]")

        model.train()
        optimizer.zero_grad()
        for i, (toks, contact_probs_gpu) in enumerate(bar):

            toks_gpu = {k: v for k, v in toks.items()} #.to(fabric.device)
            #contact_probs_gpu = contact_probs#.to(fabric.device) # padding is -1 per collate_batch
            contact_prob_preds = model(**toks_gpu) # (batch, seq_len, seq_len)
    
            training_loss = criterion(contact_prob_preds, contact_probs_gpu) / cfg.grad_accumulation_steps

            fabric.backward(training_loss)

            if ((i + 1) % cfg.grad_accumulation_steps == 0) or (i + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            wandb.log({"training_loss" : training_loss})

def evaluate(cfg, dataloader, criterion, model, fabric, dataset : str):


    model.eval()
    loss = 0.0
    with torch.no_grad():
        for (smiles_strs, quantum_coords) in tqdm(dataloader, desc = "Evaluation"):
            B = quantum_coords.shape[0]
            predicted_coords = model(smiles_strs)

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

def train(fabric, cfg: TrainConfig, out_dir : str, padding_label = -1):

    
    # 1. Load Ground-Truth Dataset
        #TODO
        dataset = 
    # 2. Parse out smiles strings and generate "low-quality" starting point
        #TODO
        smiles_strs = 
    # 3. Test Train Split (use constant random seed)
    train_data, val_data, test_data = split_data(smiles_strs, cfg.test_size, cfg.val_size, seed = 42)

    train_dataset = ClusteredHDF5Dataset(train_data)
    val_dataset = ClusteredHDF5Dataset(val_data)
    test_dataset = ClusteredHDF5Dataset(test_data)

    # Copy tags used in train, val, test datasets to outdir
    save_split(out_dir, train_dataset, val_dataset, test_dataset) #* NEED TO UPDATE NOT USING CLUSTERS

    print(f"There are training clusters: {len(train_dataset)}")
    print(f"There are validation clusters: {len(val_dataset)}")
    print(f"There are testing clusters: {len(test_dataset)}")


    model = DeltaCoordModel(cfg.model_config)

    criterion = #TODO IMPLEMENT DIFFUSION LOSS

    # optimizer = torch.optim.SGD(model.parameters(), lr = cfg.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    model, optimizer = fabric.setup(model, optimizer)


    collate_fn = partial(collate_batch, tokenizer = tokenizer,padding_label = padding_label)
    train_dl = DataLoader(train_dataset, batch_size = cfg.batch_size,
                         shuffle = True, collate_fn = collate_fn,
                         num_workers = cfg.data_loader_workers, pin_memory = True)
    val_dl = DataLoader(val_dataset, batch_size = 1, shuffle = False, collate_fn = collate_single)
    test_dl = DataLoader(test_dataset, batch_size = 1, shuffle = False, collate_fn = collate_single)
    train_dl, val_dl, test_dl = fabric.setup_dataloaders(train_dl, val_dl, test_dl)

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
        config = ContactTrainConfig(**yaml.safe_load(f))

    today = date.today()
    model_name = f"{config.esm_str}-{today.strftime('%Y-%m-%d')}"
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
                precision = config.model_config.precision,
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


