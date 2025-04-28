import numpy as np
import os
import yaml
import torch
from model import Diffusion
from tqdm import tqdm

from training.train import TrainConfig, load_model, load_quantum_dataset
from model import mol_collate_func, construct_dataset

# mean and std used to standardize training data
MU = 2.9927773761658343
STD = 1.4751436743738326

def reconstruct_coords(distances, k: int = 3):

    # 1) square the distances
    D2 = distances**2

    # 2) build the centering matrix J = I - (1/N) 11^T
    N = distances.shape[0]
    I = np.eye(N)
    one = np.ones((N, N)) / N
    J = I - one

    # 3) compute the Gram matrix B = -0.5 * J D2 J
    B = -0.5 * J.dot(D2).dot(J)

    # 4) eigen-decompose B
    eigvals, eigvecs = np.linalg.eigh(B)

    # 5) sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 6) take the top-k, discard any non-positive eigenvalues
    L = np.maximum(eigvals[:k], 0)
    V = eigvecs[:, :k]

    # 7) compute coordinates X = V * sqrt(L)
    X = V * np.sqrt(L[np.newaxis, :])

    return X


def get_samples():
    model_dir = "/mnt/mntsdb/genai/10_623_final_project/training/training_runs/model-2025-04-28_test"
    quantum_datapath = "/mnt/mntsdb/genai/10_623_final_project/dataset/quantum_features.npz"
    tags_path = os.path.join(model_dir, "testing_set.txt")
    config_path = os.path.join(model_dir, "config.yml")
    checkpoint_path = os.path.join(model_dir, "checkpoint_4.pt")

    mol_features, _ = load_quantum_dataset(quantum_datapath)
    print("Loaded Features")

    with open(config_path) as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    test_tags = np.loadtxt(tags_path, dtype = str)
    test_tags = []
    with open(tags_path, "r") as f:
        for line in f:
            test_tags.append(line.strip())


    noise_model = load_model(cfg.model_config)
    noise_model.to("cuda")
    diffusion_model = Diffusion(
                            noise_model,
                            max_atoms = cfg.model_config.d_atom,
                            timesteps = cfg.model_config.timesteps
                        )

    checkpoint = torch.load(checkpoint_path, map_location="cuda")  
    diffusion_model.load_state_dict(checkpoint['model_state_dict'])  
    diffusion_model.eval().to("cuda")

    low_quality_dist_matricies = np.load(cfg.low_qual_datapath, allow_pickle = True) 


    # Sample diffusion process for molecules in test set
    samples = {}
    for ss in tqdm(test_tags[:2]):
        feats, adj, _ = mol_features[ss]

        # Sample and remove padding
        #* FIX DIST MATRIX PASSED HERE TO BE LOW QUALITY??
        batch = construct_dataset([(feats, adj, (low_quality_dist_matricies[ss] - MU) / STD)], [np.empty((0,0))])
        batch_collated = mol_collate_func(batch)
        batch_collated = [b.to("cuda") for b in batch_collated]
        samples[ss] = diffusion_model.sample(batch_collated).cpu().numpy()

    outpath = os.path.join("/mnt/mntsdb/genai/10_623_final_project/dataset/samples.npz")
    np.savez_compressed(outpath, **samples)

def rmsd(x,y):
    return np.sqrt(np.mean(np.square(x-y)))

def calc_rmsd():

    model_dir = "/mnt/mntsdb/genai/10_623_final_project/training/training_runs/model-2025-04-28_test"
    config_path = os.path.join(model_dir, "config.yml")
    
    with open(config_path) as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    mol_features, _ = load_quantum_dataset(cfg.quantum_datapath)
    print("Loaded Features")


    model_samples = np.load("/mnt/mntsdb/genai/10_623_final_project/dataset/samples.npz")

    quantum_dist_matricies = {ss : mol_features[ss][-1] for ss in tqdm(mol_features)}

    rmses = np.array([rmsd((model_samples[ss][0]*STD), quantum_dist_matricies[ss]) for ss in tqdm(model_samples)])

    # print(np.mean(rmses))
    # print(next(iter(model_samples.values())))

    n_atoms = [len(mol_features[ss][0]) for ss in tqdm(model_samples)]

    unique_counts = np.unique(n_atoms)

    rmse_avg = np.array([rmses[n_atoms == na].mean() for na in unique_counts])
    print(rmse_avg)

if __name__ == "__main__":
    get_samples()

    calc_rmsd()