import numpy as np
import os
import yaml
import torch
from model import Diffusion
from tqdm import tqdm

from training.train import TrainConfig, load_model, load_quantum_dataset
from model import mol_collate_func, construct_dataset

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


def main():
    model_dir = "/mnt/mntsdb/genai/10_623_final_project/training/training_runs/model-2025-04-27"
    quantum_datapath = "/mnt/mntsdb/genai/10_623_final_project/dataset/quantum_features.npz"
    tags_path = os.path.join(model_dir, "testing_set.txt")
    config_path = os.path.join(model_dir, "config.yml")
    checkpoint_path = os.path.join(model_dir, "checkpoint_4.pt")

    mol_features, _ = load_quantum_dataset(quantum_datapath)
    print("Loaded Features")

    with open(config_path) as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    test_tags = np.loadtxt(tags_path, dtype = str)

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

    # Sample diffusion process for molecules in test set
    samples = {}
    for ss in tqdm(test_tags):
        features = mol_features[ss]
        n_at = len(features[0])
        # Sample and remove padding
        batch = mol_collate_func(construct_dataset(features, None))
        samples[ss] = diffusion_model.sample(10, batch)[:, :n_at, :n_at]

    outpath = os.path.join("/mnt/mntsdb/genai/10_623_final_project/dataset/samples.npz")
    np.savez_compressed(outpath, **samples)

main()