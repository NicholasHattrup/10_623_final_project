import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb

def extract(a, t, x_shape):
    """
    This function abstracts away the tedious indexing that would otherwise have
    to be done to properly compute the diffusion equations from lecture. This
    is necessary because we train data in batches, while the math taught in
    lecture only considers a single sample.
    
    To use this function, consider the example
        alpha_t * x
    To compute this in code, we would write
        extract(alpha, t, x.shape) * x

    Args:
        a: 1D tensor containing the value at each time step.
        t: 1D tensor containing a batch of time indices.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Passes the input timesteps through the cosine schedule for the diffusion process
    Args:
        timesteps: 1D tensor containing a batch of time indices.
        s: The strength of the schedule.
    Returns:
        1D tensor of the same shape as timesteps, with the computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)

#https://arxiv.org/abs/2301.10972
def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
     v_start = torch.sigmoid(start/tau) 
     v_end = torch.sigmoid(end/tau) 
     output = torch.sigmoid((t*(end-start)+start) / tau) 
     output = (v_end-output) / (v_end-v_start) 
     return torch.clip(output, clip_min, 1.0)

# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        max_atoms,
        timesteps=1000,
    ):
        super().__init__()
        self.max_atoms = max_atoms
        self.model = model
        self.num_timesteps = int(timesteps)
        self.device = model.encoder.layers[0].self_attn.linears[0].weight.device

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            max_atoms: The max number of atoms model can handle.
                size of model output is (3, max_atoms) flattened
            timesteps: The number of timesteps for the diffusion process.
        """
        self.register_buffer('a', cosine_schedule(self.num_timesteps))
        self.register_buffer('sqrt_a', torch.sqrt(self.a))
        self.register_buffer('a_bar', torch.cumprod(self.a, dim=0))
        self.register_buffer('sqrt_a_bar', torch.sqrt(self.a_bar))
        self.register_buffer('inv_sqrt_a_bar', 1 / self.sqrt_a_bar)
        self.register_buffer('op_sqrt_a_bar', torch.sqrt(1 - self.a_bar))

        # For alpha_bar[0]
        self.register_buffer('a_bar2', torch.cat((torch.tensor([1.0], device=self.a.device), self.a_bar[:-1])))

        self.register_buffer('mu_c0', self.sqrt_a * (1 - self.a_bar2) / (1 - self.a_bar))
        self.register_buffer('mu_c1', torch.sqrt(self.a_bar2) * (1 - self.a) / (1 - self.a_bar))
        self.register_buffer('std', torch.sqrt((1 - self.a) * (1 - self.a_bar2) / (1 - self.a_bar)))

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index, other_features):
        """
        Computes the (t_index)th sample from the (t_index + 1)th sample using
        the reverse diffusion process.
        Args:
            x: The sampled delta at timestep t_index + 1. Shape (B, N, N)
            t: 1D tensor of the index of the time step.
            t_index: Scalar of the index of the time step.
        Returns:
            The sampled delta at timestep t_index. Shape (B, N, N)
        """
        adjacency_matrix, node_features, distance_matrix, _ = other_features
        # Ensure inputs are on the correct device before model call
        adjacency_matrix = adjacency_matrix.to(self.device)
        node_features = node_features.to(self.device)
        distance_matrix = distance_matrix.to(self.device)
        x = x.to(self.device) # Ensure x is also on the correct device
        t = t.to(self.device) # Ensure t is on the correct device

        # distance_matrix is low quality (B, N, N), x is current noise delta (B, N, N)
        distance_matrix_noisy = distance_matrix + x 
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        batch_mask = batch_mask.to(self.device) # Ensure mask is on the correct device
        
        # Predict noise based on noisy distance matrix
        # Ensure all inputs to the model are on self.device
        ep_t = self.model(
            node_features, 
            batch_mask, 
            adjacency_matrix, 
            distance_matrix_noisy, 
            None, # edges_att is None
            t
        ) # Shape (B, N, N)

        # Ensure shapes match
        B, N, _ = x.shape
        if ep_t.shape[-1] != N: # Check based on actual dimension size N
             # Handle potential shape mismatch if Generator padded/sliced differently
             ep_t = ep_t[:, :N, :N] # Slice if needed

        # Calculate x_hat_0 (predicted original delta)
        # tmp = extract(self.op_sqrt_a_bar, t, ep_t.shape)*ep_t # op_sqrt_a_bar needs (B, 1, 1) shape
        tmp = extract(self.op_sqrt_a_bar, t, x.shape) * ep_t # Use x.shape for extract

        # inv_sqrt_a_bar needs (B, 1, 1) shape
        x_hat_0 = extract(self.inv_sqrt_a_bar, t, x.shape) * (x - tmp)
        
        # Clamp predicted delta
        x_hat_0 = torch.clamp(x_hat_0, min=-10.0, max=10.0) # Adjust range if needed based on standardized delta range
        
        # Calculate mean of posterior q(x_{t-1} | x_t, x_0)
        # mu_c0, mu_c1 need (B, 1, 1) shape
        mu_tilda_t = extract(self.mu_c0, t, x.shape)*x + extract(self.mu_c1, t, x.shape)*x_hat_0

        if t_index == 0:
            return mu_tilda_t
        else:
            z = self.noise_like(x.shape, x.device)
            # std needs (B, 1, 1) shape
            return mu_tilda_t + extract(self.std, t, z.shape)*z
        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, delta, other_features):
        """
        Passes noise through the entire reverse diffusion process to generate
        final image samples.
        Args:
            delta: The initial noise that is randomly sampled from the noise distribution.
        Returns:
            The sampled images.
        """
        b = delta.shape[0]
    
        for t in reversed(range(self.num_timesteps)):
            t_full = torch.full((b,), t, device = delta.device, dtype = torch.long)
            delta = self.p_sample(delta, t_full, t, other_features)

        #* DO WE NEED TO DO SOMETHING LIKE THIS??
        # delta = torch.clip(delta, -1.0, 1.0)
        # delta = unnormalize_to_zero_to_one(delta)

        return delta
        # ####################################################

    @torch.no_grad()
    def sample(self, other_features):
        """
        Wrapper function for p_sample_loop.
        Args:
            other_features: Tuple containing low-quality features (adj, nodes, dist, _)
        Returns:
            The sampled refined distance matrix (B, N, N).
        """
