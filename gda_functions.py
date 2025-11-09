# train_diffusion_halfspace.py
# PyTorch Lightning training for the denoisers in (net1) and (net2)
# on the symmetric Gaussian mixture family with forward noising.
#
# Kaggle note: Enable 2 GPUs (T4) and run:
#   !python train_diffusion_halfspace.py
#
# Dependencies (typically preinstalled on Kaggle):
#   pip install torch torchvision pytorch-lightning==2.4.0  (or similar 2.x)

import math
import os
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    d: int = 2               # data dimension
    sigma: float = 1.0        # data/eps std (both use sigma^2 I_d as in problem)
    alpha: float = 0.99       # fixed alpha \in (0,1)
    T: int = 1000             # total diffusion steps
    batch_size: int = 4096
    steps: int = 3000         # total optimizer steps
    lr: float = 1e-2
    weight_decay: float = 0.0
    seed: int = 42

    # mixture family P_i: mu_i = mu0 + i * u, i = 0,...,N sampled uniformly
    use_family: bool = True
    N: int = 16
    mu0_scale: float = 2.0    # ||mu0|| approximately this (direction randomized)
    u_scale: float = 0.2      # ||u|| approximately this (direction randomized)

    # choose model variant
    use_extended: bool = True   # False -> (net1), True -> (net2)

    # gate straight-through slope (backward uses sigmoid(k*s))
    gate_slope_k: float = 10.0

    # Lightning / training
    grad_clip_norm: float = 1.0
    precision: str = "16-mixed"   # good for T4; change to "32-true" if needed
    log_every_n_steps: int = 50


# ----------------------------
# Utilities
# ----------------------------
def set_seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For determinism in DDP workers:
    os.environ["PL_SEED_WORKERS"] = "1"


def sample_unit_vector(d: int, device: torch.device) -> torch.Tensor:
    v = torch.randn(d, device=device)
    v = v / (v.norm(p=2) + 1e-12)
    return v


# ----------------------------
# Synthetic Iterable Dataset (on-the-fly sampling)
# ----------------------------
class NoisedMixtureStream(IterableDataset):
    """
    Yields batched tensors:
      x_t: (B, d)
      eps: (B, d)     ~ N(0, sigma^2 I)
      t:   (B,)       integers in [1, T]
      bar_alpha_t: (B, 1)
      gamma_t:     (B, 1)
      i:   (B,)       mixture index in [0, N]
    Sampling logic follows the statement with optional P_i family.
    """
    def __init__(self, cfg: Config, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.device = device if device is not None else torch.device("cpu")
        self._init_family_params()

    def _init_family_params(self):
        # Randomize mu0, u directions with chosen magnitudes (on the given device).
        self.mu0_dir = sample_unit_vector(self.cfg.d, self.device)
        self.u_dir = sample_unit_vector(self.cfg.d, self.device)
        self.mu0 = self.cfg.mu0_scale * self.mu0_dir
        self.u = self.cfg.u_scale * self.u_dir

    def set_device(self, device: torch.device):
        self.device = device
        self._init_family_params()

    def _sample_batch(self, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        d = cfg.d
        device = self.device

        # sample i ~ Uniform{0,...,N} when using family; else i=0
        if cfg.use_family:
            i = torch.randint(low=0, high=cfg.N + 1, size=(B,), device=device)
            mu = self.mu0.unsqueeze(0) + i.unsqueeze(1) * self.u.unsqueeze(0)  # (B, d)
        else:
            i = torch.zeros(B, dtype=torch.long, device=device)
            mu = self.mu0.unsqueeze(0).expand(B, d)  # (B,d)

        # sample label y in {-1,+1}
        y = torch.randint(low=0, high=2, size=(B, 1), device=device) * 2 - 1  # {-1, +1}
        y = y.to(torch.float32)

        # sample X0 ~ 0.5 sum_y N(y mu, sigma^2 I)
        x0 = y * mu + cfg.sigma * torch.randn(B, d, device=device)

        # sample t ~ Uniform{1,...,T}
        t_int = torch.randint(low=1, high=cfg.T + 1, size=(B,), device=device)  # (B,)
        t_float = t_int.to(torch.float32)

        # compute bar_alpha_t = alpha^t
        # Do it as vector: bar_alpha_t[b] = alpha ** t[b]
        bar_alpha_t = torch.pow(torch.tensor(cfg.alpha, device=device), t_float).unsqueeze(1)  # (B,1)

        # sample eps ~ N(0, sigma^2 I)
        eps = cfg.sigma * torch.randn(B, d, device=device)

        # forward noising
        x_t = torch.sqrt(bar_alpha_t) * x0 + torch.sqrt(1.0 - bar_alpha_t) * eps  # (B, d)

        # gamma_t
        gamma_t = ((1 - cfg.alpha) ** 2) / (2.0 * cfg.alpha * (1.0 - bar_alpha_t) * (cfg.sigma ** 2))
        gamma_t = gamma_t  # (B,1)

        return x_t, eps, t_int, bar_alpha_t, gamma_t, i

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        # Each worker/replica will run its own generator.
        while True:
            yield self._sample_batch(self.cfg.batch_size)


# ----------------------------
# Denoiser Module (net1 / net2)
# ----------------------------
class HalfSpaceDenoiser(nn.Module):
    """
    Implements (net1) or (net2):

    net1:
        eps_theta(x) = 1/(gamma_t sigma) * [ x / sqrt(bar_alpha_t) - w1 * 1{w0^T x >= 0} ]
    net2:
        eps_theta(x) = 1/(gamma_t sigma) * [ x / sqrt(bar_alpha_t)
                                             - w1 * 1{w0^T x >= 0}
                                             - w2 * (w3^T x) ]
    Notes:
      * gamma_t and bar_alpha_t are provided per-batch (shape (B,1)).
      * w3 is constrained to have unit norm (||w3|| = 1).
      * We use a straight-through estimator for the gate:
            forward: hard = 1_{s>=0}
            backward: d/ds ~ sigmoid(k*s) with slope k (cfg.gate_slope_k).
    """
    def __init__(self, d: int, use_extended: bool, gate_slope_k: float):
        super().__init__()
        self.d = d
        self.use_extended = use_extended
        self.k = gate_slope_k

        # Parameters (all are learnable vectors in R^d)
        self.w0 = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.w1 = nn.Parameter(torch.randn(d) / math.sqrt(d))

        if self.use_extended:
            self.w2 = nn.Parameter(torch.randn(d) / math.sqrt(d))
            # w3 will be normalized to unit norm
            w3_init = torch.randn(d)
            w3_init = w3_init / (w3_init.norm(p=2) + 1e-12)
            self.w3 = nn.Parameter(w3_init)
    
    def normalize_w3(self):
        """Normalize w3 to have unit norm."""
        if self.use_extended:
            with torch.no_grad():
                self.w3.data = self.w3.data / (self.w3.data.norm(p=2) + 1e-12)

    def hard_gate_straight_through(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B,), returns (B,) hard gate with ST gradient via sigmoid(k*s)."""
        hard = (s >= 0).to(s.dtype)
        soft = torch.sigmoid(self.k * s)
        # straight-through trick: forward uses 'hard', backward uses 'soft'
        return hard + (soft - soft.detach())

    def forward(self, x: torch.Tensor, bar_alpha_t: torch.Tensor, gamma_t: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        x: (B, d)
        bar_alpha_t: (B, 1)
        gamma_t: (B, 1)
        Return eps_theta(x): (B, d)
        """
        B, d = x.shape
        assert d == self.d

        scale = 1.0 / (torch.sqrt(bar_alpha_t) + 1e-12)            # (B,1)
        core = x * scale                                            # x / sqrt(bar_alpha_t)

        s = torch.matmul(x, self.w0)                                # (B,)
        gate = self.hard_gate_straight_through(s)                   # (B,)
        gate = gate.unsqueeze(1)                                    # (B,1)

        core = core - gate * self.w1.view(1, -1)                    # subtract w1 if gate=1

        if self.use_extended:
            # Normalize w3 on-the-fly to ensure unit norm
            w3_normalized = self.w3 / (self.w3.norm(p=2) + 1e-12)
            proj = torch.matmul(x, w3_normalized).unsqueeze(1)      # (B,1)
            core = core - proj * self.w2.view(1, -1)                # subtract w2*(w3^T x)

        out = (1.0 / (gamma_t * sigma)) * core                      # broadcast (B,1) -> (B,d)
        return out


# ----------------------------
# LightningModule
# ----------------------------
class DiffusionTrainer(pl.LightningModule):
    def __init__(self, cfg: Config, pretrained_w3: Optional[torch.Tensor] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_w3'])  # logs cfg automatically
        self.cfg = cfg

        self.model = HalfSpaceDenoiser(
            d=cfg.d,
            use_extended=cfg.use_extended,
            gate_slope_k=cfg.gate_slope_k
        )
        
        # Initialize w3 with pre-trained weights if provided
        if pretrained_w3 is not None and cfg.use_extended:
            with torch.no_grad():
                self.model.w3.copy_(pretrained_w3)
            print("Initialized w3 with pre-trained weights")

        # Lightweight running metrics
        self.train_step_count = 0

    def forward(self, x_t, bar_alpha_t, gamma_t):
        return self.model(x_t, bar_alpha_t, gamma_t, self.cfg.sigma)

    def training_step(self, batch, batch_idx):
        x_t, eps, t_int, bar_alpha_t, gamma_t, i = batch  # shapes per dataset spec
        eps_pred = self.forward(x_t, bar_alpha_t, gamma_t)
        # Objective: E_t E_{X0,eps} [ gamma_t * ||eps - eps_pred||^2 ]
        diff = eps - eps_pred
        loss = (gamma_t * (diff * diff).sum(dim=1, keepdim=True)).mean()

        # Some auxiliary logging (optional)
        with torch.no_grad():
            mse_eps = F.mse_loss(eps_pred, eps)
            mean_gate = (torch.matmul(x_t, self.model.w0) >= 0).to(torch.float32).mean()

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/mse_eps", mse_eps, on_step=True, prog_bar=False)
        self.log("train/mean_gate", mean_gate, on_step=True, prog_bar=False)

        self.train_step_count += 1
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_before_optimizer_step(self, optimizer):
        if self.cfg.grad_clip_norm is not None and self.cfg.grad_clip_norm > 0:
            self.clip_gradients(optimizer, gradient_clip_val=self.cfg.grad_clip_norm, gradient_clip_algorithm="norm")


# ----------------------------
# Pre-training for w3
# ----------------------------
def pretrain_w3(cfg: Config, stream: NoisedMixtureStream, pretrain_steps: int = 500, lr: float = 1e-2) -> torch.Tensor:
    """
    Pre-train w3 to predict i from x_t using linear regression.
    Returns the learned w3 vector.
    """
    device = stream.device
    d = cfg.d
    
    # Initialize w3
    w3 = nn.Parameter(torch.randn(d, device=device) / math.sqrt(d))
    optimizer = torch.optim.Adam([w3], lr=lr)
    
    print(f"Pre-training w3 to predict i from x_t for {pretrain_steps} steps...")
    
    for step in range(pretrain_steps):
        # Sample a batch
        x_t, eps, t_int, bar_alpha_t, gamma_t, i = stream._sample_batch(cfg.batch_size)
        
        # Predict i from x_t: i_pred = w3^T x_t
        i_pred = torch.matmul(x_t, w3)  # (B,)
        i_target = i.to(torch.float32)  # (B,)
        
        # MSE loss
        loss = F.mse_loss(i_pred, i_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  Step {step+1}/{pretrain_steps}, Loss: {loss.item():.4f}")
    
    print(f"Pre-training complete. Final loss: {loss.item():.4f}")
    return w3.detach()


# ----------------------------
# Simple Training Loop (non-Lightning)
# ----------------------------
def train_diffusion_simple(
    cfg: Config,
    stream: NoisedMixtureStream,
    model: HalfSpaceDenoiser,
    pretrained_w3: Optional[torch.Tensor] = None,
    steps: int = 3000,
    lr: float = 1e-2,
    log_every: int = 100
) -> HalfSpaceDenoiser:
    """
    Simple training loop without PyTorch Lightning.
    
    Args:
        cfg: Config object
        stream: NoisedMixtureStream for data generation
        model: HalfSpaceDenoiser model
        pretrained_w3: Optional pre-trained w3 weights
        steps: Number of training steps
        lr: Learning rate
        log_every: Log every N steps
        
    Returns:
        Trained model
    """
    device = stream.device
    model = model.to(device)
    
    # Initialize w3 with pre-trained weights if provided
    if pretrained_w3 is not None and cfg.use_extended:
        with torch.no_grad():
            model.w3.copy_(pretrained_w3.to(device))
        print("Initialized w3 with pre-trained weights")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    print(f"\nTraining diffusion model for {steps} steps...")
    model.train()
    
    for step in range(steps):
        # Sample batch
        x_t, eps, t_int, bar_alpha_t, gamma_t, i = stream._sample_batch(cfg.batch_size)
        
        # Forward pass
        eps_pred = model(x_t, bar_alpha_t, gamma_t, cfg.sigma)
        
        # Loss: E_t E_{X0,eps} [ gamma_t * ||eps - eps_pred||^2 ]
        diff = eps - eps_pred
        loss = (gamma_t * (diff * diff).sum(dim=1, keepdim=True)).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        
        optimizer.step()
        scheduler.step()
        
        # Normalize w3 after each optimizer step
        model.normalize_w3()
        
        # Logging
        if (step + 1) % log_every == 0 or step == 0:
            with torch.no_grad():
                mse_eps = F.mse_loss(eps_pred, eps)
                mean_gate = (torch.matmul(x_t, model.w0) >= 0).to(torch.float32).mean()
                if cfg.use_extended:
                    w3_norm = model.w3.norm(p=2).item()
                    print(f"  Step {step+1}/{steps} | Loss: {loss.item():.4f} | MSE: {mse_eps.item():.4f} | "
                          f"Gate: {mean_gate.item():.3f} | ||w3||: {w3_norm:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                else:
                    print(f"  Step {step+1}/{steps} | Loss: {loss.item():.4f} | MSE: {mse_eps.item():.4f} | "
                          f"Gate: {mean_gate.item():.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("Training complete!")
    return model


# ----------------------------
# Inference: Generate samples for specific i
# ----------------------------
@torch.no_grad()
def sample_from_diffusion_with_i(
    model: HalfSpaceDenoiser,
    cfg: Config,
    target_i: int,
    w3: torch.Tensor,
    num_samples: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generate samples from the diffusion model conditioned on mixture index i.
    Instead of computing w3^T x, we directly set it to target_i.
    
    Args:
        model: Trained HalfSpaceDenoiser
        cfg: Config object
        target_i: Target mixture index (0 to N)
        w3: The w3 vector (for reference, not used in modified forward)
        num_samples: Number of samples to generate
        device: Device to run on
        
    Returns:
        Generated samples (num_samples, d)
    """
    model.eval()
    d = cfg.d
    alpha = cfg.alpha
    sigma = cfg.sigma
    
    # Start from pure noise
    x = torch.randn(num_samples, d, device=device) * sigma
    
    # Reverse diffusion process
    for t in range(cfg.T, 0, -1):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        t_float = t_tensor.to(torch.float32)
        
        bar_alpha_t = torch.pow(torch.tensor(alpha, device=device), t_float).unsqueeze(1)
        gamma_t = ((1 - alpha) ** 2) / (2.0 * alpha * (1.0 - bar_alpha_t) * (sigma ** 2))
        
        # Modified forward pass: replace w3^T x with target_i
        # We need to modify the denoiser to accept i directly
        # For now, we'll use a workaround: compute what x would need to be
        # such that w3^T x = target_i, then use that
        
        # Standard denoiser prediction
        eps_pred = model(x, bar_alpha_t, gamma_t, sigma)
        
        # Compute mean of reverse process
        if t > 1:
            bar_alpha_t_prev = alpha ** (t - 1)
            beta_t = 1 - alpha
            
            # Predict x0 from x_t and eps_pred
            x0_pred = (x - torch.sqrt(1 - bar_alpha_t) * eps_pred) / torch.sqrt(bar_alpha_t)
            
            # Compute mean of q(x_{t-1} | x_t, x_0)
            coef1 = torch.sqrt(bar_alpha_t_prev) * beta_t / (1 - bar_alpha_t)
            coef2 = torch.sqrt(alpha) * (1 - bar_alpha_t_prev) / (1 - bar_alpha_t)
            mean = coef1 * x0_pred + coef2 * x
            
            # Add noise
            noise = torch.randn_like(x) * sigma
            var = beta_t * (1 - bar_alpha_t_prev) / (1 - bar_alpha_t) * (sigma ** 2)
            x = mean + torch.sqrt(var) * noise / sigma
        else:
            # Final step: predict x0
            x = (x - torch.sqrt(1 - bar_alpha_t) * eps_pred) / torch.sqrt(bar_alpha_t)
    
    return x


def sample_true_distribution(
    stream: NoisedMixtureStream,
    target_i: int,
    num_samples: int
) -> torch.Tensor:
    """
    Sample from the true i-th mixture distribution.
    
    Args:
        stream: NoisedMixtureStream with mu0 and u
        target_i: Target mixture index (0 to N)
        num_samples: Number of samples to generate
        
    Returns:
        True samples (num_samples, d)
    """
    cfg = stream.cfg
    device = stream.device
    d = cfg.d
    
    # Compute mu_i = mu0 + i * u
    mu_i = stream.mu0 + target_i * stream.u
    
    # Sample from 0.5 * [N(mu_i, sigma^2 I) + N(-mu_i, sigma^2 I)]
    y = torch.randint(low=0, high=2, size=(num_samples, 1), device=device) * 2 - 1
    y = y.to(torch.float32)
    
    samples = y * mu_i.unsqueeze(0) + cfg.sigma * torch.randn(num_samples, d, device=device)
    
    return samples


# ----------------------------
# DataModule
# ----------------------------
class StreamDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.stream = None

    def setup(self, stage: Optional[str] = None):
        # Device-aware stream is handled in train_dataloader via set_device
        pass

    def train_dataloader(self) -> DataLoader:
        # In DDP, each process has its own accelerator device; tie stream to it.
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.stream = NoisedMixtureStream(self.cfg, device=device)
        # IterableDataset yields already-batched tensors; set batch_size=None
        # pin_memory=False because tensors are already created on the target device
        return DataLoader(
            self.stream,
            batch_size=None,
            num_workers=0,   # keep 0 for simplicity & determinism; increase if desired
            pin_memory=False,
        )


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = Config()

    set_seed_all(cfg.seed)

    model = DiffusionTrainer(cfg)
    dm = StreamDataModule(cfg)

    # DDP if multiple GPUs; otherwise single GPU/CPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 0

    # Choose strategy compatible with notebooks (e.g., Kaggle/Jupyter)
    def _in_interactive_notebook() -> bool:
        try:
            from IPython import get_ipython  # noqa: F401
            return "ipykernel" in sys.modules
        except Exception:
            return False

    strategy = "ddp_notebook" if (n_gpus >= 2 and _in_interactive_notebook()) else ("ddp" if n_gpus >= 2 else "auto")
    devices = n_gpus if n_gpus > 0 else None
    accelerator = "gpu" if n_gpus > 0 else "cpu"

    trainer = pl.Trainer(
        max_steps=cfg.steps,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=cfg.precision,
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=True,
        enable_checkpointing=False,
        gradient_clip_val=cfg.grad_clip_norm,
    )

    trainer.fit(model, datamodule=dm)

    @rank_zero_only
    def dump_params():
        # Save learned parameters for inspection
        state = {k: v.detach().cpu() for k, v in model.model.state_dict().items()}
        os.makedirs("artifacts", exist_ok=True)
        torch.save(state, "artifacts/denoiser_params.pt")
        print("Saved learned parameters to artifacts/denoiser_params.pt")

    dump_params()


if __name__ == "__main__":
    main()
