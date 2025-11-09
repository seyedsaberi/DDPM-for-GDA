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
    d: int = 32               # data dimension
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

    def _sample_batch(self, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        d = cfg.d
        device = self.device

        # sample i ~ Uniform{0,...,N} when using family; else i=0
        if cfg.use_family:
            i = torch.randint(low=0, high=cfg.N + 1, size=(B,), device=device)
            mu = self.mu0.unsqueeze(0) + i.unsqueeze(1) * self.u.unsqueeze(0)  # (B, d)
        else:
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

        return x_t, eps, t_int, bar_alpha_t, gamma_t

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
        eps_theta(x) = 1/(gamma_t sigma) * [ x / sqrt(bar_alpha_t) - w1 * 1{w0^T x >= 0} - b ]
    net2:
        eps_theta(x) = 1/(gamma_t sigma) * [ x / sqrt(bar_alpha_t)
                                             - w1 * 1{w0^T x >= 0}
                                             - w2 * (w3^T x)
                                             - b ]
    Notes:
      * gamma_t and bar_alpha_t are provided per-batch (shape (B,1)).
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
        self.b  = nn.Parameter(torch.zeros(d))

        if self.use_extended:
            self.w2 = nn.Parameter(torch.randn(d) / math.sqrt(d))
            self.w3 = nn.Parameter(torch.randn(d) / math.sqrt(d))

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
            proj = torch.matmul(x, self.w3).unsqueeze(1)            # (B,1)
            core = core - proj * self.w2.view(1, -1)                # subtract w2*(w3^T x)

        core = core - self.b.view(1, -1)

        out = (1.0 / (gamma_t * sigma)) * core                      # broadcast (B,1) -> (B,d)
        return out


# ----------------------------
# LightningModule
# ----------------------------
class DiffusionTrainer(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters()  # logs cfg automatically
        self.cfg = cfg

        self.model = HalfSpaceDenoiser(
            d=cfg.d,
            use_extended=cfg.use_extended,
            gate_slope_k=cfg.gate_slope_k
        )

        # Lightweight running metrics
        self.train_step_count = 0

    def forward(self, x_t, bar_alpha_t, gamma_t):
        return self.model(x_t, bar_alpha_t, gamma_t, self.cfg.sigma)

    def training_step(self, batch, batch_idx):
        x_t, eps, t_int, bar_alpha_t, gamma_t = batch  # shapes per dataset spec
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
        return DataLoader(
            self.stream,
            batch_size=None,
            num_workers=0,   # keep 0 for simplicity & determinism; increase if desired
            pin_memory=torch.cuda.is_available(),
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
