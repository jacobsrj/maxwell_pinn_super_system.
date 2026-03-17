"""
models.py — Neural network architectures for PINNs
===================================================
Three architectures, each targeting a different failure mode:

  StandardPINN  — exact notebook architecture (5×128 cosine layers)
  FourierPINN   — random Fourier feature encoding (fixes spectral bias)
  ResNetPINN    — skip connections every 2 layers (fixes vanishing grad)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS (18 smooth C∞ activations)
# ════════════════════════════════════════════════════════════

def _sinc(x: torch.Tensor) -> torch.Tensor:
    eps = (x.abs() < 1e-7).to(x.dtype) * 1e-7
    return torch.sin(math.pi * (x + eps)) / (math.pi * (x + eps))


ACTIVATIONS: dict = {
    "cos":        torch.cos,
    "sin":        torch.sin,
    "sincos":     lambda x: torch.sin(x) + torch.cos(x),
    "sin2x":      lambda x: torch.sin(2 * x),
    "tanh":       torch.tanh,
    "swish":      lambda x: x * torch.sigmoid(x),
    "gelu":       lambda x: x * torch.sigmoid(1.702 * x),
    "erf":        torch.erf,
    "softplus":   F.softplus,
    "morlet":     lambda x: torch.cos(5 * x) * torch.exp(-0.5 * x * x),
    "sinc":       _sinc,
    "damped_sin": lambda x: torch.sin(x) / (1 + x * x),
    "mish":       lambda x: x * torch.tanh(F.softplus(x)),
    "isru":       lambda x: x / torch.sqrt(1 + x * x),
    "gaussian":   lambda x: torch.exp(-x * x),
    "mex_hat":    lambda x: (1 - x * x) * torch.exp(-0.5 * x * x),
    "selu":       F.selu,
    "elu":        F.elu,
}

# Activations that benefit from SIREN-style initialization
SIREN_ACTS = {"cos", "sin", "sincos", "sin2x"}


class _ActModule(nn.Module):
    """Wraps a callable activation as an nn.Module."""
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x)


def get_act_module(name: str) -> _ActModule:
    fn = ACTIVATIONS.get(name, torch.cos)
    return _ActModule(fn)


# ════════════════════════════════════════════════════════════
# STANDARD PINN
# Your exact notebook MaxwellPINN / HarmonicPINN architecture.
# (x,t) → (E,B) for Maxwell · x → u for Harmonic
# ════════════════════════════════════════════════════════════

class StandardPINN(nn.Module):
    """
    Sequential MLP with configurable activation.
    Default: 5×128 layers with cosine activation (notebook default).
    SIREN init applied automatically for oscillatory activations.
    """

    def __init__(self, layers: tuple, act: str = "cos", dtype=torch.float64):
        super().__init__()
        self.act_name = act
        act_fn = ACTIVATIONS.get(act, torch.cos)

        self.linears    = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i + 1], dtype=dtype)
            self._init_linear(lin, layers[i], i, act)
            self.linears.append(lin)
            if i < len(layers) - 2:
                self.activations.append(_ActModule(act_fn))

    @staticmethod
    def _init_linear(lin: nn.Linear, fan_in: int, layer_idx: int, act: str):
        if act in SIREN_ACTS:
            if layer_idx == 0:
                nn.init.uniform_(lin.weight, -1.0 / fan_in, 1.0 / fan_in)
            else:
                b = math.sqrt(6.0 / fan_in)
                nn.init.uniform_(lin.weight, -b, b)
        else:
            nn.init.xavier_normal_(lin.weight)
        nn.init.zeros_(lin.bias)

    def forward(self, *args) -> torch.Tensor:
        h = torch.cat(args, dim=-1) if len(args) > 1 else args[0]
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < len(self.activations):
                h = self.activations[i](h)
        return h

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════
# FOURIER FEATURE PINN
# Maps inputs through random Fourier features before MLP.
# Fixes: spectral bias — standard MLPs struggle to learn
# high-frequency components of the solution.
# ════════════════════════════════════════════════════════════

class FourierPINN(nn.Module):
    """
    Random Fourier Feature Network.
    Input: x → [sin(Bx), cos(Bx)] → MLP → output
    B ~ N(0, σ²I), learned or fixed.
    σ controls frequency bandwidth of features.
    """

    def __init__(
        self,
        in_dim:  int,
        out_dim: int,
        width:   int,
        depth:   int,
        act:     str   = "cos",
        n_freq:  int   = 64,
        sigma:   float = 1.0,
        dtype          = torch.float64,
    ):
        super().__init__()
        self.act_name = act

        # Random Fourier projection matrix (fixed, not learned)
        B = torch.randn(in_dim, n_freq, dtype=dtype) * sigma
        self.register_buffer("B", B)

        # MLP processes the Fourier features
        feat_dim = n_freq * 2  # [sin, cos] concatenated
        layers   = (feat_dim,) + (width,) * depth + (out_dim,)
        self.mlp = StandardPINN(layers, act, dtype)

    def forward(self, *args) -> torch.Tensor:
        x    = torch.cat(args, dim=-1) if len(args) > 1 else args[0]
        proj = x @ self.B
        phi  = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.mlp(phi)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════
# RESNET PINN
# Skip connections added every 2 layers.
# Fixes: vanishing gradients in deep networks (>5 layers).
# Also improves gradient flow for complex loss landscapes.
# ════════════════════════════════════════════════════════════

class ResNetPINN(nn.Module):
    """
    Residual MLP: h_{i+2} = act(W_{i+1}·act(W_i·h_i)) + h_i
    Skip connections every 2 hidden layers.
    Projection layer used if input dim ≠ hidden dim.
    """

    def __init__(self, layers: tuple, act: str = "cos", dtype=torch.float64):
        super().__init__()
        self.act_name  = act
        self.act_fn    = _ActModule(ACTIVATIONS.get(act, torch.cos))
        self.linears   = nn.ModuleList()

        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i + 1], dtype=dtype)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.linears.append(lin)

        # Projection if input dim differs from hidden dim
        self.proj = None
        if len(layers) > 2 and layers[0] != layers[1]:
            self.proj = nn.Linear(layers[0], layers[1], dtype=dtype)
            nn.init.xavier_normal_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, *args) -> torch.Tensor:
        x = torch.cat(args, dim=-1) if len(args) > 1 else args[0]
        h = self.proj(x) if self.proj is not None else x

        for i, lin in enumerate(self.linears[:-1]):
            h_new = self.act_fn(lin(h))
            # Skip every 2nd layer where shapes match
            if i > 0 and i % 2 == 1 and h_new.shape == h.shape:
                h_new = h_new + h
            h = h_new

        return self.linears[-1](h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════
# FACTORY
# ════════════════════════════════════════════════════════════

def build_model(
    pde:    str,
    act:    str   = "cos",
    width:  int   = 128,
    depth:  int   = 5,
    arch:   str   = "standard",
    device        = None,
    dtype         = torch.float64,
) -> nn.Module:
    """
    Build a PINN model for the given PDE.

    Args:
        pde:   'maxwell' → in=2 (x,t), out=2 (E,B)
               'harmonic' → in=1 (x), out=1 (u)
        act:   activation name (must be in ACTIVATIONS)
        width: hidden layer width
        depth: number of hidden layers
        arch:  'standard' | 'fourier' | 'resnet'
    """
    act    = act if act in ACTIVATIONS else "cos"
    in_d   = 2 if pde == "maxwell" else 1
    out_d  = 2 if pde == "maxwell" else 1
    device = device or torch.device("cpu")

    if arch == "fourier":
        model = FourierPINN(in_d, out_d, width, depth, act, dtype=dtype)
    elif arch == "resnet":
        layers = (in_d,) + (width,) * depth + (out_d,)
        model  = ResNetPINN(layers, act, dtype=dtype)
    else:
        layers = (in_d,) + (width,) * depth + (out_d,)
        model  = StandardPINN(layers, act, dtype=dtype)

    return model.to(device)
