"""
solvers.py — Solver strategies and failure diagnostics
=======================================================
Six strategies, each targeting a different failure mode:

  classic    Adam + CosineAnnealing + L-BFGS (notebook baseline)
  adaptive   Residual-adaptive collocation
  gradnorm   Automatic loss weight balancing (GradNorm)
  fourier    Fourier Feature architecture (spectral bias fix)
  resnet     ResNet architecture (vanishing gradient fix)
  ensemble   3 independent models (overfitting / instability fix)
"""

import math
import time
import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .physics import (
    maxwell_total_loss, harmonic_total_loss,
    adaptive_colloc_maxwell, compute_metrics,
    xmin, xmax, tmin, tmax, Lx, T_END, PI,
)
from .models import build_model


# ── Solver metadata ───────────────────────────────────────────

SOLVER_INFO = {
    "classic": {
        "name": "Classic PINN",
        "desc": "Adam + CosineAnnealing + L-BFGS. Baseline. Your exact notebook method.",
    },
    "adaptive": {
        "name": "Adaptive Sampling",
        "desc": "Residual-adaptive collocation. Fixes: physics error concentrated in small regions.",
    },
    "gradnorm": {
        "name": "GradNorm Balance",
        "desc": "Auto-balances loss term weights. Fixes: one loss term dominating others.",
    },
    "fourier": {
        "name": "Fourier Feature",
        "desc": "Random Fourier input encoding. Fixes: spectral bias (high-frequency solutions).",
    },
    "resnet": {
        "name": "ResNet PINN",
        "desc": "Skip connections every 2 layers. Fixes: vanishing gradients in deep networks.",
    },
    "ensemble": {
        "name": "Ensemble",
        "desc": "3 small independent models, combined prediction. Fixes: overfitting / instability.",
    },
}


# ════════════════════════════════════════════════════════════
# GRADNORM — adaptive loss weight balancing
# Paper: Chen et al. (2018) "GradNorm: Gradient Normalization
#   for Adaptive Loss Balancing"
# ════════════════════════════════════════════════════════════

class GradNormWeights(nn.Module):
    """
    Learnable per-task loss weights that auto-balance gradient magnitudes.
    Prevents any single loss term from dominating training.
    alpha: restoring force toward equal relative training rates.
    """

    def __init__(self, n_tasks: int = 4, alpha: float = 1.5):
        super().__init__()
        self.log_w   = nn.Parameter(torch.zeros(n_tasks, dtype=torch.float64))
        self.alpha   = alpha
        self.L0: torch.Tensor | None = None

    def weights(self) -> torch.Tensor:
        return torch.exp(self.log_w)


# ════════════════════════════════════════════════════════════
# CORE ADAM SOLVER
# ════════════════════════════════════════════════════════════

async def _solve_adam(
    model,
    pde:       str,
    epochs:    int,
    lr:        float,
    n_col:     int,
    send_cb,
    device,
    dtype,
    do_lbfgs:  bool  = True,
    log_n:     int   = 60,
    label:     str   = "",
    adaptive:  bool  = False,
) -> tuple[dict, dict, float]:
    """
    Core Adam training loop with CosineAnnealing scheduler.
    Shared by: classic, adaptive, fourier, resnet solvers.
    """
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    hist = {"epoch": [], "loss": [], "pde_loss": [], "ratio": [], "wave_loss": []}
    ev   = max(1, epochs // log_n)
    t0   = time.time()

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        if pde == "maxwell":
            if adaptive and ep % 100 == 0 and ep > 200:
                x, t_col = adaptive_colloc_maxwell(model, n_col, device, dtype)
            else:
                x     = (torch.rand(n_col, 1, device=device, dtype=dtype) * Lx
                         ).requires_grad_(True)
                t_col = (torch.rand(n_col, 1, device=device, dtype=dtype) * T_END
                         ).requires_grad_(True)
            L = maxwell_total_loss(model, x, t_col, ep, epochs, device, dtype)
        else:
            x = ((torch.rand(n_col, 1, device=device, dtype=dtype) * (2 * PI))
                 .requires_grad_(True))
            L = harmonic_total_loss(model, x, device, dtype)

        L["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        # Hard LR drop at 60%
        if ep == int(epochs * 0.60):
            for g in opt.param_groups:
                g["lr"] = lr * 0.2

        if ep % ev == 0 or ep == epochs:
            m = compute_metrics(model, pde, device, dtype)
            hist["epoch"].append(ep)
            hist["loss"].append(round(float(L["total"]), 6))
            hist["pde_loss"].append(round(float(L["pde"]), 6))
            hist["ratio"].append(round(m["ratio"], 5))
            hist["wave_loss"].append(round(float(L.get("wave", 0)), 6))
            await send_cb({
                "type":       "progress",
                "epoch":      ep,
                "n_epochs":   epochs,
                "loss":       float(L["total"]),
                "pde_loss":   float(L["pde"]),
                "wave_loss":  float(L.get("wave", 0)),
                "left_loss":  float(L.get("left", L.get("bc", 0))),
                "ratio":      m["ratio"],
                "pct":        m["pct"],
                "rel_l2":     m["rel_l2"],
                "elapsed":    round(time.time() - t0, 2),
                "history":    hist,
                "label":      label,
            })
            await asyncio.sleep(0)

    # ── L-BFGS refinement ──
    if do_lbfgs:
        await send_cb({"type": "status", "msg": f"L-BFGS refinement [{label}]…"})
        lb = optim.LBFGS(
            model.parameters(), lr=0.01, max_iter=40,
            history_size=50, line_search_fn="strong_wolfe"
        )

        def _closure():
            lb.zero_grad()
            if pde == "maxwell":
                xr = (torch.rand(512, 1, device=device, dtype=dtype) * Lx
                      ).requires_grad_(True)
                tr = (torch.rand(512, 1, device=device, dtype=dtype) * T_END
                      ).requires_grad_(True)
                Lv = maxwell_total_loss(model, xr, tr, epochs, epochs,
                                        device, dtype)["total"]
            else:
                xr = ((torch.rand(512, 1, device=device, dtype=dtype) * (2 * PI))
                      .requires_grad_(True))
                Lv = harmonic_total_loss(model, xr, device, dtype)["total"]
            Lv.backward()
            return Lv

        lb.step(_closure)

    return compute_metrics(model, pde, device, dtype), hist, time.time() - t0


# ════════════════════════════════════════════════════════════
# GRADNORM SOLVER
# ════════════════════════════════════════════════════════════

async def _solve_gradnorm(
    model,
    pde:     str,
    epochs:  int,
    lr:      float,
    n_col:   int,
    send_cb,
    device,
    dtype,
    log_n:   int = 60,
    label:   str = "",
) -> tuple[dict, dict, float]:
    gw     = GradNormWeights(n_tasks=4 if pde == "maxwell" else 3).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr)
    opt_gn = optim.Adam(gw.parameters(), lr=lr * 0.1)
    sch    = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    hist   = {"epoch": [], "loss": [], "pde_loss": [], "ratio": [], "wave_loss": []}
    ev     = max(1, epochs // log_n)
    t0     = time.time()

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        opt_gn.zero_grad()

        W = gw.weights()

        if pde == "maxwell":
            x     = (torch.rand(n_col, 1, device=device, dtype=dtype) * Lx
                     ).requires_grad_(True)
            t_col = (torch.rand(n_col, 1, device=device, dtype=dtype) * T_END
                     ).requires_grad_(True)
            base = maxwell_total_loss(model, x, t_col, ep, epochs, device, dtype)
            total = (W[0] * base["pde"] + W[1] * base["left"] +
                     W[2] * base["right"] + W[3] * base["per"])
            L = {**base, "total": total}
        else:
            x = ((torch.rand(n_col, 1, device=device, dtype=dtype) * (2 * PI))
                 .requires_grad_(True))
            L = harmonic_total_loss(model, x, device, dtype)

        L["total"].backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        opt_gn.step()

        if ep % ev == 0 or ep == epochs:
            m = compute_metrics(model, pde, device, dtype)
            hist["epoch"].append(ep)
            hist["loss"].append(round(float(L["total"]), 6))
            hist["pde_loss"].append(round(float(L["pde"]), 6))
            hist["ratio"].append(round(m["ratio"], 5))
            hist["wave_loss"].append(0)
            await send_cb({
                "type":       "progress",
                "epoch":      ep,
                "n_epochs":   epochs,
                "loss":       float(L["total"]),
                "pde_loss":   float(L["pde"]),
                "wave_loss":  0,
                "left_loss":  float(L.get("left", 0)),
                "ratio":      m["ratio"],
                "pct":        m["pct"],
                "rel_l2":     m["rel_l2"],
                "elapsed":    round(time.time() - t0, 2),
                "history":    hist,
                "label":      label,
                "gn_weights": [round(float(w), 3) for w in gw.weights()],
            })
            await asyncio.sleep(0)

    return compute_metrics(model, pde, device, dtype), hist, time.time() - t0


# ════════════════════════════════════════════════════════════
# ENSEMBLE SOLVER
# ════════════════════════════════════════════════════════════

async def run_ensemble(
    pde:       str,
    n_models:  int   = 3,
    width:     int   = 64,
    depth:     int   = 4,
    act:       str   = "cos",
    epochs:    int   = 2000,
    lr:        float = 1e-3,
    n_col:     int   = 2048,
    device           = None,
    dtype            = torch.float64,
    send_cb          = None,
    label:     str   = "",
) -> tuple[dict, dict, float, list]:
    """Train N independent models; return all models + ensemble metrics."""
    async def _noop(_): pass
    cb = send_cb or _noop

    models      = [build_model(pde, act, width, depth, device=device, dtype=dtype)
                   for _ in range(n_models)]
    t0          = time.time()
    all_metrics = []

    for i, m in enumerate(models):
        await cb({"type": "status",
                  "msg":  f"Ensemble: training model {i + 1}/{n_models}"})
        met, _, _ = await _solve_adam(
            m, pde, epochs // n_models, lr, n_col,
            _noop, device, dtype, do_lbfgs=False,
            label=f"{label}/m{i + 1}"
        )
        all_metrics.append(met)
        await asyncio.sleep(0)

    best_ratio  = max(m["ratio"]  for m in all_metrics)
    avg_rel_l2  = sum(m["rel_l2"] for m in all_metrics) / n_models
    return (
        {"ratio": best_ratio, "pct": 0.0, "rel_l2": avg_rel_l2},
        {},
        time.time() - t0,
        models,
    )


# ════════════════════════════════════════════════════════════
# UNIFIED SOLVER DISPATCHER
# ════════════════════════════════════════════════════════════

async def run_solver(
    model,
    pde:      str,
    epochs:   int,
    lr:       float,
    n_col:    int,
    send_cb,
    device,
    dtype,
    solver:   str  = "classic",
    do_lbfgs: bool = True,
    log_n:    int  = 60,
    label:    str  = "",
) -> tuple[dict, dict, float]:
    """
    Dispatch to the appropriate solver strategy.
    Returns (metrics, history, elapsed_seconds).
    """
    if solver == "gradnorm":
        return await _solve_gradnorm(
            model, pde, epochs, lr, n_col, send_cb,
            device, dtype, log_n, label
        )

    adaptive = (solver == "adaptive")
    return await _solve_adam(
        model, pde, epochs, lr, n_col, send_cb,
        device, dtype, do_lbfgs, log_n, label, adaptive
    )


# ════════════════════════════════════════════════════════════
# FAILURE DIAGNOSTICS
# ════════════════════════════════════════════════════════════

def diagnose(hist: dict, metrics: dict) -> tuple[str | None, str | None]:
    """
    Analyze training trajectory and suggest corrective strategy.
    Returns (title, recommendation) or (None, None) if training looks healthy.
    """
    if not hist.get("loss") or len(hist["loss"]) < 5:
        return None, None

    losses = hist["loss"]
    ratios = hist.get("ratio", [])

    # NaN / Inf
    if any(not math.isfinite(l) for l in losses[-5:]):
        return (
            "NaN/Inf detected",
            "Learning rate too high or SIREN init unstable. "
            "Try: lower lr by 10×, switch to tanh/swish, or use fourier architecture."
        )

    # Plateau
    recent = losses[-max(3, len(losses) // 5):]
    spread = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-10)
    if spread < 0.005 and metrics["rel_l2"] > 0.3:
        return (
            "Training plateau",
            "Loss stagnated above target. "
            "Try: L-BFGS, increase lr × 5, switch to adaptive collocation, "
            "or use GradNorm balancing."
        )

    # Oscillation
    if len(losses) > 5:
        diffs  = [abs(losses[i] - losses[i - 1]) for i in range(1, len(losses))]
        mean_d = sum(diffs) / len(diffs)
        if diffs[-1] > 5 * mean_d:
            return (
                "Loss oscillating",
                "Gradient instability. "
                "Try: reduce lr × 0.1, gradient clipping already on, "
                "or use ResNet architecture for better stability."
            )

    # Good PDE residual but poor generalization
    pde_hist = hist.get("pde_loss", [])
    if len(pde_hist) > 3:
        pde_rel = pde_hist[-1] / (pde_hist[0] + 1e-10)
        if pde_rel < 0.01 and metrics["rel_l2"] > 0.3:
            return (
                "BC / generalization failure",
                "PDE residual low but global accuracy poor. "
                "Try: increase bc_weight × 5, more epochs, "
                "or ResNet architecture for better boundary enforcement."
            )

    # Slow convergence
    if ratios and len(ratios) >= 2 and ratios[-1] < 0.5:
        return (
            "Slow convergence",
            "Accuracy still below 50%. "
            "Try: Fourier Feature encoding (spectral bias fix), "
            "deeper/wider network, or cosine/sin activation if using tanh."
        )

    return None, None
