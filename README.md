# PINN Research Engine

Autonomous self-training Physics-Informed Neural Network research platform.
Runs indefinitely discovering optimal configurations for Maxwell EM and Harmonic ODE problems.

## Physics

**Maxwell EM** — `(x,t) → (E, B)`
```
∂E/∂x + ∂B/∂t = 0
∂B/∂x + (1/c²)∂E/∂t = 0
```
Left BC: traveling wave `E=B=sin(kx−ωt)` · Right BC: characteristic outflow · Periodic in x

**Harmonic ODE** — `x → u`
```
u″ + u = 0,  u(0)=0,  u(π/2)=1
```
Training domain `[0, 2π]` · Evaluated on `[−50, 50]`

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/pinn-research
cd pinn-research
pip install -r requirements.txt
python run.py
```

Browser opens automatically at `http://localhost:8765`

## Project Structure

```
pinn-research/
├── run.py              # Entry point
├── requirements.txt
├── src/
│   ├── physics.py      # All PDE loss functions (exact from research notebook)
│   ├── models.py       # Architectures: Standard, Fourier Feature, ResNet
│   ├── solvers.py      # 6 solver strategies with failure diagnostics
│   ├── engine.py       # Autonomous research engine (UCB1 bandit + evolutionary)
│   └── server.py       # FastAPI app + WebSocket handlers
└── static/
    └── index.html      # Frontend: Three.js 3D EM field + Chart.js dashboards
```

## Solver Strategies

| Solver | Method | Fixes |
|--------|--------|-------|
| `classic` | Adam → CosineAnnealing → L-BFGS | Baseline (exact notebook) |
| `adaptive` | Residual-adaptive collocation | Physics error concentrated in small regions |
| `gradnorm` | GradNorm loss weight balancing | One loss term dominating others |
| `fourier` | Random Fourier Feature encoding | Spectral bias — high-frequency solutions |
| `resnet` | Skip connections every 2 layers | Vanishing gradients in deep networks |
| `ensemble` | 3 independent models voting | Single-model overfitting / instability |

## Autonomous Engine

The research engine runs indefinitely through generations:
1. **Phase 1** — Rapid scan of population (default 300 epochs each)
2. **Phase 2** — Deep retrain top 2 configs (default 1500 epochs)
3. **Champion** — Every 4 generations, full training of all-time best config

Uses UCB1 bandit to bias exploration toward high-performing activation/architecture combinations.
Evolutionary mutation seeds each generation from the Hall of Fame.
Failure diagnostics automatically identify NaN, plateau, oscillation, and BC failure modes.

## 3D EM Field Visualization

Three.js WebGL renderer driven by actual PINN output data:
- Animated E-field vectors in Y-plane (red)
- B-field vectors in Z-plane (blue)
- Filled wave surfaces with opacity
- Drag to orbit, scroll to zoom
- Toggle: PINN / truth / overlay comparison
- Slice charts with pointwise error at current timestep

## Activations (18)

`cos · sin · sincos · sin2x · tanh · swish · gelu · erf · softplus`
`morlet · sinc · damped_sin · mish · isru · gaussian · mex_hat · selu · elu`

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CPU works; CUDA/MPS auto-detected)
- No GPU required
