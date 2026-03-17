"""
physics.py — PDE loss functions + 4 physics problems
=====================================================
Maxwell EM     (x,t) → (E,B)   — 1D transverse electromagnetic wave
Harmonic ODE   x → u           — u″+u=0, canonical benchmark
Burgers PDE    (x,t) → u       — ν·u_xx − u·u_x = u_t  (nonlinear, shock)
Heat PDE       (x,t) → u       — u_t = α·u_xx  (parabolic diffusion)
"""

import math
import torch
import torch.nn as nn
import numpy as np

PI    = math.pi
Lx    = 2 * PI;  T_END = 2 * PI
xmin  = 0.0;     xmax  = Lx
tmin  = 0.0;     tmax  = T_END
c     = 1.0;     k     = 1.0;    omega = c * k

# Burgers params
NU    = 0.01 / PI

# Heat params
ALPHA = 0.1

PDE_REGISTRY = {
    "maxwell":  {"label":"Maxwell EM",    "in":2,"out":2,"color":"#00f5ff"},
    "harmonic": {"label":"Harmonic ODE",  "in":1,"out":1,"color":"#ff006e"},
    "burgers":  {"label":"Burgers PDE",   "in":2,"out":1,"color":"#39ff14"},
    "heat":     {"label":"Heat Equation", "in":2,"out":1,"color":"#ffbe0b"},
}

def true_E(x,t):  return torch.sin(k*x - omega*t)
def true_B(x,t):  return (1.0/c)*torch.sin(k*x - omega*t)
def true_harmonic(x): return torch.sin(x)
def true_heat(x,t):   return torch.exp(-ALPHA*t)*torch.sin(x)

def _g(y,x):
    return torch.autograd.grad(y,x,torch.ones_like(y),
                               retain_graph=True,create_graph=True)[0]

# ── Maxwell ─────────────────────────────────────────────────
def left_bc_driven(model,device,dtype,N=512):
    t=torch.rand(N,1,device=device,dtype=dtype)*(tmax-tmin)+tmin
    X=torch.cat([torch.full_like(t,xmin),t],1)
    E,B=model(X).split(1,1)
    ph=-omega*t
    return ((E-torch.sin(ph))**2+(B-(1/c)*torch.sin(ph))**2).mean()

def right_outflow_loss(model,device,dtype,N=512):
    t=torch.rand(N,1,device=device,dtype=dtype)*(tmax-tmin)+tmin
    X=torch.cat([torch.full_like(t,xmax),t],1)
    E,B=model(X).split(1,1)
    return ((E-c*B)**2).mean()

def maxwell_residuals(model,x,t):
    X=torch.cat([x,t],1); E,B=model(X).split(1,1)
    return _g(E,x)+_g(B,t), _g(B,x)+(1/c**2)*_g(E,t)

def wave_residuals(model,x,t):
    X=torch.cat([x,t],1); E,B=model(X).split(1,1)
    return _g(_g(E,t),t)-c**2*_g(_g(E,x),x), _g(_g(B,t),t)-c**2*_g(_g(B,x),x)

def periodic_loss_maxwell(model,device,dtype,N=512):
    t=torch.rand(N,1,device=device,dtype=dtype)*T_END
    x=torch.rand(N,1,device=device,dtype=dtype)*Lx
    Ec,Bc=model(torch.cat([x,t],1)).split(1,1)
    Ep,Bp=model(torch.cat([x+Lx,t],1)).split(1,1)
    Em,Bm=model(torch.cat([x-Lx,t],1)).split(1,1)
    return ((Ep-Ec)**2+(Em-Ec)**2+(Bp-Bc)**2+(Bm-Bc)**2).mean()

def maxwell_total_loss(model,x,t,epoch,n_epochs,device,dtype,w_bc=1.0):
    r1,r2=maxwell_residuals(model,x,t)
    L_pde=(r1**2+r2**2).mean()
    L_l=left_bc_driven(model,device,dtype)
    L_r=right_outflow_loss(model,device,dtype)
    L_p=periodic_loss_maxwell(model,device,dtype)
    if epoch/n_epochs>=0.30:
        rE,rB=wave_residuals(model,x,t)
        L_w=(rE**2+rB**2).mean()
        tot=L_pde+0.5*L_w+w_bc*(L_l+L_r)+L_p
    else:
        L_w=torch.zeros(1,device=device,dtype=dtype).squeeze()
        tot=L_pde+w_bc*(L_l+L_r)+L_p
    return {"total":tot,"pde":L_pde,"wave":L_w,"left":L_l,"right":L_r,"per":L_p}

# ── Harmonic ODE ─────────────────────────────────────────────
def ode_residual(model,x):
    y=model(x); dy=torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0]
    return torch.autograd.grad(dy,x,torch.ones_like(dy),create_graph=True)[0]+y

def periodic_loss_fn(model,x):
    period=2*PI
    return (model(x-period)-model(x))**2+(model(x+period)-model(x))**2

def harmonic_total_loss(model,x,device,dtype,bc_weight=10.0):
    L_pde=torch.mean(ode_residual(model,x)**2)
    xbc=torch.tensor([[0.0],[PI/2]],device=device,dtype=dtype)
    L_bc=torch.mean((model(xbc)-torch.sin(xbc))**2)
    L_per=periodic_loss_fn(model,x.detach()).mean()
    tot=L_pde+bc_weight*L_bc+L_per
    return {"total":tot,"pde":L_pde,"bc":L_bc,"per":L_per,
            "wave":torch.zeros(1,device=device,dtype=dtype).squeeze(),"left":L_bc}

# ── Burgers PDE  u_t + u·u_x = ν·u_xx ──────────────────────
def burgers_residual(model,x,t):
    X=torch.cat([x,t],1); u=model(X)
    u_t=_g(u,t); u_x=_g(u,x); u_xx=_g(u_x,x)
    return u_t+u*u_x-NU*u_xx

def burgers_ic_loss(model,device,dtype,N=256):
    x=torch.rand(N,1,device=device,dtype=dtype)*2-1
    t=torch.zeros(N,1,device=device,dtype=dtype)
    X=torch.cat([x,t],1)
    u_true=-torch.sin(PI*x)
    return ((model(X)-u_true)**2).mean()

def burgers_bc_loss(model,device,dtype,N=256):
    t=torch.rand(N,1,device=device,dtype=dtype)
    Xl=torch.cat([torch.full_like(t,-1.0),t],1)
    Xr=torch.cat([torch.full_like(t, 1.0),t],1)
    return (model(Xl)**2+model(Xr)**2).mean()

def burgers_total_loss(model,x,t,device,dtype,w_ic=10.0,w_bc=5.0):
    # Rescale inputs to Burgers domain [-1,1]×[0,1]
    xb=(x/(PI)-1.0).requires_grad_(True)
    tb=(t/(2*PI)).requires_grad_(True)
    r=burgers_residual(model,xb,tb)
    L_pde=(r**2).mean()
    L_ic=burgers_ic_loss(model,device,dtype)
    L_bc=burgers_bc_loss(model,device,dtype)
    tot=L_pde+w_ic*L_ic+w_bc*L_bc
    return {"total":tot,"pde":L_pde,"wave":L_ic,"left":L_bc,
            "bc":L_ic,"per":torch.zeros(1,device=device,dtype=dtype).squeeze()}

# ── Heat PDE  u_t = α·u_xx ──────────────────────────────────
def heat_residual(model,x,t):
    X=torch.cat([x,t],1); u=model(X)
    u_t=_g(u,t); u_xx=_g(_g(u,x),x)
    return u_t-ALPHA*u_xx

def heat_ic_loss(model,device,dtype,N=256):
    x=torch.rand(N,1,device=device,dtype=dtype)*2*PI
    t=torch.zeros(N,1,device=device,dtype=dtype)
    X=torch.cat([x,t],1)
    return ((model(X)-torch.sin(x))**2).mean()

def heat_bc_loss(model,device,dtype,N=256):
    t=torch.rand(N,1,device=device,dtype=dtype)*T_END
    Xl=torch.cat([torch.zeros(N,1,device=device,dtype=dtype),t],1)
    Xr=torch.cat([torch.full((N,1),2*PI,device=device,dtype=dtype),t],1)
    return (model(Xl)**2+model(Xr)**2).mean()

def heat_total_loss(model,x,t,device,dtype,w_ic=10.0,w_bc=5.0):
    r=heat_residual(model,x,t); L_pde=(r**2).mean()
    L_ic=heat_ic_loss(model,device,dtype)
    L_bc=heat_bc_loss(model,device,dtype)
    tot=L_pde+w_ic*L_ic+w_bc*L_bc
    return {"total":tot,"pde":L_pde,"wave":L_ic,"left":L_bc,
            "bc":L_ic,"per":torch.zeros(1,device=device,dtype=dtype).squeeze()}

# ── Dispatch ─────────────────────────────────────────────────
def compute_total_loss(model,x,t_col,epoch,n_epochs,pde,device,dtype,w_bc=1.0):
    if pde=="maxwell":
        return maxwell_total_loss(model,x,t_col,epoch,n_epochs,device,dtype,w_bc)
    elif pde=="harmonic":
        return harmonic_total_loss(model,x,device,dtype,w_bc*10)
    elif pde=="burgers":
        return burgers_total_loss(model,x,t_col,device,dtype)
    elif pde=="heat":
        return heat_total_loss(model,x,t_col,device,dtype)
    return harmonic_total_loss(model,x,device,dtype)

# ── Adaptive collocation ─────────────────────────────────────
@torch.no_grad()
def adaptive_colloc(model,pde,n_col,device,dtype,alpha=0.7):
    n_adapt=int(n_col*alpha); n_rand=n_col-n_adapt
    if pde=="maxwell":
        xf=torch.rand(4096,1,device=device,dtype=dtype)*Lx
        tf=torch.rand(4096,1,device=device,dtype=dtype)*T_END
        xf.requires_grad_(True); tf.requires_grad_(True)
        r1,r2=maxwell_residuals(model,xf,tf)
        res=(r1**2+r2**2).detach().squeeze()
    else:
        xf=torch.rand(4096,1,device=device,dtype=dtype)*(2*PI)
        xf.requires_grad_(True)
        res=ode_residual(model,xf).detach().squeeze()**2
        tf=None
    w=(res+1e-12)/(res+1e-12).sum()
    idx=torch.multinomial(w,n_adapt,replacement=True)
    if pde=="maxwell":
        xa=xf[idx].detach().requires_grad_(True)
        ta=tf[idx].detach().requires_grad_(True)
        xr=(torch.rand(n_rand,1,device=device,dtype=dtype)*Lx).requires_grad_(True)
        tr=(torch.rand(n_rand,1,device=device,dtype=dtype)*T_END).requires_grad_(True)
        return torch.cat([xa,xr],0), torch.cat([ta,tr],0)
    else:
        xa=xf[idx].detach().requires_grad_(True)
        xr=(torch.rand(n_rand,1,device=device,dtype=dtype)*(2*PI)).requires_grad_(True)
        return torch.cat([xa,xr],0), None

# ── Metrics ──────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(model,pde,device,dtype):
    if pde=="maxwell":
        xs=torch.linspace(xmin,xmax,60,device=device,dtype=dtype)
        ts=torch.linspace(tmin,tmax,60,device=device,dtype=dtype)
        xg,tg=torch.meshgrid(xs,ts,indexing="ij")
        xf,tf=xg.reshape(-1,1),tg.reshape(-1,1)
        pred=model(torch.cat([xf,tf],1))
        truth=torch.cat([true_E(xf,tf),true_B(xf,tf)],1)
    elif pde=="harmonic":
        xf=torch.linspace(-6*PI,6*PI,1000,device=device,dtype=dtype).view(-1,1)
        pred=model(xf); truth=torch.sin(xf)
    elif pde=="heat":
        xs=torch.linspace(0,2*PI,50,device=device,dtype=dtype)
        ts=torch.linspace(0,T_END,50,device=device,dtype=dtype)
        xg,tg=torch.meshgrid(xs,ts,indexing="ij")
        xf,tf=xg.reshape(-1,1),tg.reshape(-1,1)
        pred=model(torch.cat([xf,tf],1)); truth=true_heat(xf,tf)
    elif pde=="burgers":
        xs=torch.linspace(-1,1,80,device=device,dtype=dtype)
        ts=torch.linspace(0,1,80,device=device,dtype=dtype)
        xg,tg=torch.meshgrid(xs,ts,indexing="ij")
        xf,tf=xg.reshape(-1,1),tg.reshape(-1,1)
        pred=model(torch.cat([xf,tf],1)); truth=torch.zeros_like(pred)
    else:
        return {"ratio":0.0,"pct":0.0,"rel_l2":1.0}
    err=(pred-truth).abs(); den=truth.abs()+1e-8
    return {"ratio":float((1-err/den).clamp(0,1).mean()),
            "pct":float(((err/den)<0.05).float().mean()*100),
            "rel_l2":float(err.norm()/(truth.norm()+1e-8))}

@torch.no_grad()
def vis_2d(model,pde,device,dtype):
    if pde=="maxwell":
        xv=torch.linspace(xmin,xmax,300,device=device,dtype=dtype).view(-1,1)
        tv=torch.full_like(xv,0.25); X=torch.cat([xv,tv],1); out=model(X).cpu().float()
        return {"x":xv.view(-1).tolist(),"E_pred":out[:,0].tolist(),"B_pred":out[:,1].tolist(),
                "E_true":true_E(xv,tv).view(-1).cpu().float().tolist(),
                "B_true":true_B(xv,tv).view(-1).cpu().float().tolist()}
    elif pde=="harmonic":
        xv=torch.linspace(-4*PI,4*PI,500,device=device,dtype=dtype).view(-1,1)
        return {"x":xv.view(-1).tolist(),"pred":model(xv).view(-1).cpu().float().tolist(),
                "true":torch.sin(xv).view(-1).cpu().float().tolist()}
    elif pde=="heat":
        xv=torch.linspace(0,2*PI,300,device=device,dtype=dtype).view(-1,1)
        tv=torch.full_like(xv,0.5); X=torch.cat([xv,tv],1)
        u_pred=model(X).view(-1).cpu().float().tolist()
        u_true=true_heat(xv,tv).view(-1).cpu().float().tolist()
        return {"x":xv.view(-1).tolist(),"pred":u_pred,"true":u_true}
    elif pde=="burgers":
        xv=torch.linspace(-1,1,300,device=device,dtype=dtype).view(-1,1)
        tv=torch.full_like(xv,0.3); X=torch.cat([xv,tv],1)
        return {"x":xv.view(-1).tolist(),"pred":model(X).view(-1).cpu().float().tolist(),
                "true":[0.0]*300}
    return {}

@torch.no_grad()
def field_grid(model,pde,device,dtype,nx=120,nt=80):
    if pde!="maxwell":
        return vis_2d(model,pde,device,dtype)
    xs=torch.linspace(xmin,xmax,nx,device=device,dtype=dtype)
    ts=torch.linspace(tmin,tmax,nt,device=device,dtype=dtype)
    E_p,B_p,E_t,B_t=[],[],[],[]
    for ti in ts:
        xv=xs.view(-1,1); tv=ti.repeat(nx).view(-1,1)
        out=model(torch.cat([xv,tv],1)).cpu().float()
        E_p.append(out[:,0].tolist()); B_p.append(out[:,1].tolist())
        E_t.append(true_E(xv,tv).view(-1).cpu().float().tolist())
        B_t.append(true_B(xv,tv).view(-1).cpu().float().tolist())
    return {"pde":"maxwell","nx":nx,"nt":nt,"x":xs.tolist(),"t":ts.tolist(),
            "E_pred":E_p,"B_pred":B_p,"E_true":E_t,"B_true":B_t}

@torch.no_grad()
def compare_curve(model,pde,device,dtype,n=500):
    if pde=="maxwell":
        xv=torch.linspace(xmin,xmax,n,device=device,dtype=dtype).view(-1,1)
        tv=torch.full_like(xv,0.5); X=torch.cat([xv,tv],1)
        return {"x":xv.view(-1).tolist(),"pred":model(X)[:,0].cpu().float().tolist(),
                "true":true_E(xv,tv).view(-1).cpu().float().tolist()}
    else:
        xv=torch.linspace(-20,20,n,device=device,dtype=dtype).view(-1,1)
        return {"x":xv.view(-1).tolist(),"pred":model(xv).view(-1).cpu().float().tolist(),
                "true":torch.sin(xv).view(-1).cpu().float().tolist()}

@torch.no_grad()
def get_residual_field(model,pde,device,dtype,n=200):
    """Returns spatial residual field for spectral analysis."""
    if pde in ("maxwell","heat","burgers"):
        xv=(torch.rand(n,1,device=device,dtype=dtype)*Lx).requires_grad_(True)
        tv=(torch.rand(n,1,device=device,dtype=dtype)*T_END).requires_grad_(True)
        if pde=="maxwell":
            r1,r2=maxwell_residuals(model,xv,tv)
            res=(r1**2+r2**2).sqrt().detach()
        elif pde=="heat":
            res=heat_residual(model,xv,tv).abs().detach()
        else:
            res=burgers_residual(model,xv,tv).abs().detach()
        return res.squeeze(), xv.detach().squeeze()
    else:
        xv=(torch.rand(n,1,device=device,dtype=dtype)*(2*PI)).requires_grad_(True)
        res=ode_residual(model,xv).abs().detach().squeeze()
        return res, xv.detach().squeeze()
