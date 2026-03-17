"""
algorithms.py v3 — Autonomous Self-Training Intelligence
=========================================================
15 algorithms forming a closed self-directing research loop.

  BayesianOptimizer      GP surrogate + Expected Improvement
  PCGrad                 Gradient surgery for conflicting losses
  CurriculumScheduler    Auto-adapt weights, wave activation, LR drops
  SelfDistiller          Champion → student knowledge distillation
  SpectralAnalyzer       FFT residuals → dominant freq → Fourier σ rec
  ParetoTracker          Multi-objective front: accuracy×speed×size
  NoveltySearch          k-NN behavioral diversity → exploration bonus
  FailureMemory          Gaussian kernel blacklist of failed regions
  NTKMonitor             NTK condition number → architecture diagnosis
  MetaLearner            Warm-start from weighted avg of best run params
  ReplayBuffer           Prioritized high-residual point replay
  HomoscedasticWeighter  Learnable uncertainty weights (Kendall 2018)
  LayerHealthMonitor     Per-layer gradient + dead-neuron detection
  FourierSigmaTuner      EMA of spectral-recommended σ values
  ConfigScorer           kNN pre-filter on HOF feature patterns
"""

import math, random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Shared encoding ──────────────────────────────────────────
_ACTS  = ["cos","sin","sincos","sin2x","tanh","swish","gelu","erf",
           "softplus","morlet","sinc","damped_sin","mish","isru",
           "gaussian","mex_hat","selu","elu"]
_ARCHS = ["standard","fourier","resnet"]
_SOLS  = ["classic","adaptive","gradnorm"]

def _enc(cfg: dict) -> np.ndarray:
    ai  = _ACTS.index( cfg.get("act","cos"))       / len(_ACTS)
    ari = _ARCHS.index(cfg.get("arch","standard")) / len(_ARCHS)
    si  = _SOLS.index( cfg.get("solver","classic"))/ len(_SOLS)
    w   = cfg.get("width",128)/512.0
    d   = cfg.get("depth",5  )/9.0
    lr  = (math.log10(max(1e-6,cfg.get("lr",1e-3)))+5)/4.0
    return np.array([ai,ari,si,w,d,lr],dtype=np.float64)


# ════════════════════════════════════════════════════════════
# 1. BAYESIAN OPTIMIZER
# ════════════════════════════════════════════════════════════
class BayesianOptimizer:
    def __init__(self, ls=1.0, noise=1e-3):
        self.ls=ls; self.noise=noise
        self.X: list=[];  self.y: list=[]

    def _K(self,A,B):
        d=A[:,None,:]-B[None,:,:]
        return np.exp(-0.5*np.sum(d**2,-1)/self.ls**2)

    def observe(self,cfg,score):
        self.X.append(_enc(cfg)); self.y.append(float(score))

    def suggest(self,cands):
        if len(self.X)<3: return random.choice(cands)
        Xo=np.array(self.X); yo=np.array(self.y); yb=yo.max()
        K=self._K(Xo,Xo)+(self.noise+1e-8)*np.eye(len(Xo))
        try: Ki=np.linalg.inv(K)
        except: return random.choice(cands)
        Xc=np.array([_enc(c) for c in cands])
        Ks=self._K(Xc,Xo); mu=Ks@Ki@yo
        sig=np.sqrt(np.maximum(np.diag(self._K(Xc,Xc)-Ks@Ki@Ks.T),1e-12))
        z=(mu-yb)/sig
        cdf=0.5*(1+np.vectorize(math.erf)(z/math.sqrt(2)))
        pdf=np.exp(-0.5*z**2)/math.sqrt(2*math.pi)
        ei=(mu-yb)*cdf+sig*pdf
        return cands[int(np.argmax(ei))]

    def state(self):
        return {"n":len(self.X),
                "best":round(float(max(self.y)),4) if self.y else None,
                "last5":[round(v,4) for v in self.y[-5:]]}


# ════════════════════════════════════════════════════════════
# 2. PCGRAD
# ════════════════════════════════════════════════════════════
class PCGrad:
    @staticmethod
    def apply(model, losses):
        params=[p for p in model.parameters() if p.requires_grad]
        grads=[]
        for i,L in enumerate(losses):
            model.zero_grad()
            L.backward(retain_graph=(i<len(losses)-1))
            grads.append([p.grad.clone() if p.grad is not None
                          else torch.zeros_like(p) for p in params])
        proj=[list(g) for g in grads]
        for i in range(len(grads)):
            for j in range(len(grads)):
                if i==j: continue
                gi=torch.cat([g.flatten() for g in proj[i]])
                gj=torch.cat([g.flatten() for g in grads[j]])
                dot=(gi*gj).sum()
                if dot<0:
                    coef=dot/((gj*gj).sum()+1e-12)
                    for k in range(len(params)):
                        proj[i][k]=proj[i][k]-coef*grads[j][k]
        model.zero_grad()
        for p,*pg in zip(params,*proj): p.grad=sum(pg)
        return torch.cat([p.grad.flatten() for p in params if p.grad is not None]).norm()

    @staticmethod
    def conflicts(model,losses):
        params=[p for p in model.parameters() if p.requires_grad]
        flat=[]
        for i,L in enumerate(losses):
            model.zero_grad(); L.backward(retain_graph=(i<len(losses)-1))
            flat.append(torch.cat([p.grad.flatten() if p.grad is not None
                                    else torch.zeros(p.numel(),device=p.device)
                                    for p in params]))
        model.zero_grad()
        return [round(float(F.cosine_similarity(flat[i].unsqueeze(0),
                                                flat[j].unsqueeze(0))),3)
                for i in range(len(flat)) for j in range(i+1,len(flat))]


# ════════════════════════════════════════════════════════════
# 3. CURRICULUM SCHEDULER
# ════════════════════════════════════════════════════════════
class CurriculumScheduler:
    def __init__(self):
        self.step=0; self.bc_weight=1.0; self.wave_on=False
        self.pde_h=deque(maxlen=30); self.bc_h=deque(maxlen=30)
        self.tot_h=deque(maxlen=60); self.plateau_n=0
        self.n_drops=0; self.log=[]

    def update(self,L):
        self.step+=1
        pde=float(L.get("pde",0)); bc=float(L.get("left",L.get("bc",0)))
        tot=float(L.get("total",0))
        self.pde_h.append(pde); self.bc_h.append(bc); self.tot_h.append(tot)
        adj={}
        if len(self.pde_h)>=8:
            pm=sum(self.pde_h)/len(self.pde_h); bm=sum(self.bc_h)/len(self.bc_h)
            r=bm/(pm+1e-12)
            if r>8 and self.bc_weight<40:
                self.bc_weight=min(40,self.bc_weight*1.08)
                adj["bc_weight"]=round(self.bc_weight,2)
            elif r<0.05 and self.bc_weight>1.5:
                self.bc_weight=max(1.0,self.bc_weight*0.97)
        if len(self.tot_h)>=12 and self.n_drops<3:
            win=list(self.tot_h)[-12:]
            sp=(max(win)-min(win))/(abs(min(win))+1e-10)
            if sp<0.002:
                self.plateau_n+=1
                if self.plateau_n>=4:
                    adj["reduce_lr"]=True; self.plateau_n=0; self.n_drops+=1
                    self.log.append(f"s{self.step}:lrdrop#{self.n_drops}")
            else: self.plateau_n=0
        if not self.wave_on and pde<5e-3 and self.step>50:
            self.wave_on=True; adj["activate_wave"]=True
            self.log.append(f"s{self.step}:wave_on")
        return adj

    def bc_w(self): return self.bc_weight
    def state(self):
        return {"step":self.step,"bc_weight":round(self.bc_weight,3),
                "wave_active":self.wave_on,"n_drops":self.n_drops,
                "log":self.log[-3:]}


# ════════════════════════════════════════════════════════════
# 4. SELF-DISTILLER
# ════════════════════════════════════════════════════════════
class SelfDistiller:
    def __init__(self,teacher,alpha=0.3,T=2.0):
        self.teacher=teacher; self.alpha=alpha; self.T=T
        self.teacher.eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)

    def dist_loss(self,student,X):
        with torch.no_grad(): t=self.teacher(X)/self.T
        return F.mse_loss(student(X)/self.T,t)*self.T**2

    def combined(self,student,Lp,X):
        return (1-self.alpha)*Lp+self.alpha*self.dist_loss(student,X)


# ════════════════════════════════════════════════════════════
# 5. SPECTRAL ANALYZER
# ════════════════════════════════════════════════════════════
class SpectralAnalyzer:
    @staticmethod
    @torch.no_grad()
    def analyze(res,x,n_modes=32):
        idx=torch.argsort(x.view(-1))
        rs=res.view(-1)[idx].cpu().float().numpy()
        xs=x.view(-1)[idx].cpu().float().numpy()
        if len(rs)<8:
            return {"dominant_freq":1.0,"sigma_recommend":1.0,
                    "power_spectrum":[],"spectral_flatness":1.0,"high_freq":False}
        N=len(rs); fft=np.fft.rfft(rs)
        frq=np.fft.rfftfreq(N,d=max((xs[-1]-xs[0])/N,1e-10))
        pwr=np.abs(fft)**2
        dom=float(frq[1+np.argmax(pwr[1:])]) if len(pwr)>1 else 1.0
        sig=float(np.clip(dom*2*math.pi,0.3,25.0))
        eps=1e-12
        gm=float(np.exp(np.mean(np.log(pwr[1:]+eps))))
        am=float(np.mean(pwr[1:])+eps)
        return {"dominant_freq":round(dom,4),"sigma_recommend":round(sig,3),
                "spectral_flatness":round(gm/am,4),"high_freq":bool(dom>2.0),
                "power_spectrum":[{"freq":round(float(frq[i]),3),
                                   "power":round(float(pwr[i]),6)}
                                  for i in range(min(n_modes,len(frq)))]}


# ════════════════════════════════════════════════════════════
# 6. PARETO TRACKER
# ════════════════════════════════════════════════════════════
@dataclass
class _PP:
    cfg:dict; accuracy:float; speed:float; compactness:float
    run_id:int; label:str=""

class ParetoTracker:
    def __init__(self): self.pts=[]

    def add(self,cfg,metrics,elapsed,n_params,run_id,label=""):
        self.pts.append(_PP(cfg=cfg,
            accuracy=1-metrics.get("rel_l2",1),
            speed=1/(elapsed+1),compactness=1/(n_params+1),
            run_id=run_id,label=label))
        self._prune()

    def _dom(self,a,b):
        return(a.accuracy>=b.accuracy and a.speed>=b.speed
               and a.compactness>=b.compactness
               and(a.accuracy>b.accuracy or a.speed>b.speed
                   or a.compactness>b.compactness))

    def _prune(self):
        self.pts=[p for p in self.pts
                  if not any(self._dom(q,p) for q in self.pts if q is not p)]

    def front(self):
        return[{"run_id":p.run_id,"label":p.label,
                "accuracy":round(p.accuracy,4),"speed":round(p.speed,6),
                "compactness":round(p.compactness,8),
                "act":p.cfg.get("act","?"),"arch":p.cfg.get("arch","?")}
               for p in sorted(self.pts,key=lambda x:-x.accuracy)]


# ════════════════════════════════════════════════════════════
# 7. NOVELTY SEARCH
# ════════════════════════════════════════════════════════════
class NoveltySearch:
    def __init__(self,k=5,maxsize=60):
        self.k=k; self.maxsize=maxsize
        self.archive=[]; self.meta=[]

    def _beh(self,curve):
        arr=np.array(curve,dtype=np.float32)
        if not len(arr): return np.zeros(32)
        return arr[np.round(np.linspace(0,len(arr)-1,32)).astype(int)]

    def novelty(self,curve):
        b=self._beh(curve)
        if len(self.archive)<self.k: return 1.0
        ds=np.array([np.linalg.norm(b-a) for a in self.archive])
        return float(np.sort(ds)[:self.k].mean())

    def add(self,curve,meta={}):
        self.archive.append(self._beh(curve)); self.meta.append(meta)
        if len(self.archive)>self.maxsize:
            self.archive.pop(0); self.meta.pop(0)

    def state(self): return {"archive_size":len(self.archive)}


# ════════════════════════════════════════════════════════════
# 8. FAILURE MEMORY
# ════════════════════════════════════════════════════════════
class FailureMemory:
    def __init__(self,radius=0.18,cap=120):
        self.radius=radius; self.cap=cap
        self.fails=[]; self.wins=[]

    def record_fail(self,cfg):
        self.fails.append(cfg)
        if len(self.fails)>self.cap: self.fails.pop(0)

    def record_win(self,cfg,score): self.wins.append({"cfg":cfg,"score":score})

    def penalty(self,cfg):
        if not self.fails: return 0.0
        return max(0.0,1-min(np.linalg.norm(_enc(cfg)-_enc(f))
                              for f in self.fails)/self.radius)

    def blacklisted(self,cfg): return self.penalty(cfg)>0.88

    def state(self): return {"fails":len(self.fails),"wins":len(self.wins)}


# ════════════════════════════════════════════════════════════
# 9. NTK MONITOR
# ════════════════════════════════════════════════════════════
class NTKMonitor:
    @staticmethod
    @torch.no_grad()
    def condition_number(model,X,max_n=48):
        X=X[:max_n]; ps=[p for p in model.parameters() if p.requires_grad]
        rows=[]
        for i in range(X.shape[0]):
            o=model(X[i:i+1]).sum()
            gs=torch.autograd.grad(o,ps,create_graph=False,allow_unused=True)
            rows.append(torch.cat([g.flatten() if g is not None
                                    else torch.zeros(p.numel(),device=X.device)
                                    for g,p in zip(gs,ps)]).float())
        J=torch.stack(rows); K=J@J.T
        try:
            ev=torch.linalg.eigvalsh(K); ev=ev[ev>0]
            return round(float(ev.max()/ev.min()),2) if len(ev)>=2 else 1.0
        except: return -1.0

    @staticmethod
    def diagnose(kappa):
        if kappa<0:   return "error","Jacobian failed"
        if kappa<1e2: return "healthy","Well-conditioned"
        if kappa<1e4: return "moderate","Monitor loss variance"
        if kappa<1e6: return "ill","Consider ResNet or lower LR"
        return "severe","Switch ResNet + reduce depth"


# ════════════════════════════════════════════════════════════
# 10. META-LEARNER
#     Weighted average of best-run parameters as warm-start.
# ════════════════════════════════════════════════════════════
class MetaLearner:
    def __init__(self,lr_meta=0.05,cap=8):
        self.lr_meta=lr_meta; self.cap=cap
        self.snaps=[]; self.meta_params=None

    def record(self,model,score):
        state={k:v.clone().cpu() for k,v in model.state_dict().items()}
        self.snaps.append({"state":state,"score":float(score)})
        self.snaps.sort(key=lambda s:-s["score"])
        self.snaps=self.snaps[:self.cap]; self._update()

    def _update(self):
        if not self.snaps: return
        sc=np.array([s["score"] for s in self.snaps])
        w=np.exp(sc-sc.max()); w/=w.sum()
        keys=list(self.snaps[0]["state"].keys()); meta={}
        for k in keys:
            v0=self.snaps[0]["state"][k]
            if v0.dtype.is_floating_point:
                meta[k]=sum(wi*s["state"][k] for wi,s in zip(w,self.snaps))
            else: meta[k]=v0.clone()
        self.meta_params=meta

    def warm_start(self,model,noise=0.02):
        if self.meta_params is None: return False
        try:
            cur=model.state_dict(); new={}
            for k,v in cur.items():
                if k in self.meta_params and v.shape==self.meta_params[k].shape:
                    m=self.meta_params[k].to(v.device).to(v.dtype)
                    new[k]=m+noise*torch.randn_like(v)
                else: new[k]=v
            model.load_state_dict(new,strict=False); return True
        except: return False

    def state(self):
        return{"n_snaps":len(self.snaps),
               "best":round(self.snaps[0]["score"],4) if self.snaps else None}


# ════════════════════════════════════════════════════════════
# 11. REPLAY BUFFER
# ════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self,capacity=2048,ratio=0.3):
        self.capacity=capacity; self.ratio=ratio
        self.pts=[]; self.wts=[]

    def push(self,pts,residuals):
        pc=pts.detach().cpu(); rc=residuals.abs().detach().cpu().float()
        n=len(pc); k=max(1,n//2)
        _,idx=torch.topk(rc.view(-1),k)
        for i in idx: self.pts.append(pc[i]); self.wts.append(float(rc[i]))
        if len(self.pts)>self.capacity:
            p=sorted(zip(self.wts,self.pts),key=lambda x:-x[0])[:self.capacity]
            self.wts,self.pts=zip(*p) if p else ([],[])
            self.wts=list(self.wts); self.pts=list(self.pts)

    def sample(self,n,device,dtype):
        if len(self.pts)<n//4: return None
        k=min(n,len(self.pts)); w=np.array(self.wts[:k],dtype=np.float64)
        w/=w.sum()
        idx=np.random.choice(k,size=min(k,int(n*self.ratio)),replace=False,p=w)
        return torch.stack([self.pts[i] for i in idx]).to(device).to(dtype)

    def state(self): return{"size":len(self.pts),
                            "max_res":round(max(self.wts),4) if self.wts else 0}


# ════════════════════════════════════════════════════════════
# 12. HOMOSCEDASTIC WEIGHTER  (Kendall et al. 2018)
# ════════════════════════════════════════════════════════════
class HomoscedasticWeighter(nn.Module):
    def __init__(self,n_tasks,device=None,dtype=torch.float64):
        super().__init__()
        self.log_s=nn.Parameter(torch.zeros(n_tasks,device=device,dtype=dtype))

    def forward(self,losses):
        return sum(L/(2*torch.exp(2*s))+s for L,s in zip(losses,self.log_s))

    def weights(self):
        with torch.no_grad():
            return[round(float(1/(2*torch.exp(2*s))),4) for s in self.log_s]

    def state(self):
        return{"sigmas":[round(float(s.exp()),4) for s in self.log_s.detach()],
               "weights":self.weights()}


# ════════════════════════════════════════════════════════════
# 13. LAYER HEALTH MONITOR
# ════════════════════════════════════════════════════════════
class LayerHealthMonitor:
    def __init__(self): self.history=[]

    def check(self,model):
        stats={}
        for name,p in model.named_parameters():
            if p.grad is None: continue
            gn=float(p.grad.norm()); pn=float(p.data.norm())
            stats[name]={"gn":round(gn,6),"pn":round(pn,4),
                         "ratio":round(gn/(pn+1e-10),6)}
        if not stats: return {}
        gnorms=[v["gn"] for v in stats.values()]
        flags=[]
        if max(gnorms)>10: flags.append("exploding")
        if min(gnorms)<1e-8: flags.append("vanishing")
        for name,mod in model.named_modules():
            if isinstance(mod,nn.Linear) and mod.weight.grad is not None:
                dead=(mod.weight.grad.abs().sum(dim=1)==0).float().mean()
                if float(dead)>0.1: flags.append(f"dead:{name}:{float(dead):.2f}")
        r={"flags":flags,"max_g":round(max(gnorms),6),"min_g":round(min(gnorms),6)}
        self.history.append(r); return r

    def advice(self):
        if not self.history: return None
        f=self.history[-1].get("flags",[])
        if "exploding" in f: return "Exploding grads: reduce LR"
        if "vanishing" in f: return "Vanishing grads: use ResNet"
        return None


# ════════════════════════════════════════════════════════════
# 14. FOURIER SIGMA TUNER
# ════════════════════════════════════════════════════════════
class FourierSigmaTuner:
    def __init__(self): self.history=[]; self.current=1.0

    def update(self,spec):
        rec=spec.get("sigma_recommend",1.0); self.history.append(rec)
        self.current=0.3*rec+0.7*self.current
        return round(self.current,3)

    def get(self): return self.current
    def state(self): return{"sigma":round(self.current,3),
                            "hist":[round(v,3) for v in self.history[-5:]]}


# ════════════════════════════════════════════════════════════
# 15. CONFIG SCORER — kNN pre-filter
# ════════════════════════════════════════════════════════════
class ConfigScorer:
    def __init__(self,k=5): self.k=k; self.X=[]; self.y=[]

    def observe(self,cfg,score):
        self.X.append(_enc(cfg)); self.y.append(float(score))

    def score(self,cfg):
        if len(self.X)<self.k: return 0.5
        enc=_enc(cfg)
        dist=np.array([np.linalg.norm(enc-x) for x in self.X])
        idx=np.argsort(dist)[:self.k]; ws=1.0/(dist[idx]+1e-6)
        return float((ws*np.array(self.y)[idx]).sum()/ws.sum())

    def filter_top(self,cands,keep=0.5):
        if len(self.X)<self.k: return cands
        sc=sorted([(c,self.score(c)) for c in cands],key=lambda x:-x[1])
        return[c for c,_ in sc[:max(1,int(len(cands)*keep))]]
