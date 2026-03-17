"""
engine.py v3 — Self-Directing Autonomous Research Engine
=========================================================
All 15 algorithms integrated into a closed feedback loop.

Self-correction decisions made automatically:
  • NTK κ > 1e4  → switch to ResNet architecture
  • SpectralAnalyzer high_freq → use Fourier arch with tuned σ
  • FailureMemory blacklist → skip region entirely
  • Plateau detected → trigger L-BFGS mid-training
  • NaN/Inf → auto-restart with lr × 0.1
  • PCGrad → applied when cosine similarity < -0.2
  • MetaLearner warm-start → applied to all Phase 2 models
  • ReplayBuffer → mixed into every batch after epoch 200
  • Homoscedastic weighting → replaces manual BC weight tuning
  • ConfigScorer pre-filter → discards bottom-50% candidates
  • BayesianOptimizer → suggests Phase 2 config from GP surface
"""

import math, time, copy, random, asyncio, traceback
from collections import defaultdict

import torch
import torch.nn as nn

from .physics   import (compute_metrics, vis_2d, field_grid, compare_curve,
                        compute_total_loss, adaptive_colloc, get_residual_field,
                        PDE_REGISTRY, Lx, T_END, PI)
from .models    import build_model, ACTIVATIONS
from .solvers   import diagnose, SOLVER_INFO
from .algorithms import (
    BayesianOptimizer, PCGrad, CurriculumScheduler, SelfDistiller,
    SpectralAnalyzer, ParetoTracker, NoveltySearch, FailureMemory,
    NTKMonitor, MetaLearner, ReplayBuffer, HomoscedasticWeighter,
    LayerHealthMonitor, FourierSigmaTuner, ConfigScorer,
)

_ACTS   = list(ACTIVATIONS.keys())
_WIDTHS = [32,64,96,128,192,256,320]
_DEPTHS = [3,4,5,6,7,8]
_LRS    = [0.04,0.02,0.01,5e-3,1e-3,5e-4,1e-4]
_ARCHS  = ["standard","fourier","resnet"]
_SOLS   = ["classic","adaptive","gradnorm"]


# ════════════════════════════════════════════════════════════
# CORE TRAINING LOOP  — used by both single and autonomous
# ════════════════════════════════════════════════════════════

async def _train(model, pde, epochs, lr, n_col, send_cb, device, dtype,
                 solver="classic", do_lbfgs=True, label="",
                 curriculum=None, distiller=None, replay=None,
                 homo_w=None, meta=None, phase=1):
    """
    Single training run with all active algorithm hooks.
    Returns (metrics, history, elapsed).
    """
    if curriculum is None: curriculum = CurriculumScheduler()
    layer_mon = LayerHealthMonitor()

    # Meta warm-start
    if meta is not None and phase >= 2:
        meta.warm_start(model, noise=0.015)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
          opt, T_max=epochs, eta_min=lr*0.01)
    if homo_w is not None:
        opt_hw = torch.optim.Adam(homo_w.parameters(), lr=lr*0.05)

    hist = {"epoch":[],"loss":[],"pde_loss":[],"ratio":[],"wave_loss":[]}
    ev   = max(1, epochs//60)
    t0   = time.time()
    nan_restart = [0]

    def _colloc():
        if pde in ("maxwell","heat","burgers"):
            x=(torch.rand(n_col,1,device=device,dtype=dtype)*Lx).requires_grad_(True)
            t=(torch.rand(n_col,1,device=device,dtype=dtype)*T_END).requires_grad_(True)
            return x,t
        else:
            x=((torch.rand(n_col,1,device=device,dtype=dtype)*(2*PI))
               .requires_grad_(True))
            return x,x

    for ep in range(1, epochs+1):
        opt.zero_grad()

        # Adaptive collocation every 100 epochs after warmup
        if solver=="adaptive" and ep%100==0 and ep>200:
            x,t_col = adaptive_colloc(model,pde,n_col,device,dtype)
            x=x.requires_grad_(True)
            if pde not in ("harmonic",):
                t_col=t_col.requires_grad_(True) if t_col is not None else x
        else:
            x,t_col = _colloc()

        # Replay buffer injection
        if replay is not None and ep > 200:
            rb = replay.sample(n_col//4, device, dtype)
            if rb is not None:
                if rb.shape[-1]==2:
                    rx,rt=rb[:,0:1].requires_grad_(True),rb[:,1:2].requires_grad_(True)
                    x=torch.cat([x,rx],0); t_col=torch.cat([t_col,rt],0)
                else:
                    x=torch.cat([x,rb.requires_grad_(True)],0)
                    t_col=x

        w_bc = curriculum.bc_w()
        L = compute_total_loss(model,x,t_col,ep,epochs,pde,device,dtype,w_bc)

        # NaN restart
        if not torch.isfinite(L["total"]):
            nan_restart[0]+=1
            if nan_restart[0]>3: break
            for g in opt.param_groups: g["lr"]*=0.1
            continue

        # Homoscedastic weighting (replaces manual w_bc after warmup)
        if homo_w is not None and ep > 100:
            task_losses = [L["pde"], L.get("left",L.get("bc",L["pde"])),
                           L.get("wave",torch.zeros(1,device=device,dtype=dtype).squeeze())]
            total_loss = homo_w(task_losses)
        else:
            total_loss = L["total"]

        # PCGrad on phase>=2, every 50 epochs
        if phase >= 2 and ep%50==0:
            key_l = [v for k,v in L.items()
                     if k!="total" and isinstance(v,torch.Tensor) and v.requires_grad]
            if len(key_l)>=2:
                try: PCGrad.apply(model, key_l[:3])
                except: total_loss.backward()
            else: total_loss.backward()
        else:
            total_loss.backward()

        # Distillation supplement
        if distiller is not None and ep > 50:
            Xb = torch.cat([x,t_col],1) if pde in ("maxwell","heat","burgers") else x
            try:
                dl=distiller.dist_loss(model,Xb.detach())
                dl.backward()
            except: pass

        # Gradient health check
        lh = layer_mon.check(model)
        if lh.get("flags"):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step(); sch.step()
        if homo_w is not None and ep > 100: opt_hw.step()

        # Hard LR drop at 60%
        if ep == int(epochs*0.60):
            for g in opt.param_groups: g["lr"]*=0.2

        # Curriculum adjustments
        adj = curriculum.update(L)
        if adj.get("reduce_lr"):
            for g in opt.param_groups: g["lr"]*=0.5

        # Push to replay buffer
        if replay is not None and ep%50==0:
            try:
                rf,xf = get_residual_field(model,pde,device,dtype,n=256)
                pts = torch.cat([xf.view(-1,1),
                                  torch.zeros_like(xf.view(-1,1))],1) \
                      if pde not in ("harmonic",) else xf.view(-1,1)
                replay.push(pts, rf)
            except: pass

        if ep%ev==0 or ep==epochs:
            m = compute_metrics(model,pde,device,dtype)
            hist["epoch"].append(ep); hist["loss"].append(round(float(total_loss),6))
            hist["pde_loss"].append(round(float(L["pde"]),6))
            hist["ratio"].append(round(m["ratio"],5))
            hist["wave_loss"].append(round(float(L.get("wave",0)),6))
            await send_cb({
                "type":"run_progress","label":label,"phase":phase,
                "epoch":ep,"n_epochs":epochs,
                "ratio":m["ratio"],"rel_l2":m["rel_l2"],"pct":m["pct"],
                "loss":float(total_loss),"pde_loss":float(L["pde"]),
                "wave_loss":float(L.get("wave",0)),
                "elapsed":round(time.time()-t0,2),"history":hist,
                "curriculum":curriculum.state(),
                "layer_flags":lh.get("flags",[]),
                "homo_weights":homo_w.weights() if homo_w else None,
            })
            await asyncio.sleep(0)

    # L-BFGS refinement
    if do_lbfgs:
        lb = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=40,
                                history_size=50, line_search_fn="strong_wolfe")
        def _cl():
            lb.zero_grad()
            _x,_t=_colloc()
            Lv=compute_total_loss(model,_x,_t,epochs,epochs,pde,device,dtype)["total"]
            Lv.backward(); return Lv
        try: lb.step(_cl)
        except: pass

    return compute_metrics(model,pde,device,dtype), hist, time.time()-t0


# ════════════════════════════════════════════════════════════
# POST-RUN ANALYSIS
# ════════════════════════════════════════════════════════════

def _analyze(model, pde, device, dtype, label,
             bayes, failure_mem, pareto, novelty,
             sigma_tuner, config_scorer, meta,
             cfg, metrics, elapsed, run_id):
    """Run all post-training analysis algorithms."""
    spec={}; ntk_cond=-1; nov=0.0

    try:
        res,xf = get_residual_field(model,pde,device,dtype)
        spec   = SpectralAnalyzer.analyze(res,xf)
        sigma_tuner.update(spec)
    except: pass

    try:
        if pde in ("maxwell","heat","burgers"):
            Xb=torch.rand(32,2,device=device,dtype=dtype)
        else:
            Xb=torch.rand(32,1,device=device,dtype=dtype)
        ntk_cond = NTKMonitor.condition_number(model,Xb)
    except: pass

    try:
        comp = compare_curve(model,pde,device,dtype)
        nov  = novelty.novelty(comp.get("pred",[]))
        novelty.add(comp.get("pred",[]), {"label":label})
    except: comp={}; pass

    score = 1.0 - metrics.get("rel_l2",1.0)
    bayes.observe(cfg, score)
    config_scorer.observe(cfg, score)

    if metrics["rel_l2"] > 0.85:
        failure_mem.record_fail(cfg)
    else:
        failure_mem.record_win(cfg, score)
        if meta is not None:
            meta.record(model, score)

    try:
        pareto.add(cfg, metrics, elapsed, model.n_params(), run_id, label)
    except: pass

    # Auto-correction advice from NTK
    ntk_status, ntk_advice = NTKMonitor.diagnose(ntk_cond)

    return {"spec":spec,"ntk_cond":ntk_cond,"ntk_status":ntk_status,
            "ntk_advice":ntk_advice,"novelty":nov,
            "compare":comp if "x" in comp else {}}


# ════════════════════════════════════════════════════════════
# AUTO-ARCH CORRECTION
#   Reads NTK + spectral results and possibly overrides arch.
# ════════════════════════════════════════════════════════════

def _maybe_override_arch(cfg, ntk_cond, spec, sigma_tuner):
    cfg = dict(cfg)
    if ntk_cond > 1e6 and cfg.get("arch") != "resnet":
        cfg["arch"]  = "resnet"
        cfg["depth"] = min(cfg.get("depth",5), 5)
        cfg["_note"] = "arch→resnet (NTK severe)"
    elif spec.get("high_freq") and cfg.get("arch") != "fourier":
        cfg["arch"]   = "fourier"
        cfg["_sigma"] = sigma_tuner.get()
        cfg["_note"]  = f"arch→fourier (high_freq σ={sigma_tuner.get()})"
    return cfg


# ════════════════════════════════════════════════════════════
# POPULATION GENERATION
# ════════════════════════════════════════════════════════════

def _gen_population(n, hof, bayes, failure_mem, config_scorer, pde):
    pool = []
    # HOF mutations
    for h in hof[:min(3,len(hof))]:
        c = dict(h["cfg"])
        p = random.choice(["act","width","depth","lr","arch","solver"])
        if   p=="act":    c["act"]   = random.choice(_ACTS)
        elif p=="width":  c["width"] = random.choice(_WIDTHS)
        elif p=="depth":  c["depth"] = random.choice(_DEPTHS)
        elif p=="lr":     c["lr"]    = random.choice(_LRS)
        elif p=="arch":   c["arch"]  = random.choice(_ARCHS)
        elif p=="solver": c["solver"]= random.choice(_SOLS)
        c["pde"] = pde
        if not failure_mem.blacklisted(c): pool.append(c)
    # Random fill
    while len(pool) < n*4:
        c={"act":random.choice(_ACTS),"arch":random.choice(_ARCHS),
           "width":random.choice(_WIDTHS),"depth":random.choice(_DEPTHS),
           "lr":random.choice(_LRS),"solver":random.choice(_SOLS),"pde":pde}
        if not failure_mem.blacklisted(c): pool.append(c)
    # Pre-filter via ConfigScorer
    pool = config_scorer.filter_top(pool, keep=0.6)
    # Bayesian suggestion as top candidate
    if len(pool) >= n:
        best = bayes.suggest(pool)
        selected = [best] + random.sample([c for c in pool if c is not best],
                                           min(n-1, len(pool)-1))
    else:
        selected = random.sample(pool, min(n, len(pool)))
    return selected[:n]


# ════════════════════════════════════════════════════════════
# MAIN AUTONOMOUS LOOP
# ════════════════════════════════════════════════════════════

async def stream_autonomous(ws, cfg_in: dict, device, dtype):
    pde         = cfg_in.get("pde","maxwell")
    p1_ep       = int(cfg_in.get("phase1_epochs",300))
    p2_ep       = int(cfg_in.get("phase2_epochs",1500))
    champ_ep    = int(cfg_in.get("champ_epochs", 3000))
    pop_size    = int(cfg_in.get("pop_size",6))
    max_gen     = int(cfg_in.get("max_gen",9999))
    n_col       = int(cfg_in.get("n_colloc",1024))
    use_pcgrad  = bool(cfg_in.get("use_pcgrad",True))
    use_distill = bool(cfg_in.get("use_distill",True))
    use_bayes   = bool(cfg_in.get("use_bayes",True))
    use_novelty = bool(cfg_in.get("use_novelty",True))

    # Algorithm instances (shared across all generations)
    bayes        = BayesianOptimizer()
    pareto       = ParetoTracker()
    novelty      = NoveltySearch()
    failure_mem  = FailureMemory()
    sigma_tuner  = FourierSigmaTuner()
    config_scorer= ConfigScorer()
    meta         = MetaLearner()
    replay       = ReplayBuffer(capacity=3000)

    hof          = []    # sorted by rel_l2, top-30
    best_ever    = None
    teacher      = None
    total_runs   = [0]
    start_t      = time.time()
    compare_all  = {"x":None,"true":None,"curves":{}}
    spectral_hist= []
    ntk_hist     = []

    def rt():
        s=int(time.time()-start_t)
        return f"{s//3600}h {(s%3600)//60}m {s%60}s"

    async def send(d):
        try:
            import json
            await ws.send_text(json.dumps(d))
        except: pass

    def hof_snap(n=15):
        return [{
            "rank":i+1,"run_id":h.get("run_id",0),
            "label":h["cfg"].get("act","?"),
            "arch":h["cfg"].get("arch","std"),
            "solver":h["cfg"].get("solver","cls"),
            "width":h["cfg"].get("width","?"),
            "depth":h["cfg"].get("depth","?"),
            "lr":h["cfg"].get("lr",0),
            "pde":pde,
            "ratio":round(h["metrics"]["ratio"],4),
            "rel_l2":round(h["metrics"]["rel_l2"],5),
            "rel_l2_e":f'{h["metrics"]["rel_l2"]:.2e}',
            "elapsed":round(h.get("elapsed",0),1),
            "gen":h.get("gen",0),
            "phase":h.get("phase","scan"),
            "novelty":round(h.get("novelty",0),3),
            "ntk_cond":h.get("ntk_cond",-1),
            "ntk_status":h.get("ntk_status","—"),
            "spectral":h.get("spec",{}),
            "note":h.get("note",""),
        } for i,h in enumerate(hof[:n])]

    def update_hof(cfg_, metrics, elapsed, gen, run_id, phase, extras={}):
        nonlocal best_ever, teacher
        entry = dict(extras)
        entry.update({"cfg":cfg_,"metrics":metrics,"elapsed":elapsed,
                      "gen":gen,"run_id":run_id,"phase":phase})
        hof.append(entry)
        hof.sort(key=lambda x:x["metrics"]["rel_l2"])
        del hof[30:]
        if best_ever is None or metrics["rel_l2"]<best_ever["metrics"]["rel_l2"]:
            best_ever = entry

    await send({"type":"autonomous_start","pde":pde,"pop_size":pop_size,
                "algos":{"bayes":use_bayes,"pcgrad":use_pcgrad,
                         "distill":use_distill,"novelty":use_novelty,
                         "meta":True,"replay":True,"homo":True,
                         "ntk":True,"spectral":True,"pareto":True,
                         "novelty_search":True,"failure_mem":True,
                         "config_scorer":True,"fourier_sigma":True},
                "pde_info":PDE_REGISTRY.get(pde,{})})

    for gen in range(max_gen):
        await send({"type":"generation_start","gen":gen,"runtime":rt(),
                    "n_hof":len(hof),
                    "bayes":bayes.state(),"failure":failure_mem.state(),
                    "meta":meta.state(),"replay":replay.state(),
                    "sigma":sigma_tuner.state(),
                    "pareto_front":pareto.front()[:6]})

        population = _gen_population(pop_size, hof, bayes, failure_mem,
                                      config_scorer, pde)

        # ── PHASE 1: Rapid scan ──────────────────────────────
        await send({"type":"phase","phase":1,"gen":gen,
                    "label":f"Gen {gen} · Phase 1 · {len(population)} configs × {p1_ep} ep"})

        gen_results = []

        for i, pcfg in enumerate(population):
            total_runs[0] += 1
            run_id = total_runs[0]
            act    = pcfg.get("act","cos")
            arch   = pcfg.get("arch","standard")
            label  = f"g{gen}r{i}·{act}·{arch[:3]}"

            await send({"type":"run_start","gen":gen,"run":i,
                        "total":len(population),"config":pcfg,
                        "label":label,"run_id":run_id,"phase":1})

            model = build_model(pde, act, pcfg.get("width",128),
                                pcfg.get("depth",5), arch, device, dtype)
            curr  = CurriculumScheduler()

            async def _cb(d,_l=label):
                if d.get("type")=="run_progress": await send({**d,"label":_l})

            try:
                metrics,hist,elapsed = await _train(
                    model,pde,p1_ep,pcfg.get("lr",1e-3),n_col,
                    _cb,device,dtype,
                    solver=pcfg.get("solver","classic"),
                    do_lbfgs=False,label=label,
                    curriculum=curr,replay=replay,phase=1)
            except Exception as e:
                await send({"type":"run_error","gen":gen,"run":i,
                            "label":label,"msg":str(e)})
                failure_mem.record_fail(pcfg); continue

            anl = _analyze(model,pde,device,dtype,label,
                           bayes,failure_mem,pareto,novelty,
                           sigma_tuner,config_scorer,meta,
                           pcfg,metrics,elapsed,run_id)

            # Auto-arch correction for Phase 2
            pcfg_corrected = _maybe_override_arch(
                pcfg, anl["ntk_cond"], anl["spec"], sigma_tuner)

            diag_t,diag_m = diagnose(hist, metrics)
            if anl["compare"]: compare_all.update({
                "x":compare_all.get("x") or anl["compare"].get("x"),
                "true":compare_all.get("true") or anl["compare"].get("true"),
            })
            if anl["compare"].get("pred"):
                compare_all["curves"][label] = anl["compare"]["pred"]
            elif anl["compare"].get("E_pred"):
                compare_all["curves"][label] = anl["compare"]["E_pred"]

            spectral_hist.append(anl["spec"])
            ntk_hist.append(anl["ntk_cond"])

            update_hof(pcfg,metrics,elapsed,gen,run_id,"scan",
                       {"spec":anl["spec"],"ntk_cond":anl["ntk_cond"],
                        "ntk_status":anl["ntk_status"],"novelty":anl["novelty"],
                        "note":pcfg_corrected.get("_note","")})

            gen_results.append({"cfg":pcfg,"cfg_c":pcfg_corrected,
                                 "metrics":metrics,"hist":hist,"elapsed":elapsed,
                                 "label":label,"analysis":anl})
            gen_results.sort(key=lambda r:r["metrics"]["rel_l2"])

            await send({"type":"run_done","gen":gen,"run":i,"label":label,
                        "metrics":metrics,"elapsed":elapsed,"phase":1,
                        "hall_of_fame":hof_snap(),"pareto_front":pareto.front()[:6],
                        "diag_title":diag_t,"diag_msg":diag_m,
                        "compare_data":compare_all,
                        "spectral":anl["spec"],"ntk_cond":anl["ntk_cond"],
                        "ntk_status":anl["ntk_status"],"ntk_advice":anl["ntk_advice"],
                        "novelty":anl["novelty"],
                        "arch_note":pcfg_corrected.get("_note",""),
                        "curriculum":curr.state(),"bayes":bayes.state(),
                        "failure":failure_mem.state(),"replay":replay.state(),
                        "meta":meta.state(),"sigma":sigma_tuner.state(),
                        "total_runs":total_runs[0],"runtime":rt()})
            await asyncio.sleep(0)

        # ── PHASE 2: Deep retrain top-2 + highest-novelty ───
        top_n = gen_results[:min(2,len(gen_results))]
        if use_novelty:
            for r in gen_results[2:5]:
                if r["analysis"]["novelty"] > 0.55:
                    top_n.append(r); break

        await send({"type":"phase","phase":2,"gen":gen,
                    "label":f"Gen {gen} · Phase 2 · {len(top_n)} configs × {p2_ep} ep + distill"})

        for r in top_n:
            pcfg_c = r["cfg_c"]   # potentially arch-corrected
            total_runs[0] += 1; run_id = total_runs[0]
            label = r["label"]+"_deep"

            await send({"type":"run_start","gen":gen,"run":-1,"total":len(top_n),
                        "config":pcfg_c,"label":label,"run_id":run_id,"phase":2})

            model = build_model(pde,pcfg_c.get("act","cos"),
                                pcfg_c.get("width",128),pcfg_c.get("depth",5),
                                pcfg_c.get("arch","standard"),device,dtype)
            curr  = CurriculumScheduler()
            homo  = HomoscedasticWeighter(
                3,device=device,dtype=dtype) if True else None
            dist  = SelfDistiller(teacher,alpha=0.25) \
                    if (use_distill and teacher is not None) else None

            async def _cb2(d,_l=label):
                if d.get("type")=="run_progress":
                    await send({**d,"label":_l,"phase":2})

            try:
                metrics,hist,elapsed = await _train(
                    model,pde,p2_ep,pcfg_c.get("lr",1e-3),n_col*2,
                    _cb2,device,dtype,
                    solver=pcfg_c.get("solver","classic"),
                    do_lbfgs=True,label=label,
                    curriculum=curr,distiller=dist,replay=replay,
                    homo_w=homo,meta=meta,phase=2)
            except Exception as e:
                await send({"type":"run_error","gen":gen,"run":-1,
                            "label":label,"msg":str(e)}); continue

            anl = _analyze(model,pde,device,dtype,label,
                           bayes,failure_mem,pareto,novelty,
                           sigma_tuner,config_scorer,meta,
                           pcfg_c,metrics,elapsed,run_id)
            fg  = field_grid(model,pde,device,dtype)
            vis = vis_2d(model,pde,device,dtype)

            # Update teacher if best
            if (use_distill and best_ever is not None and
                metrics["rel_l2"] < best_ever["metrics"]["rel_l2"]):
                teacher = copy.deepcopy(model)
                for p in teacher.parameters(): p.requires_grad_(False)
            elif use_distill and teacher is None:
                teacher = copy.deepcopy(model)
                for p in teacher.parameters(): p.requires_grad_(False)

            update_hof(pcfg_c,metrics,elapsed,gen,run_id,"deep",
                       {"spec":anl["spec"],"ntk_cond":anl["ntk_cond"],
                        "ntk_status":anl["ntk_status"],"novelty":anl["novelty"]})

            if anl["compare"].get("pred"):
                compare_all["curves"][label] = anl["compare"]["pred"]

            await send({"type":"deep_done","gen":gen,"label":label,"phase":2,
                        "metrics":metrics,"elapsed":elapsed,
                        "field_grid":fg,"vis_data":vis,
                        "compare_data":compare_all,
                        "hall_of_fame":hof_snap(),"pareto_front":pareto.front()[:6],
                        "spectral":anl["spec"],"ntk_cond":anl["ntk_cond"],
                        "ntk_status":anl["ntk_status"],"ntk_advice":anl["ntk_advice"],
                        "novelty":anl["novelty"],
                        "homo_state":homo.state() if homo else None,
                        "curriculum":curr.state(),"bayes":bayes.state(),
                        "total_runs":total_runs[0],"runtime":rt()})
            await asyncio.sleep(0)

        # ── CHAMPION: every 4 generations ───────────────────
        if gen > 0 and gen%4 == 0 and best_ever:
            bc = _maybe_override_arch(best_ever["cfg"],
                                       best_ever.get("ntk_cond",-1),
                                       best_ever.get("spec",{}), sigma_tuner)
            total_runs[0] += 1
            await send({"type":"champion_start","gen":gen,"config":bc,
                        "note":"Full training of all-time best config"})
            model = build_model(pde,bc.get("act","cos"),bc.get("width",128),
                                bc.get("depth",5),bc.get("arch","standard"),
                                device,dtype)
            curr  = CurriculumScheduler()

            async def _chcb(d):
                if d.get("type")=="run_progress":
                    await send({**d,"label":"CHAMPION","phase":3})

            try:
                metrics,hist,elapsed = await _train(
                    model,pde,champ_ep,bc.get("lr",1e-3),n_col*3,
                    _chcb,device,dtype,solver="classic",do_lbfgs=True,
                    label="CHAMPION",curriculum=curr,replay=replay,
                    homo_w=HomoscedasticWeighter(3,device=device,dtype=dtype),
                    meta=meta,phase=3)
                anl = _analyze(model,pde,device,dtype,"CHAMPION",
                               bayes,failure_mem,pareto,novelty,
                               sigma_tuner,config_scorer,meta,
                               bc,metrics,elapsed,total_runs[0])
                fg  = field_grid(model,pde,device,dtype)
                vis = vis_2d(model,pde,device,dtype)
                teacher = copy.deepcopy(model)
                for p in teacher.parameters(): p.requires_grad_(False)
                update_hof(bc,metrics,elapsed,gen,total_runs[0],"champion",
                           {"spec":anl["spec"],"ntk_cond":anl["ntk_cond"],
                            "ntk_status":anl["ntk_status"],"novelty":anl["novelty"]})
                await send({"type":"champion_done","gen":gen,
                            "metrics":metrics,"elapsed":elapsed,
                            "field_grid":fg,"vis_data":vis,
                            "compare_data":compare_all,
                            "hall_of_fame":hof_snap(),"pareto_front":pareto.front()[:6],
                            "spectral":anl["spec"],"ntk_cond":anl["ntk_cond"],
                            "total_runs":total_runs[0],"runtime":rt()})
            except Exception as e:
                await send({"type":"run_error","gen":gen,"run":-2,
                            "label":"champion","msg":str(e)})

        await send({"type":"generation_done","gen":gen,"runtime":rt(),
                    "total_runs":total_runs[0],
                    "best_rl2":best_ever["metrics"]["rel_l2"] if best_ever else 1.0,
                    "pareto_front":pareto.front(),"bayes":bayes.state(),
                    "meta":meta.state(),"replay":replay.state(),
                    "spectral_hist":[s.get("dominant_freq",0)
                                      for s in spectral_hist[-10:]],
                    "ntk_hist":[n for n in ntk_hist[-10:] if n>0]})
        await asyncio.sleep(0)

    await send({"type":"autonomous_done",
                "total_runs":total_runs[0],"runtime":rt()})
