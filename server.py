"""server.py"""
import json, traceback, pathlib, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from .physics  import (compute_metrics, vis_2d, field_grid, compare_curve,
                       compute_total_loss, PDE_REGISTRY)
from .models   import build_model, ACTIVATIONS
from .solvers  import run_solver, diagnose, SOLVER_INFO
from .engine   import stream_autonomous

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
DTYPE  = torch.float64
torch.set_default_dtype(torch.float64)
_HERE  = pathlib.Path(__file__).parent.parent
_IDX   = _HERE / "static" / "index.html"
app    = FastAPI(title="PINN Research Engine")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.get("/")
async def root(): return HTMLResponse(_IDX.read_text(encoding="utf-8"))

@app.get("/api/health")
async def health():
    return JSONResponse({"status":"ok","device":str(DEVICE),"torch":torch.__version__,
                         "acts":len(ACTIVATIONS),"solvers":list(SOLVER_INFO.keys()),
                         "pdes":list(PDE_REGISTRY.keys())})

async def _stream_single(ws, cfg):
    pde=cfg.get("pde","harmonic"); act=cfg.get("suite",cfg.get("act","cos"))
    if act not in ACTIVATIONS: act="cos"
    epochs=int(cfg.get("epochs",2000)); width=int(cfg.get("width",128))
    depth=int(cfg.get("depth",5)); lr=float(cfg.get("lr",1e-3))
    n_col=int(cfg.get("n_colloc",2048)); lbfgs=bool(cfg.get("use_lbfgs",True))
    solver=cfg.get("solver","classic"); arch=cfg.get("arch","standard")
    async def send(d):
        try: await ws.send_text(json.dumps(d))
        except: pass
    await send({"type":"status","msg":f"{DEVICE} | {arch} | {solver} | {act}"})
    model=build_model(pde,act,width,depth,arch,DEVICE,DTYPE)
    await send({"type":"status","msg":f"{model.n_params():,} params"})
    last,hist,elapsed=await run_solver(model,pde,epochs,lr,n_col,send,DEVICE,DTYPE,
                                       solver=solver,do_lbfgs=lbfgs,label=act)
    diag_t,diag_m=diagnose(hist,last)
    fg=field_grid(model,pde,DEVICE,DTYPE)
    vis=vis_2d(model,pde,DEVICE,DTYPE)
    cmp=compare_curve(model,pde,DEVICE,DTYPE)
    await send({"type":"done","ratio":last["ratio"],"pct":last["pct"],"rel_l2":last["rel_l2"],
                "elapsed":round(elapsed,1),"n_params":model.n_params(),
                "vis_data":vis,"field_grid":fg,"compare":cmp,"history":hist,
                "act":act,"diag_title":diag_t,"diag_msg":diag_m})

@app.websocket("/ws")
async def ws_main(ws: WebSocket):
    await ws.accept()
    try:
        raw=await ws.receive_text(); cfg=json.loads(raw)
        if cfg.get("mode")=="single": await _stream_single(ws,cfg)
        elif cfg.get("mode")=="autonomous": await stream_autonomous(ws,cfg,DEVICE,DTYPE)
        else: await ws.send_text(json.dumps({"type":"error","msg":"Unknown mode"}))
    except WebSocketDisconnect: pass
    except Exception as e:
        try: await ws.send_text(json.dumps({"type":"error","msg":str(e),"tb":traceback.format_exc()}))
        except: pass
    finally:
        try: await ws.close()
        except: pass
