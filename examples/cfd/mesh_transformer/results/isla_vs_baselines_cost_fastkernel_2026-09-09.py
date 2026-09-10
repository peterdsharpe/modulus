"""Matched-instrument cost comparison with the fast ISLA kernel (notebook #sec-nb-cost-matched-fast).

A minimal equivalent of the audit session's harness ($T/audit_mem/matched_memory.py,
which is not in the repository): one process, one GPU, batch 1, bf16 autocast, fp32
parameters, AdamW(lr=1e-3), MSE on a random 9-channel target, 3 warm-up + 5 measured
steps, the same synthetic inputs (positions ~ N(0,1) * (3,2,1), unit normals, measure
weights U(0.5,1.5), drive (1,0,0)) and the same four instruments per step:
  incremental peak allocated (reset_peak_memory_stats; max_memory_allocated - memory_allocated before the step),
  total peak allocated, memory_reserved after the step, cuda-synchronized step time (median of 5).
Summary values are the max over the five measured steps, as in the audit's artifact.

Configurations, all with the recipe's surface settings (out_dim 9 = 3 scalars + 2 vectors):
  ISLA  (physicsnemo.experimental.nn.isla): hidden 192 or 512; fast_point_softmax on/off; geo_checkpoint off/on
  GeoTransolver (geotransolver_surface.yaml: 12 layers, n_hidden 256 or 512, 256 slices, no local features)
  Transolver    (transolver_surface.yaml: 8 layers, n_hidden 256 or 512, 256 slices; fx = drive expanded per cell)
at 10,000 cells (reference and width 512) and 80,000 cells (reference width).

Usage: python isla_vs_baselines_cost_fastkernel_2026-09-09.py <out_json>
"""
import gc, json, statistics, sys, time
import torch

from physicsnemo.experimental.nn.isla import ISLA
from physicsnemo.models.geotransolver import GeoTransolver
from physicsnemo.models.transolver import Transolver

OUT = sys.argv[1]
dev = "cuda"
N_WARM, N_MEAS = 3, 5
GIB = 2**30
torch.manual_seed(0)
ISLA_KW = dict(out_scalars=3, out_vectors=2, n_layers=12, n_slices=256, mlp_ratio=4, reference_length=8.0)
GT_KW = dict(out_dim=9, functional_dim=6, geometry_dim=3, global_dim=3, n_layers=12, dropout=0.0, n_head=8, act="gelu",
             mlp_ratio=4, slice_num=256, use_te=False, plus=False, include_local_features=False,
             radii=[0.1, 0.5, 2.0], neighbors_in_radius=[16, 32, 64], n_hidden_local=32, state_mixing_mode="weighted")
TR_KW = dict(functional_dim=3, out_dim=9, embedding_dim=6, n_layers=8, dropout=0.0, n_head=8, act="gelu", mlp_ratio=4,
             slice_num=256, unified_pos=False, use_te=False, plus=False)


def make_inputs(n):
    g = torch.Generator(device="cpu").manual_seed(0)
    pts = (torch.randn(1, n, 3, generator=g) * torch.tensor([3.0, 2.0, 1.0])).to(dev)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, generator=g), dim=-1).to(dev)
    mw = (torch.rand(1, n, generator=g) + 0.5).to(dev)
    drive = torch.tensor([[1.0, 0.0, 0.0]], device=dev)
    target = torch.randn(1, n, 9, generator=g).to(dev)
    return pts, nrm, mw, drive, target


def run(name, desc, build, fwd, kw, n):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    pts, nrm, mw, drive, target = make_inputs(n)
    rec = dict(name=name, description=desc, config=kw, tokens=n)
    try:
        torch.manual_seed(1)
        m = build().to(dev)
        rec["n_params"] = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fwd(m, pts, nrm, mw, drive)
            loss = torch.nn.functional.mse_loss(out.float(), target)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            return loss

        for _ in range(N_WARM):
            step()
        torch.cuda.synchronize()
        rec["resident_allocated_after_warmup_bytes"] = torch.cuda.memory_allocated()
        inc, tot, res, tms = [], [], [], []
        for _ in range(N_MEAS):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); b = torch.cuda.memory_allocated()
            t0 = time.perf_counter(); step(); torch.cuda.synchronize(); tms.append((time.perf_counter() - t0) * 1e3)
            p = torch.cuda.max_memory_allocated(); inc.append(p - b); tot.append(p); res.append(torch.cuda.memory_reserved())
        rec.update(incremental_peak_allocated_bytes=inc, total_peak_allocated_bytes=tot, reserved_after_step_bytes=res, step_ms=tms,
                   step_ms_median=statistics.median(tms),
                   summary_gib=dict(incremental_peak_allocated=max(inc) / GIB, total_peak_allocated=max(tot) / GIB, reserved_after_step=max(res) / GIB),
                   status="ok")
        print(f"{name:26s} N={n:6d} params {rec['n_params']/1e6:6.2f}M  inc {max(inc)/GIB:6.2f} GiB  tot {max(tot)/GIB:6.2f}  reserved {max(res)/GIB:6.2f}  step {statistics.median(tms):8.1f} ms", flush=True)
        del m, opt
    except torch.OutOfMemoryError as e:
        rec["status"] = "oom"; rec["error"] = str(e)[:300]; print(f"{name} N={n}: OOM", flush=True)
    except Exception as e:  # noqa: BLE001 — record and continue
        import traceback
        rec["status"] = "error"; rec["error"] = traceback.format_exc()[-1500:]; print(f"{name} N={n}: ERROR {e!r}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    return rec


def isla_fwd(m, pts, nrm, mw, drive):
    return m(pts, nrm, drive, measure_weights=mw)


def gt_fwd(m, pts, nrm, mw, drive):
    return m(local_embedding=torch.cat([pts, nrm], dim=-1), local_positions=pts, global_embedding=drive[:, None, :], geometry=pts)


def tr_fwd(m, pts, nrm, mw, drive):
    return m(fx=drive[:, None, :].expand(pts.shape[0], pts.shape[1], 3), embedding=torch.cat([pts, nrm], dim=-1))


def isla(hidden, fast, ckpt):
    kw = dict(ISLA_KW, hidden=hidden, fast_point_softmax=fast, geo_checkpoint=ckpt)
    return (f"isla_h{hidden}_{'fast' if fast else 'orig'}_{'ckpt' if ckpt else 'plain'}",
            f"ISLA hidden {hidden}, fast_point_softmax={fast}, geo_checkpoint={ckpt}", lambda: ISLA(**kw), isla_fwd, kw)


def gt(width):
    kw = dict(GT_KW, n_hidden=width)
    return (f"gt_h{width}", f"GeoTransolver n_hidden {width}", lambda: GeoTransolver(**kw), gt_fwd, kw)


def tr(width):
    kw = dict(TR_KW, n_hidden=width)
    return (f"transolver_h{width}", f"Transolver n_hidden {width}", lambda: Transolver(**kw), tr_fwd, kw)


PLAN = []
for n in (10_000, 80_000):
    for fast in (True, False):
        for ckpt in (False, True):
            PLAN.append((*isla(192, fast, ckpt), n))
    PLAN += [(*gt(256), n), (*tr(256), n)]
for cfg in (isla(512, True, False), isla(512, True, True), isla(512, False, False), gt(512), tr(512)):
    PLAN.append((*cfg, 10_000))

results = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda, batch=1, n_warmup=N_WARM, n_measured=N_MEAS,
               precision="bf16 autocast, fp32 params, AdamW(lr=1e-3), MSE on random 9-channel target",
               instruments={"incremental_peak_allocated": "reset_peak_memory_stats(); b=memory_allocated(); step; max_memory_allocated()-b",
                            "total_peak_allocated": "max_memory_allocated() after step", "reserved_after_step": "memory_reserved() after step; empty_cache() between configurations",
                            "step_ms": "cuda-synchronized wall time per step, median of 5"},
               aggregation="summary_gib = max over the 5 measured steps", configs=[])
for cfg in PLAN:
    results["configs"].append(run(*cfg))
    json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
