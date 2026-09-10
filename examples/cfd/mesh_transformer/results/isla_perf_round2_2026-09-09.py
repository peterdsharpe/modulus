"""ISLA-PERF round 2 (notebook #sec-nb-isla-perf-prereg): with the point softmax fixed,
(a) the bf16-autocast output difference between the fast and the original point softmax
with identical weights (the exactness statement the bar requires), and (b) whether
compiling the per-layer geometry region (_geo_region: invariants -> routing bias ->
point->slice mix -> pooled invariants) buys a further step-time reduction now that
that region is the dominant cost, with its output difference against eager.

One GPU, bf16 autocast, AdamW step, batch 1, reference surface configuration
(12 layers, 256 slices, 3 scalar + 2 vector outputs); configurations
(192, 10k), (512, 10k), (192, 80k), (512, 40k); with and without geo_checkpoint.

Usage: python isla_perf_round2_2026-09-09.py <out.json>
"""
import json, sys, time
import numpy as np
import torch

import physicsnemo.experimental.nn.isla.model as isla_model
from physicsnemo.experimental.nn.isla import ISLA

out_path = sys.argv[1]
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
CONFIGS = [(192, 10_000), (512, 10_000), (192, 80_000), (512, 40_000)]
EAGER_GEO = isla_model._geo_region
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "exactness_bf16": [], "compile": []}


def inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, 3, generator=g) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, generator=g), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, generator=g), dim=-1)
    w = torch.rand(1, n, generator=g) + 0.5
    return [t.to(dev) for t in (pts, nrm, drv, w)]


def make(hidden, ckpt, fast=True):
    torch.manual_seed(0)
    return ISLA(hidden=hidden, n_layers=12, n_slices=256, out_scalars=3, out_vectors=2, geo_checkpoint=ckpt, fast_point_softmax=fast).to(dev)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm()), float((a.float() - b.float()).abs().max())


# (a) exactness under bf16 autocast: fast vs native point softmax, same weights
for hidden, n in ((192, 10_000), (192, 40_000), (512, 10_000)):
    fast = make(hidden, False, True).eval(); native = make(hidden, False, False).eval(); native.load_state_dict(fast.state_dict())
    pts, nrm, drv, w = inputs(n)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b = fast(pts, nrm, drv, w), native(pts, nrm, drv, w)
        a32, b32 = fast(pts, nrm, drv, w), native(pts, nrm, drv, w)
    r, m = rel(a, b); r32, m32 = rel(a32, b32)
    # bf16 roundoff reference: the difference two bf16 evaluations of the SAME model show against fp32
    rb, _ = rel(a, a32)
    res["exactness_bf16"].append({"hidden": hidden, "tokens": n, "rel_l2_fast_vs_native_bf16": r, "max_abs_bf16": m,
                                  "rel_l2_fast_vs_native_fp32": r32, "max_abs_fp32": m32, "rel_l2_bf16_vs_fp32_same_model": rb})
    print(res["exactness_bf16"][-1], flush=True)
    del fast, native; torch.cuda.empty_cache()


# (b) compiled geometry region
def bench(hidden, n, ckpt, compiled):
    isla_model._geo_region = torch.compile(EAGER_GEO, dynamic=False) if compiled else EAGER_GEO
    m = make(hidden, ckpt).train()
    pts, nrm, drv, w = inputs(n)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(pts, nrm, drv, w)
        out.float().square().mean().backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    times = []
    for _ in range(10):
        torch.cuda.synchronize(); t0 = time.perf_counter(); step(); torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(pts, nrm, drv, w).float().clone()
    r = {"hidden": hidden, "tokens": n, "geo_checkpoint": ckpt, "compiled_geo": compiled,
         "step_s_median": float(np.median(times)), "incremental_allocated_gib": (torch.cuda.max_memory_allocated() - resident) / 2**30,
         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    del m, opt; torch.cuda.empty_cache()
    isla_model._geo_region = EAGER_GEO
    return r, out


for ckpt in (False, True):
    for hidden, n in CONFIGS:
        try:
            r0, o0 = bench(hidden, n, ckpt, False)
            r1, o1 = bench(hidden, n, ckpt, True)
            rl, ma = rel(o1, o0)
            rec = {"hidden": hidden, "tokens": n, "geo_checkpoint": ckpt, "eager": r0, "compiled": r1,
                   "speedup": r0["step_s_median"] / r1["step_s_median"], "rel_l2_compiled_vs_eager_bf16": rl, "max_abs": ma}
        except Exception as e:  # noqa: BLE001 — record and continue; the compile path is the experiment
            rec = {"hidden": hidden, "tokens": n, "geo_checkpoint": ckpt, "error": str(e)[:300]}
            isla_model._geo_region = EAGER_GEO; torch.cuda.empty_cache()
        res["compile"].append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("eager", "compiled")}), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
