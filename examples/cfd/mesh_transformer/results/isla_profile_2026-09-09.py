"""ISLA-PERF step 1: where does an ISLA training step spend its time, and why does it
scale superlinearly in tokens? (notebook #sec-nb-isla-perf-prereg)

Reference surface configuration (12 layers, 256 slices, 3 scalar + 2 vector outputs,
bf16 autocast, AdamW step, batch 1, one GPU, synthetic anisotropic cloud so no data
loading enters). For each (hidden, tokens) configuration: median step time over 10
steps after 3 warm-ups, peak allocated / reserved memory, and a torch.profiler trace
of 3 steps whose CUDA kernel time is bucketed by operator family:

  invariants   : sub/norm/div/mul/sum/log/cat/expand over the (B,N,S,.) geometry
  geo_linear   : the 8->1 routing-bias Linear and the 8->hidden/2 pooled projection
  softmax_pts  : softmax over points (dim=1 of (B,N,S)) and its backward
  softmax_slc  : softmax over slices (last dim) and its backward
  einsum_bmm   : the bmm behind the slice-state / readback / pooled einsums
  linear_mlp   : all other Linear layers (assign, broadcast, MLPs, head)
  norm_act     : LayerNorm, GELU
  optimizer    : AdamW foreach kernels
  other        : everything else (copies, casts, fills, autograd glue)

Also the scaling exponents (log-log slopes) of step time in tokens at hidden 192
(10k, 20k, 40k, 80k) and in hidden at 10k tokens (192, 256, 384, 512), with and
without the exact geometry recompute (geo_checkpoint).

Usage: python isla_profile_2026-09-09.py <out.json>
"""
import json, re, sys, time
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from physicsnemo.experimental.nn.isla import ISLA

out_path = sys.argv[1]
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
CONFIGS = [(192, 10_000), (192, 20_000), (192, 40_000), (192, 80_000), (256, 10_000), (384, 10_000), (512, 10_000), (512, 40_000)]
PROFILE_AT = {(192, 10_000), (512, 10_000), (192, 80_000), (512, 40_000)}


def inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, 3, generator=g) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, generator=g), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, generator=g), dim=-1)
    w = torch.rand(1, n, generator=g) + 0.5
    return [t.to(dev) for t in (pts, nrm, drv, w)]


def make(hidden, ckpt):
    torch.manual_seed(0)
    return ISLA(hidden=hidden, n_layers=12, n_slices=256, out_scalars=3, out_vectors=2, geo_checkpoint=ckpt).to(dev).train()


def run(hidden, n, ckpt, do_profile):
    m = make(hidden, ckpt)
    pts, nrm, drv, w = inputs(n)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in m.parameters())

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
    r = {"hidden": hidden, "tokens": n, "geo_checkpoint": ckpt, "params": n_params,
         "step_s_median": float(np.median(times)), "step_s_min": float(min(times)),
         "tokens_per_s": n / float(np.median(times)),
         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
         "incremental_allocated_gib": (torch.cuda.max_memory_allocated() - resident) / 2**30}
    if do_profile:
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
        # aggregate CUDA kernel time by kernel name, then bucket
        by_kernel = {}
        for ev in prof.key_averages(group_by_input_shape=False):
            if ev.self_device_time_total <= 0:
                continue
            by_kernel[ev.key] = by_kernel.get(ev.key, 0.0) + ev.self_device_time_total / 3 / 1e3  # ms per step
        # point-softmax vs slice-softmax: identify by the kernel template names PyTorch uses
        buckets = {}
        top = sorted(by_kernel.items(), key=lambda kv: -kv[1])[:40]
        for name, ms in by_kernel.items():
            lname = name.lower()
            if "softmax" in lname:
                tag = "softmax_pts" if "spatial" in lname else "softmax_slc"
            elif re.search(r"adam|foreach|lerp|addcmul|addcdiv", lname):
                tag = "optimizer"
            elif re.search(r"bmm|baddbmm|batched|gemv", lname):
                tag = "einsum_bmm"
            elif re.search(r"gemm|cutlass|cublas|addmm|matmul|sm100|sm90|xmma", lname):
                tag = "linear_mlp"
            elif re.search(r"layer_norm|layernorm|gelu", lname):
                tag = "norm_act"
            elif re.search(r"elementwise|vectorized|reduce_kernel|unrolled|catarray|cat_|index|gather|scatter|fill|copy|where|clamp|log|norm|div|mul|sub|sum|pow|binaryfunctor|unaryfunctor|cunn", lname):
                tag = "invariants_or_elementwise"
            else:
                tag = "other"
            buckets[tag] = buckets.get(tag, 0.0) + ms
        r["profile_ms_per_step_by_bucket"] = buckets
        r["profile_total_cuda_ms_per_step"] = sum(by_kernel.values())
        r["top_kernels_ms_per_step"] = [{"kernel": k[:160], "ms": v} for k, v in top]
    del m, opt; torch.cuda.empty_cache()
    return r


res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "runs": []}
for ckpt in (False, True):
    for hidden, n in CONFIGS:
        try:
            r = run(hidden, n, ckpt, do_profile=(hidden, n) in PROFILE_AT)
        except torch.cuda.OutOfMemoryError as e:
            r = {"hidden": hidden, "tokens": n, "geo_checkpoint": ckpt, "oom": str(e)[:200]}
            torch.cuda.empty_cache()
        res["runs"].append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("top_kernels_ms_per_step",)}), flush=True)


def slope(xs, ys):
    xs, ys = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    return float(np.polyfit(xs, ys, 1)[0])


res["exponents"] = {}
for ckpt in (False, True):
    rr = [r for r in res["runs"] if r.get("geo_checkpoint") == ckpt and "step_s_median" in r]
    tok = [(r["tokens"], r["step_s_median"]) for r in rr if r["hidden"] == 192]
    wid = [(r["hidden"], r["step_s_median"]) for r in rr if r["tokens"] == 10_000]
    res["exponents"][f"ckpt{int(ckpt)}"] = {"tokens_at_h192": slope(*zip(*tok)) if len(tok) > 1 else None,
                                            "hidden_at_10k": slope(*zip(*wid)) if len(wid) > 1 else None,
                                            "points_tokens": tok, "points_hidden": wid}
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res["exponents"], indent=1))
