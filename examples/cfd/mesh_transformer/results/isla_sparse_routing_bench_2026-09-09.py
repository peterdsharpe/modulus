"""SPARSE: sparse anchor routing for ISLA — step time, peak memory, and output change.

Notebook #sec-nb-sparse-routing-prereg. Reference surface configuration (hidden 192,
12 layers, 256 slices, 1 scalar + 1 vector output), 10,000 tokens, batch 1, bf16
autocast on one GPU. For k in {dense, 128, 64, 32, 16} and geo_checkpoint in
{off, on}: forward+backward wall time (median of 10 after 3 warm-ups) and peak
allocated / reserved memory (incremental over the resident model + grads).

Output change: (a) at initialization, the relative L2 change of the full output and
of the scalar channel between dense and k-sparse routing with the same weights, on
a random anisotropic cloud; (b) on a trained checkpoint if one loads
(MeshTransformer2.*.mdlus from a HiLift 35-case run), on saved validation inputs
(the prediction artifact's points, cell normals, measure weights and unit drive).

Usage: python isla_sparse_routing_bench_2026-09-09.py <out.json> [run_dir] [pred_case_dir ...]
"""
import glob, json, os, sys, time
import numpy as np
import torch

from physicsnemo.experimental.nn.isla import ISLA

out_path = sys.argv[1]
run_dir = sys.argv[2] if len(sys.argv) > 2 else None
case_dirs = sys.argv[3:]
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
KS = [0, 128, 64, 32, 16]
N = 10_000
res = {"config": {"hidden": 192, "n_layers": 12, "n_slices": 256, "tokens": N, "dtype": "bf16 autocast", "gpu": torch.cuda.get_device_name(0)},
       "bench": {}, "init_output_change": {}, "checkpoint_output_change": {}}


def inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, 3, generator=g) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, generator=g), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, generator=g), dim=-1)
    w = torch.rand(1, n, generator=g) + 0.5
    return [t.to(dev) for t in (pts, nrm, drv, w)]


def set_k(m, k):
    for blk in m.blocks:
        blk.anchor_topk = int(k)


def bench(k, ckpt):
    torch.manual_seed(0)
    m = ISLA(hidden=192, n_layers=12, n_slices=256, out_scalars=1, out_vectors=1, anchor_topk=k, geo_checkpoint=ckpt).to(dev).train()
    pts, nrm, drv, w = inputs(N)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(points=pts, normals=nrm, drive=drv, measure_weights=w)
        out.float().square().mean().backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(10):
        torch.cuda.synchronize(); t0 = time.perf_counter(); step(); torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    r = {"step_s_median": float(np.median(times)), "step_s_min": float(min(times)),
         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
         "resident_gib": resident / 2**30, "incremental_allocated_gib": (torch.cuda.max_memory_allocated() - resident) / 2**30}
    del m, opt; torch.cuda.empty_cache()
    return r


for ckpt in (False, True):
    for k in KS:
        key = f"k{k or 'dense'}_ckpt{int(ckpt)}"
        res["bench"][key] = bench(k, ckpt)
        d = res["bench"][f"kdense_ckpt{int(ckpt)}"]
        r = res["bench"][key]
        r["speedup_vs_dense"] = d["step_s_median"] / r["step_s_median"]
        r["incremental_memory_vs_dense"] = r["incremental_allocated_gib"] / max(d["incremental_allocated_gib"], 1e-9)
        print(key, {kk: round(v, 4) for kk, v in r.items()}, flush=True)

# (a) initialization-level output change, fp32, same weights
torch.manual_seed(0)
m = ISLA(hidden=192, n_layers=12, n_slices=256, out_scalars=1, out_vectors=1).to(dev).eval()
pts, nrm, drv, w = inputs(N, seed=3)
with torch.no_grad():
    set_k(m, 0); ref = m(points=pts, normals=nrm, drive=drv, measure_weights=w).float()
    for k in KS[1:]:
        set_k(m, k); o = m(points=pts, normals=nrm, drive=drv, measure_weights=w).float()
        res["init_output_change"][f"k{k}"] = {"rel_l2_all": float((o - ref).norm() / ref.norm()), "rel_l2_scalar": float((o[..., 0] - ref[..., 0]).norm() / ref[..., 0].norm())}
        print("init", k, res["init_output_change"][f"k{k}"], flush=True)
del m

# (b) trained checkpoint on saved validation inputs
if run_dir:
    try:
        from physicsnemo.core.module import Module
        ck = sorted(glob.glob(f"{run_dir}/checkpoints/*.mdlus"), key=lambda p: int(p.split(".")[-2]))[-1]
        tm = Module.from_checkpoint(ck).to(dev).eval()
        res["checkpoint"] = {"path": ck, "class": type(tm).__name__, "n_blocks": len(tm.blocks)}

        def mm(path):
            d, name = os.path.split(path)
            meta = json.load(open(f"{d}/meta.json"))[name.replace(".memmap", "")]
            return torch.from_numpy(np.array(np.memmap(path, dtype=np.float32, mode="r", shape=tuple(meta["shape"]))))

        per_case = {}
        for cd in case_dirs:
            base = f"{cd}/_tensordict"
            p = mm(f"{base}/interior/_tensordict/points.memmap")[None].to(dev)
            n_ = mm(f"{base}/boundaries/vehicle/_tensordict/cell_data/normals.memmap")[None].to(dev)
            wt = mm(f"{base}/boundaries/vehicle/_tensordict/cell_data/_measure_weights.memmap").reshape(1, -1).to(dev)
            d_ = mm(f"{base}/global_data/U_inf_dir.memmap").reshape(1, 3).to(dev)
            with torch.no_grad():
                set_k(tm, 0); ref = tm(points=p, normals=n_, drive=d_, measure_weights=wt).float()
                row = {}
                for k in KS[1:]:
                    set_k(tm, k); o = tm(points=p, normals=n_, drive=d_, measure_weights=wt).float()
                    row[f"k{k}"] = {"rel_l2_all": float((o - ref).norm() / ref.norm()), "rel_l2_scalar": float((o[..., 0] - ref[..., 0]).norm() / ref[..., 0].norm())}
            per_case[os.path.basename(cd)] = row
            print("ckpt", os.path.basename(cd)[:30], {k: round(v["rel_l2_scalar"], 4) for k, v in row.items()}, flush=True)
        res["checkpoint_output_change"] = {"per_case": per_case,
                                           "median_rel_l2_scalar": {f"k{k}": float(np.median([c[f"k{k}"]["rel_l2_scalar"] for c in per_case.values()])) for k in KS[1:]},
                                           "median_rel_l2_all": {f"k{k}": float(np.median([c[f"k{k}"]["rel_l2_all"] for c in per_case.values()])) for k in KS[1:]}}
    except Exception as e:  # report, do not hide
        res["checkpoint_output_change"] = {"error": repr(e)}
        print("checkpoint test failed:", repr(e), flush=True)

json.dump(res, open(out_path, "w"), indent=1)
print("wrote", out_path)
