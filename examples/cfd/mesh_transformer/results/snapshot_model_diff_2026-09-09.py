"""SNAPSHOT-LADDER, step 2: is the code -> code_isla2 evaluation change in the model class or in the inputs?

Imports the training snapshot ($T/code, class physicsnemo.experimental.nn.mt2.MeshTransformer2) and grafts the
code_isla2 ISLA module onto it (the audit's trick), loads the SAME checkpoint state dict into both, and compares
outputs on identical inputs: fp32 (no autocast) and bf16 autocast, plus the per-block hidden states to locate the
first divergence. Also reports state-dict key/shape differences and the two classes' default hyperparameters.
Usage: python snapshot_model_diff.py <task_dir> <ckpt.mdlus> <out.json>
"""
import json, sys
import torch

T, CKPT, OUT = sys.argv[1:4]
import physicsnemo  # noqa: E402  (must be the $T/code snapshot)
import physicsnemo.experimental.nn as pnn  # noqa: E402
assert physicsnemo.__file__.startswith(f"{T}/code/"), physicsnemo.__file__
pnn.__path__.append(f"{T}/code_isla2/physicsnemo/experimental/nn")
from physicsnemo.experimental.nn.mt2 import MeshTransformer2 as Old  # noqa: E402
import physicsnemo.experimental.nn.isla as isla_mod  # noqa: E402
from physicsnemo.experimental.nn.isla import ISLA as New  # noqa: E402
assert isla_mod.__file__.startswith(f"{T}/code_isla2/"), isla_mod.__file__

dev = "cuda"
KW = dict(out_scalars=1, out_vectors=1, hidden=192, n_layers=12, n_slices=256, mlp_ratio=4, reference_length=8.0)
res = {"old_class_file": Old.__module__, "new_class_file": isla_mod.__file__}
torch.manual_seed(0)
old, new = Old(**KW).to(dev).eval(), New(**KW).to(dev).eval()
# state dict from the .mdlus (a zip with model.pt inside) via Module.load on the old class, then copy to the new
old.load(CKPT)
sd = old.state_dict()
missing, unexpected = new.load_state_dict(sd, strict=False)
res["state_dict"] = {"missing_in_new": list(missing), "unexpected_for_new": list(unexpected),
                     "n_params_old": sum(p.numel() for p in old.parameters()), "n_params_new": sum(p.numel() for p in new.parameters())}
print("state dict: missing", missing, "unexpected", unexpected, flush=True)
# default hyperparameters that could differ
attrs = ["use_measure_weights", "similarity_gauge", "parity_fix", "parity_gate_scale", "vector_basis", "odd_head", "reference_length", "eps",
         "seed_mode", "use_relational_geo", "interior_queries", "anchor_normal_rho", "query_independent", "raw_coord_channel", "use_local_features"]
res["attrs"] = {a: (getattr(old, a, "<absent>") if not torch.is_tensor(getattr(old, a, None)) else "tensor",
                    getattr(new, a, "<absent>") if not torch.is_tensor(getattr(new, a, None)) else "tensor") for a in attrs}
for a, (vo, vn) in res["attrs"].items():
    if vo != vn:
        print(f"ATTR DIFFERS {a}: old={vo!r} new={vn!r}", flush=True)

g = torch.Generator().manual_seed(0)
n = 10_000
pts = (torch.randn(1, n, 3, generator=g) * torch.tensor([3.0, 2.0, 1.0])).to(dev)
nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, generator=g), dim=-1).to(dev)
drv = torch.tensor([[1.0, 0.0, 0.0]], device=dev)
w = (torch.rand(1, n, generator=g) + 0.5).to(dev)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


with torch.no_grad():
    o32, n32 = old(pts, nrm, drv, measure_weights=w), new(pts, nrm, drv, measure_weights=w)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o16, n16 = old(pts, nrm, drv, measure_weights=w), new(pts, nrm, drv, measure_weights=w)
    # uniform weights: does the difference vanish when measure weights are constant?
    ou, nu = old(pts, nrm, drv, measure_weights=torch.ones_like(w)), new(pts, nrm, drv, measure_weights=torch.ones_like(w))
res["output_rel_l2"] = {"fp32": rel(n32, o32), "bf16": rel(n16, o16), "fp32_uniform_weights": rel(nu, ou),
                        "old_bf16_vs_fp32": rel(o16, o32), "new_bf16_vs_fp32": rel(n16, n32)}
print("output rel L2 new vs old:", res["output_rel_l2"], flush=True)

# per-block divergence: hook the encoder blocks of both models
acts = {"old": [], "new": []}
hooks = [b.register_forward_hook(lambda m, i, o, k="old": acts[k].append(o.detach().float())) for b in old.blocks]
hooks += [b.register_forward_hook(lambda m, i, o, k="new": acts[k].append(o.detach().float())) for b in new.blocks]
with torch.no_grad():
    old(pts, nrm, drv, measure_weights=w); new(pts, nrm, drv, measure_weights=w)
for h in hooks:
    h.remove()
res["per_block_rel_l2_fp32"] = [rel(a, b) for a, b in zip(acts["new"], acts["old"])]
print("per-block rel L2 (fp32):", [round(x, 5) for x in res["per_block_rel_l2_fp32"]], flush=True)
# embedding (seed) stage
with torch.no_grad():
    res["embed_weight_equal"] = bool(torch.equal(old.embed[0].weight, new.embed[0].weight))
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
