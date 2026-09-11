"""Routing-mass probe for ISLA context tokens (wake / latent volume tokens).

Reproduces the recipe's inference construction (same Hydra config, overrides,
checkpoint loader, collate) for ONE validation case, then records, in every
slice block, the final point->slice assignment ``a`` (normalized over points)
and the token positions ``r`` by wrapping ``_softmax_over_points`` and
``_SliceBlock.forward``. Reports per layer: how much slice mass the context
tokens and the query tokens hold, how many slices they dominate, where the
slice anchors sit relative to the surface (fraction farther than 0.05 and 0.2
body lengths from the nearest surface token; number downstream of the rear),
and each context token's own mass. Usage (from $T/recipe, PYTHONPATH with the
run's code snapshot first):

  python probe_routing_mass.py <out.json> <hydra overrides as for infer.py>
"""
import json
import os
import sys

import torch

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
sys.path.insert(0, f"{T}/recipe/src")

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from datasets import build_dataloaders  # noqa: E402
from utils import recursive_to_device  # noqa: E402

from physicsnemo import datapipes  # noqa: F401,E402  (registers ${dp:...})
from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402
import physicsnemo.experimental.nn.isla.model as isla_model  # noqa: E402

REC = {"a": [], "r": [], "d": []}
_orig_softmax = isla_model._softmax_over_points
_orig_forward = isla_model._SliceBlock.forward


def _rec_softmax(x, fast=True):
    a = _orig_softmax(x, fast)
    REC["a"].append(a.detach())
    return a


def _rec_forward(self, h, log_w, r, n_hat, d_hat, eps):
    REC["r"].append(r.detach())
    REC["d"].append(d_hat.detach())
    return _orig_forward(self, h, log_w, r, n_hat, d_hat, eps)


isla_model._softmax_over_points = _rec_softmax
isla_model._SliceBlock.forward = _rec_forward


def main():
    out_path = sys.argv[1]
    overrides = sys.argv[2:]
    DistributedManager.initialize()
    device = DistributedManager().device
    with initialize_config_dir(config_dir=f"{T}/recipe/conf", version_base=None):
        cfg = compose(config_name="infer", overrides=overrides)
    _train_loader, val_loader, _normalizer, _info = build_dataloaders(cfg)
    model = instantiate(cfg.model, _convert_="partial").to(device)
    ckpt = os.path.join(str(cfg.checkpoint_dir), str(cfg.run_id), "checkpoints")
    epoch = load_checkpoint(path=ckpt, models=model, device=device)
    assert epoch > 0, "no checkpoint restored"
    model.eval()
    dataset, sampler = val_loader.dataset, val_loader.sampler
    idx = next(iter(sampler))
    sample = dataset[idx]
    batch = recursive_to_device(val_loader.collate_fn([sample]), device)
    fk = batch["forward_kwargs"]
    nq = int(fk["query_points"].shape[1]) if fk.get("query_points") is not None else 0
    n_surface = int(fk["points"].shape[1])
    with torch.no_grad():
        model(**fk)
    n_blocks = len(REC["r"])
    # two softmax calls per slice block (routing, then final); LVT pre-assignment adds one call BEFORE the blocks
    a_calls = REC["a"]
    n_extra = len(a_calls) - 2 * n_blocks
    finals = [a_calls[n_extra + 2 * i + 1] for i in range(n_blocks)]
    res = {"run_id": str(cfg.run_id), "epoch": int(epoch), "n_surface": n_surface, "n_query": nq, "layers": []}
    for li, (a, r, d) in enumerate(zip(finals, REC["r"], REC["d"])):
        a = a[0].float(); r = r[0].float(); dv = d[0, 0].float()
        n_tokens, S = a.shape
        n_ctx = n_tokens - n_surface - nq
        sur = slice(0, n_surface); ctx = slice(n_surface, n_surface + n_ctx); qry = slice(n_surface + n_ctx, n_tokens)
        share_ctx = a[ctx].sum(0) if n_ctx else torch.zeros(S, device=a.device)
        share_qry = a[qry].sum(0) if nq else torch.zeros(S, device=a.device)
        z = a.T @ r  # (S,3) anchors
        proj_s = r[sur] @ dv
        body_len = float(proj_s.max() - proj_s.min())
        dmin = torch.cdist(z, r[sur]).min(dim=1).values / body_len
        rear = float(proj_s.max())
        layer = {
            "n_tokens": n_tokens, "n_ctx": n_ctx, "n_slices": S,
            "ctx_share_max": float(share_ctx.max()), "ctx_share_mean": float(share_ctx.mean()),
            "ctx_slices_dominated": int((share_ctx > 0.5).sum()),
            "qry_share_mean": float(share_qry.mean()), "qry_slices_dominated": int((share_qry > 0.5).sum()),
            "anchors_frac_beyond_0.05Lb": float((dmin > 0.05).float().mean()),
            "anchors_frac_beyond_0.2Lb": float((dmin > 0.2).float().mean()),
            "anchors_downstream_of_rear": int(((z @ dv) > rear).sum()),
            "anchor_surface_dist_Lb_max": float(dmin.max()),
        }
        if n_ctx:
            tok_mass = a[ctx].sum(1)  # slice-units held by each context token (sum over all tokens = S)
            layer["ctx_token_mass_top5"] = [round(float(v), 4) for v in tok_mass.topk(min(5, n_ctx)).values]
            layer["ctx_token_mass_sum"] = float(tok_mass.sum())
            layer["ctx_pos_along_drive_Lb"] = [round(float(v), 3) for v in ((r[ctx] @ dv - rear) / body_len)[: min(8, n_ctx)]]
        res["layers"].append(layer)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    for li, L in enumerate(res["layers"]):
        print(li, {k: L[k] for k in ("ctx_share_max", "ctx_slices_dominated", "ctx_token_mass_sum", "qry_share_mean", "qry_slices_dominated", "anchors_frac_beyond_0.05Lb", "anchors_frac_beyond_0.2Lb", "anchors_downstream_of_rear") if k in L})
    print("wrote", out_path)


if __name__ == "__main__":
    main()
