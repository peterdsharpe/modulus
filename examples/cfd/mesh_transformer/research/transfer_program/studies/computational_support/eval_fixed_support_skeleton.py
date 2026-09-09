"""Sentinel-query evaluation with a FIXED per-case support (D1; skeleton, not run).

Mirrors the audit's companion-dependence probe
(results/isla_qt_companion_dependence_2026-09-08.json): hold 1,000 sentinel
query points fixed per car, vary how many companion queries are requested with
them (1k, 2k, 5k, 10k, 40k), and score the sentinels. For the support-token
configuration the support (boundaries.support, the first n_support interior
points of the case's deterministic sample) is loaded once per case and never
changes with the requested query set, so the prediction at a sentinel must be
independent of the companions up to the deployed precision's floor.

How the recipe produces a fixed support at inference: the dataset variant
drivaer_ml_volume_support.yaml draws the interior sample with the reader's
per-index generator (same sample for the same case index and seed) and the
SplitInteriorSupport transform moves its first n_support points into
boundaries.support. Loading the case through the same pipeline at inference
therefore yields the same support; the companions below are drawn from the
REMAINING interior points (and, for 40k, from a second pass of the reader with
a larger subsample_n_points), never from the support.

Usage (on the cluster, inside the recipe venv, after training):
  python eval_fixed_support_skeleton.py --run v0_isla_support12_seed42 \
      --checkpoint_dir $T/runs --dataset drivaer_ml_volume_support \
      --n_cars 4 --precision bfloat16 --out $T/v0_evals/support_companions.json
"""
import argparse, json

import torch

# The recipe's build helpers (imported lazily on the cluster; names as in infer.py).
# from datasets import build_dataloaders_for_split   # noqa
# from forward_kwargs import resolve_forward_kwargs   # noqa
# from output_normalize import normalize_output_to_tensordict  # noqa


def sentinel_study(model, case, sentinels, companion_counts, precision):
    """Return {n_companions: pressure error on the sentinels} for one case.

    `case` is the DomainMesh after the support split; `sentinels` a (1000, 3)
    index set into case.interior; companions are drawn deterministically
    (torch.Generator seeded by the case index) from the other interior points.
    """
    out = {}
    fk = None  # resolve_forward_kwargs(cfg.forward_kwargs, case) -> dict incl. support_* tensors
    for n_comp in companion_counts:
        # queries = sentinels + n_comp companions; support_* unchanged
        # with torch.autocast(device_type="cuda", dtype=precision):
        #     pred = model(**{**fk, "query_points": q, "query_normals": qn, "query_scalars": qs})
        # out[n_comp] = rel_l2(pred[:, :1000, pressure], true[sentinels])
        raise NotImplementedError("fill in with the recipe's loaders when the lanes have trained")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--dataset", default="drivaer_ml_volume_support"); ap.add_argument("--n_cars", type=int, default=4)
    ap.add_argument("--precision", default="bfloat16"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    companion_counts = [1000, 2000, 5000, 10000, 40000]
    json.dump({"skeleton": True, "companion_counts": companion_counts, "args": vars(args)}, open(args.out, "w"), indent=1)
