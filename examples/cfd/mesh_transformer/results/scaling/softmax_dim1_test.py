"""Micro-test: is torch.softmax along the middle dimension of a (B, N, S) tensor
exact for N above 65,536 on this device?  Compares the ISLA reference kernel
(softmax over dim=1) with the fast kernel (transpose, softmax over the last
dim, transpose back) and with a float64 log-sum-exp reference, in float32 and
bfloat16, for N bracketing 2^16.  Also checks that the softmax output sums to
one over the points and that no column is uniform (the mean-field signature).
"""
import sys, json, torch

dev = torch.device("cuda")
torch.manual_seed(0)
out = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "rows": []}
S = 256
for N in [40000, 60000, 65535, 65536, 65537, 70000, 80000, 100000]:
    for dtype in (torch.float32, torch.bfloat16):
        x = (torch.randn(1, N, S, device=dev) * 3.0).to(dtype)
        ref = torch.softmax(x, dim=1)                                        # ISLA reference kernel
        fast = torch.softmax(x.transpose(1, 2), dim=-1).transpose(1, 2)      # ISLA fast kernel
        x64 = x.double()
        exact = torch.exp(x64 - torch.logsumexp(x64, dim=1, keepdim=True))
        row = {
            "N": N, "dtype": str(dtype).split(".")[-1],
            "ref_vs_exact_max": float((ref.double() - exact).abs().max()),
            "fast_vs_exact_max": float((fast.double() - exact).abs().max()),
            "ref_colsum_min": float(ref.double().sum(1).min()), "ref_colsum_max": float(ref.double().sum(1).max()),
            "fast_colsum_min": float(fast.double().sum(1).min()), "fast_colsum_max": float(fast.double().sum(1).max()),
            "exact_max_weight": float(exact.max()),
            "ref_max_weight": float(ref.double().max()),
        }
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
# the same check through the ISLA helper itself, if importable from the evaluation snapshot
try:
    sys.path.insert(0, sys.argv[1])
    from physicsnemo.experimental.nn.isla.model import _softmax_over_points
    for N in [60000, 80000]:
        x = torch.randn(1, N, S, device=dev) * 3.0
        a_ref = _softmax_over_points(x, fast=False).double()
        a_fast = _softmax_over_points(x, fast=True).double()
        exact = torch.exp(x.double() - torch.logsumexp(x.double(), dim=1, keepdim=True))
        row = {"helper_N": N, "ref_vs_exact": float((a_ref - exact).abs().max()), "fast_vs_exact": float((a_fast - exact).abs().max())}
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
except Exception as e:  # noqa: BLE001
    print("helper check skipped:", repr(e), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("DONE", flush=True)
