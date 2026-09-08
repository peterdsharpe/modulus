"""Tables and figures for the PR #1966 before/after benchmark comment.

Inputs (any may be missing):
  bench_slice_points_local.json        full-mesh slice_points, synthetic, local
  bench_slice_points_block_local.json  reader path (cell block -> compaction), synthetic, local
  bench1966.jsonl                      HiLiftAeroML reader path on AGA (lustre memmaps)
Outputs: markdown to stdout, PNGs into <outdir>.
"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
BLUE, RED, INK2, GRID = "#2a78d6", "#e34948", "#52514e", "#e1e0d9"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def med(v):
    return st.median(v) if v else float("nan")


def fmt_s(x):
    return f"{x*1e3:.0f} ms" if x < 1 else f"{x:.2f} s"


def group(recs, keys):
    g = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if "error" in r:
            continue
        g[tuple(r[k] for k in keys)][r["impl"]].append(r)
    return g


md = []

# ---------- synthetic, full mesh ----------
GREY = "#898781"


def load_with_searchonly(main_file, searchonly_file, keys):
    """Group the hybrid sweep and, if present, the rejected search-only sweep
    (its "after" records are relabelled "search_only")."""
    recs = json.load(open(main_file))
    so = D / searchonly_file
    if so.exists():
        recs += [dict(r, impl="search_only") for r in json.load(open(so)) if r["impl"] == "after"]
    return group(recs, keys)


def synth_table(g, key_labels, impls=("before", "after", "search_only")):
    rows = []
    for key, byimpl in sorted(g.items()):
        t = {i: med([r["seconds"] for r in byimpl[i]]) for i in impls if byimpl[i]}
        cells = [fmt_s(t[i]) if i in t else "" for i in impls]
        sp = f"{t['before']/t['after']:.1f}x" if "after" in t else ""
        rows.append("| " + " | ".join(key_labels(key)) + " | " + " | ".join(cells) + f" | {sp} |")
    return rows


def synth_figure(g, path, title, xlabel, emphasize, series_key):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for impl, c in (("before", RED), ("after", BLUE), ("search_only", GREY)):
        for b, ls in (("memory", "-"), ("memmap", "--")):
            for s in sorted({series_key(k) for k in g}):
                pts = sorted((k[0], med([r["seconds"] for r in g[k][impl]])) for k in g if k[2] == b and series_key(k) == s and g[k][impl])
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color=c, ls=":" if impl == "search_only" else ls, marker="o", ms=4,
                            alpha=0.95 if s == emphasize else 0.4, lw=1.8 if s == emphasize else 1.0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("time [s]")
    ax.set_title(title, loc="left", fontsize=9, color=INK2)
    fig.savefig(OUT / path); plt.close(fig)


p = D / "bench_slice_points_local.json"
if p.exists():
    g = load_with_searchonly(p, "bench1966_synth_full_searchonly.json", ("n_points", "k", "backing"))
    md += ["### Synthetic, `slice_points` on the full mesh", "",
           "Keep k random vertices of an N-vertex triangle mesh (2N random cells, one scalar and one vector point field), on one AGA node. Median of 3 runs. The third column is the first version of this PR (binary search always), kept to show why the algorithm is chosen by shape.", "",
           "| N vertices | k kept | backing | before | after (this PR) | search only (rejected) | speed-up after/before |", "|---|---|---|---|---|---|---|"]
    md += synth_table(g, lambda key: (f"{key[0]:,}", f"{key[1]:,}", key[2]))
    md.append("")
    synth_figure(g, "synthetic_full_mesh.png",
                 "slice_points on the full mesh, keeping k of N vertices\nbefore = red, after = blue, search-only = grey dotted\nsolid = in memory, dashed = memmap; k = 1e3 / 3e4 / 1e6, 3e4 emphasized",
                 "N vertices", 30000, lambda k: k[1])

# ---------- synthetic, reader path ----------
p = D / "bench_slice_points_block_local.json"
if p.exists():
    g = load_with_searchonly(p, "bench1966_synth_block_searchonly.json", ("n_points", "k_cells", "backing"))
    md += ["### Synthetic, the reader's path (cell block, then vertex compaction)", "",
           "`slice_cells(block)` then `slice_points(unique(cells))`: the mesh reader's per-sample operation, on one AGA node. The block is already small; the cost that matters is how compaction scales with the size of the mesh it came from. Median of 3 runs.", "",
           "| N vertices | block cells | kept vertices | backing | before | after (this PR) | search only | speed-up after/before |", "|---|---|---|---|---|---|---|---|"]
    def _labels(key):
        n, kc, b = key
        kp = next(r["k_points"] for i in ("after", "before") for r in g[key][i])
        return (f"{n:,}", f"{kc:,}", f"{kp:,}", b)
    md += synth_table(g, _labels)
    md.append("")
    synth_figure(g, "synthetic_reader_path.png",
                 "reader path: a block of 1e4 / 1e5 cells, then vertex compaction\nbefore = red, after = blue, search-only = grey dotted\nsolid = in memory, dashed = memmap; 1e4-cell block emphasized",
                 "N vertices of the source mesh", 10000, lambda k: k[1])

# ---------- HiLiftAeroML on AGA ----------
p = D / "bench1966.jsonl"
if p.exists():
    recs = [json.loads(l) for l in open(p) if l.strip()]
    for r in recs:  # cluster runs label the snapshots main / pr
        r["impl"] = {"main": "before", "pr": "after"}.get(r["impl"], r["impl"])
    errs = [r for r in recs if "error" in r]
    g = group(recs, ("n_cells",))
    md += ["### HiLiftAeroML boundaries on the AGA cluster (lustre memmaps, cold per case)", "",
           "Real reader operation on 285M-cell / 142M-vertex surfaces: lazy memmap load, contiguous cell block, vertex compaction. Each measurement is a fresh process on a case no other measurement touched (cold page cache); the two packages alternate. Median (min–max) over 6 cases per row.", "",
           "| block cells | metric | before (main) | after (PR) | speed-up |", "|---|---|---|---|---|"]
    def stat(rs, k):
        v = [r[k] for r in rs if k in r]
        return (med(v), min(v), max(v)) if v else (float("nan"),) * 3
    for (nc,), byimpl in sorted(g.items()):
        for key, label in (("subsample_cold_s", "cold sample (load + block + compaction)"),
                           ("subsample_second_block_s", "second block, same case (partly cached)"),
                           ("slice_points_warm_s", "slice_points alone, in-memory block"),
                           ("reader_getitem_s", "recipe reader `__getitem__`"),
                           ("subsample_peak_rss_delta_mib", "peak RSS increase during the sample")):
            (mb, lb, hb), (ma, la, ha) = stat(byimpl["before"], key), stat(byimpl["after"], key)
            if key.endswith("_mib"):
                md.append(f"| {nc:,} | {label} | {mb:,.0f} ({lb:,.0f}–{hb:,.0f}) MiB | {ma:,.0f} ({la:,.0f}–{ha:,.0f}) MiB | {mb/ma:.1f}x |")
            else:
                md.append(f"| {nc:,} | {label} | {fmt_s(mb)} ({fmt_s(lb)}–{fmt_s(hb)}) | {fmt_s(ma)} ({fmt_s(la)}–{fmt_s(ha)}) | {mb/ma:.1f}x |")
    if errs:
        md.append(f"\n{len(errs)} measurement(s) errored: " + "; ".join(f"{e['impl']} {e['case']} {e['n_cells']}" for e in errs))
    md.append("")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    for ax, key, title in ((axes[0], "subsample_cold_s", "cold sample: load + block + compaction [s]"),
                           (axes[1], "slice_points_warm_s", "slice_points alone, block in memory [s]"),
                           (axes[2], "subsample_peak_rss_delta_mib", "peak RSS increase during the sample [MiB]")):
        for j, ((nc,), byimpl) in enumerate(sorted(g.items())):
            for i, (impl, c) in enumerate((("before", RED), ("after", BLUE))):
                v = [r[key] for r in byimpl[impl] if key in r]
                x = j * 3 + i
                ax.scatter([x] * len(v), v, color=c, s=20, alpha=0.75, zorder=3)
                if v:
                    ax.hlines(med(v), x - 0.4, x + 0.4, color=c, lw=3, zorder=4)
        ax.set_xlim(-0.8, 4.8)
        ax.set_xticks([0.5, 3.5]); ax.set_xticklabels([f"{nc:,}-cell block" for (nc,) in sorted(g)])
        ax.set_yscale("log"); ax.set_title(title, loc="left", fontsize=9.5, color=INK2)
        ax.grid(axis="x", visible=False)
    fig.suptitle("HiLiftAeroML boundaries on AGA (lustre memmaps, 285M cells): before = main (red), after = this PR (blue); dots = cases, bars = medians",
                 fontsize=9.5, color=INK2, x=0.01, ha="left")
    fig.savefig(OUT / "hilift_aga.png"); plt.close(fig)

print("\n".join(md))
