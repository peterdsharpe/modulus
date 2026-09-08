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
from matplotlib.ticker import FuncFormatter, NullFormatter

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


def synth_table(g, key_labels, impls=("before", "after")):
    rows = []
    for key, byimpl in sorted(g.items()):
        t = {i: med([r["seconds"] for r in byimpl[i]]) for i in impls if byimpl[i]}
        cells = [fmt_s(t[i]) if i in t else "" for i in impls]
        sp = f"{t['before']/t['after']:.1f}x" if "after" in t else ""
        rows.append("| " + " | ".join(key_labels(key)) + " | " + " | ".join(cells) + f" | {sp} |")
    return rows


IMPLS = (  # (record label, legend label, colour)
    ("before", "main: full-mesh lookup table", RED),
    ("after", "this PR: lookup table or binary search, chosen by mesh shape", BLUE),
)
BACKINGS = (("memory", "mesh in memory", "-"), ("memmap", "mesh memory-mapped from disk", "--"))


def _legend(fig, backings=True):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=4, label=lab) for _, lab, c in IMPLS]
    if backings:
        handles += [Line2D([], [], color=INK2, lw=1.5, ls=ls, label=lab) for _, lab, ls in BACKINGS]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8.5)


def synth_figure(g, path, suptitle, xlabel, panel_key, panel_title):
    """One panel per value of ``panel_key`` (a component of the group key);
    colour = implementation, line style = storage backing."""
    panels = sorted({k[panel_key] for k in g})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, pv in zip(axes, panels):
        for impl, _, c in IMPLS:
            for b, _, ls in BACKINGS:
                pts = sorted((k[0], med([r["seconds"] for r in g[k][impl]]))
                             for k in g if k[panel_key] == pv and k[2] == b and g[k][impl])
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color=c, ls=ls, marker="o", ms=4, lw=1.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(panel_title(pv, g), fontsize=9.5, color=INK2)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("wall time [s]")
    fig.suptitle(suptitle, fontsize=10.5)
    _legend(fig)
    fig.savefig(OUT / path, bbox_inches="tight"); plt.close(fig)


p = D / "bench_slice_points_local.json"
if p.exists():
    g = group(json.load(open(p)), ("n_points", "k", "backing"))
    md += ["### Synthetic, `slice_points` on the full mesh", "",
           "Keep k random vertices of an N-vertex triangle mesh (2N random cells, one scalar and one vector point field), on one AGA node. Median of 3 runs.", "",
           "| N vertices | k kept | backing | main: lookup table | this PR: table or search, by shape | speed-up |", "|---|---|---|---|---|---|"]
    md += synth_table(g, lambda key: (f"{key[0]:,}", f"{key[1]:,}", key[2]))
    md.append("")
    synth_figure(g, "synthetic_full_mesh.png",
                 "Mesh.slice_points on a whole mesh: keep k random vertices of an N-vertex triangle mesh with 2N cells (median of 3 runs)",
                 "N, vertices in the mesh", 1, lambda k, g: f"keep k = {k:,} vertices")

# ---------- synthetic, reader path ----------
p = D / "bench_slice_points_block_local.json"
if p.exists():
    g = group(json.load(open(p)), ("n_points", "k_cells", "backing"))
    md += ["### Synthetic, the reader's path (cell block, then vertex compaction)", "",
           "`slice_cells(block)` then `slice_points(unique(cells))`: the mesh reader's per-sample operation, on one AGA node. The block is already small; the cost that matters is how compaction scales with the size of the mesh it came from. Median of 3 runs.", "",
           "| N vertices | block cells | kept vertices | backing | main: lookup table | this PR: table or search, by shape | speed-up |", "|---|---|---|---|---|---|---|"]
    def _labels(key):
        n, kc, b = key
        kp = next(r["k_points"] for i in ("after", "before") for r in g[key][i])
        return (f"{n:,}", f"{kc:,}", f"{kp:,}", b)
    md += synth_table(g, _labels)
    md.append("")
    def _panel_title(kc, g):
        kp = [next(r["k_points"] for i in ("after", "before") for r in g[k][i]) for k in g if k[1] == kc]
        return f"block of {kc:,} cells (about {med(kp):,.0f} vertices kept)"
    synth_figure(g, "synthetic_reader_path.png",
                 "The mesh reader's per-sample path: slice_cells(block) then slice_points(unique(block cells)), on an N-vertex triangle mesh (median of 3 runs)",
                 "N, vertices in the source mesh", 1, _panel_title)

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
           "| block cells | metric | main: lookup table | this PR: table or search, by shape | speed-up |", "|---|---|---|---|---|"]
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
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), constrained_layout=True)
    for ax, key, title, ylabel in (
            (axes[0], "subsample_cold_s", "cold sample: lazy load + cell block + vertex compaction", "wall time [s]"),
            (axes[1], "slice_points_warm_s", "slice_points alone, block already in memory", "wall time [s]"),
            (axes[2], "subsample_peak_rss_delta_mib", "peak resident memory added during the sample", "peak RSS increase [MiB]")):
        for j, ((nc,), byimpl) in enumerate(sorted(g.items())):
            for i, (impl, _, c) in enumerate(IMPLS[:2]):
                v = [r[key] for r in byimpl[impl] if key in r]
                x = j * 3 + i
                ax.scatter([x] * len(v), v, color=c, s=20, alpha=0.75, zorder=3)
                if v:
                    ax.hlines(med(v), x - 0.4, x + 0.4, color=c, lw=3, zorder=4)
        ax.set_xlim(-0.8, 4.8)
        ax.set_xticks([0.5, 3.5]); ax.set_xticklabels([f"{nc:,}" for (nc,) in sorted(g)])
        ax.set_xlabel("cells per sample block")
        ax.set_yscale("log"); ax.set_ylabel(ylabel)
        plain = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:g}")
        ax.yaxis.set_major_formatter(plain)
        allv = [r[key] for byimpl in g.values() for impl in ("before", "after") for r in byimpl[impl] if key in r]
        if max(allv) / min(allv) < 30:  # narrow range: label the minor ticks too
            ax.yaxis.set_minor_formatter(plain)
        else:
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=9.5, color=INK2)
        ax.grid(axis="x", visible=False)
    fig.suptitle("HiLiftAeroML boundary meshes (285M cells, 142M vertices, memory-mapped on Lustre), one fresh process per case", fontsize=10.5)
    handles = [Line2D([], [], color=c, lw=3, marker="o", ms=5, label=lab) for _, lab, c in IMPLS[:2]]
    handles += [Line2D([], [], color=INK2, lw=0, marker="o", ms=5, label="one case (6 per group)"),
                Line2D([], [], color=INK2, lw=3, label="median of the 6 cases")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8.5)
    fig.savefig(OUT / "hilift_aga.png", bbox_inches="tight"); plt.close(fig)

print("\n".join(md))
