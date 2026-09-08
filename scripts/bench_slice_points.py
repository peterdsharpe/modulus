"""Before/after benchmark of Mesh.slice_points.

"before" = the lookup-table algorithm (an arange and an old->new map over all
points, fancy-index gathers), reproduced here verbatim; "after" = the current
Mesh.slice_points on the import path. Each (impl, case) runs in its own
subprocess so peak RSS is attributable. Cases: keep k random vertices of an
N-vertex triangle mesh with one scalar and one vector point field, in memory
and loaded from a memmap on disk. Output: JSON lines on stdout.

usage: python bench_slice_points.py driver <out.json> [--sizes 1e6,1e7,1e8]
       python bench_slice_points.py worker <impl> <N> <k> <backing>
"""
import json, resource, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import torch
from physicsnemo.mesh import Mesh


def reference_slice_points(mesh, indices):
    """The previous Mesh.slice_points algorithm (lookup table + fancy gather)."""
    all_indices = torch.arange(mesh.n_points, device=mesh.points.device)
    kept = all_indices[indices]
    old_to_new = torch.full((mesh.n_points,), -1, dtype=torch.long)
    old_to_new[kept] = torch.arange(len(kept), dtype=torch.long)
    remapped = old_to_new[mesh.cells]
    valid = (remapped >= 0).all(dim=-1)
    return Mesh(
        points=mesh.points[kept],
        cells=remapped[valid],
        point_data=mesh.point_data[kept],
        cell_data=mesh.cell_data[valid],
        global_data=mesh.global_data,
    )


def make_mesh(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 3, generator=g)
    cells = torch.randint(0, n, (2 * n, 3), generator=g)
    return Mesh(points=pts, cells=cells,
                point_data={"f": torch.rand(n, generator=g), "v": torch.rand(n, 3, generator=g)})


def worker(impl, n, k, backing, path):
    torch.manual_seed(1)
    if backing == "memmap":
        mesh = Mesh.load(path)
    else:
        mesh = Mesh.load(path)
        mesh = Mesh(points=mesh.points.clone(), cells=mesh.cells.clone(),
                    point_data=mesh.point_data.clone(), cell_data=mesh.cell_data.clone())
    kept = torch.randperm(n)[:k].sort().values  # sorted unique, like a reader's unique(cells)
    fn = reference_slice_points if impl == "before" else Mesh.slice_points
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter(); out = fn(mesh, kept); dt = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"impl": impl, "n_points": n, "k": k, "backing": backing,
                      "seconds": dt, "peak_rss_delta_mib": (rss1 - rss0) / 1024,
                      "n_points_out": out.n_points, "n_cells_out": out.n_cells}))


def driver(out_path, sizes):
    results = []
    with tempfile.TemporaryDirectory() as td:
        for n in sizes:
            path = Path(td) / f"m{n}.pmsh"
            make_mesh(n).save(path)
            for k in (1_000, 30_000, 1_000_000):
                if k >= n:
                    continue
                for backing in ("memory", "memmap"):
                    for rep in range(3):
                        for impl in ("before", "after"):
                            r = subprocess.run([sys.executable, __file__, "worker", impl, str(n), str(k), backing, str(path)],
                                               capture_output=True, text=True, check=True)
                            rec = json.loads(r.stdout.strip().splitlines()[-1]); rec["rep"] = rep
                            results.append(rec); print(rec, flush=True)
                            json.dump(results, open(out_path, "w"), indent=1)
            shutil.rmtree(path, ignore_errors=True)  # a saved .pmsh is a directory


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
    else:
        sizes = [int(float(s)) for s in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1e6", "1e7", "1e8"])]
        driver(sys.argv[2], sizes)  # usage: driver <out.json> [1e6,1e7,1e8]
