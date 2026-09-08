"""Before/after benchmark of Mesh.slice_points on the reader's path.

The reader keeps a contiguous block of k cells from an N-vertex mesh and then
drops the vertices the block does not reference: slice_cells(block) followed
by slice_points(unique(cells)). Here the mesh is already cell-sliced, so the
cost that matters is how slice_points scales with N when almost everything is
dropped. "before" = lookup-table algorithm (reproduced), "after" = current
Mesh.slice_points. Each (impl, case) runs in its own subprocess for peak RSS.

usage: python bench_slice_points_block.py driver <out.json> [1e6,1e7,1e8]
"""
import json, resource, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import torch
from physicsnemo.mesh import Mesh

sys.path.insert(0, str(Path(__file__).parent))
from bench_slice_points import make_mesh, reference_slice_points  # noqa: E402


def worker(impl, n, k_cells, backing, path):
    mesh = Mesh.load(path)
    if backing == "memory":
        mesh = Mesh(points=mesh.points.clone(), cells=mesh.cells.clone(),
                    point_data=mesh.point_data.clone(), cell_data=mesh.cell_data.clone())
    start = int(torch.randint(0, mesh.n_cells - k_cells, (1,), generator=torch.Generator().manual_seed(3)))
    block = mesh.slice_cells(slice(start, start + k_cells))
    referenced = torch.unique(block.cells)
    fn = reference_slice_points if impl == "before" else Mesh.slice_points
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter(); out = fn(block, referenced); dt = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"impl": impl, "n_points": n, "k_cells": k_cells, "k_points": int(referenced.numel()),
                      "backing": backing, "seconds": dt, "peak_rss_delta_mib": (r1 - r0) / 1024,
                      "n_points_out": out.n_points, "n_cells_out": out.n_cells}))


def driver(out_path, sizes):
    results = []
    with tempfile.TemporaryDirectory() as td:
        for n in sizes:
            path = Path(td) / f"m{n}.pmsh"
            make_mesh(n).save(path)
            for k_cells in (10_000, 100_000):
                for backing in ("memory", "memmap"):
                    for rep in range(3):
                        for impl in ("before", "after"):
                            r = subprocess.run([sys.executable, __file__, "worker", impl, str(n), str(k_cells), backing, str(path)],
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
        driver(sys.argv[2], sizes)
