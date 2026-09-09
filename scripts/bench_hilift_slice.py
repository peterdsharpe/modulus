"""Before/after benchmark of Mesh.slice_points on HiLiftAeroML boundaries.

Two package snapshots (``main`` = upstream/main, ``pr`` = PR #1966) are compared
on the reader path the recipe uses for every training sample: load a boundary
mesh lazily from its memmap files, take a contiguous block of cells, drop the
vertices the block does not reference. Each measurement runs in a fresh
subprocess (cold process, cold page cache for its case) with the snapshot on
PYTHONPATH, and the two snapshots alternate over disjoint cases so file-system
drift cancels. Records wall time and peak RSS.

usage: python bench_hilift_slice.py driver <out.jsonl> <code_main> <code_pr> <recipe_src>
       python bench_hilift_slice.py worker <case_dir> <n_cells>
"""
import json, os, resource, subprocess, sys, time

ROOT = os.environ.get("HILIFT_ROOT", "/path/to/PhysicsNeMo-HighLiftAeroML")


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def worker(case_dir, n_cells):
    import torch
    from physicsnemo.mesh import Mesh
    from physicsnemo.datapipes.readers.mesh import _cyclic_block_indices, _subsample_mesh_cells
    from merge_global_data import MeshReaderWithGlobalData

    rec = {"case": os.path.basename(case_dir), "n_cells": n_cells, "impl": os.environ["BENCH_IMPL"]}
    bpath = f"{case_dir}/_tensordict/boundaries/vehicle"

    # (1) the exact reader operation: lazy load, cell block, vertex compaction
    r0 = rss_mib(); t0 = time.perf_counter()
    mesh = Mesh.load(bpath)
    idx = _cyclic_block_indices(mesh.n_cells, n_cells, generator=torch.Generator().manual_seed(0), device=mesh.cells.device)
    sub = _subsample_mesh_cells(mesh, n_cells, generator=torch.Generator().manual_seed(0))
    rec["subsample_cold_s"] = time.perf_counter() - t0
    rec["subsample_peak_rss_delta_mib"] = rss_mib() - r0
    rec["n_points_full"], rec["n_cells_full"] = mesh.n_points, mesh.n_cells
    rec["n_points_out"], rec["n_cells_out"] = sub.n_points, sub.n_cells
    # a second, different block on the same (now partly cached) case
    t0 = time.perf_counter()
    sub2 = _subsample_mesh_cells(mesh, n_cells, generator=torch.Generator().manual_seed(1))
    rec["subsample_second_block_s"] = time.perf_counter() - t0
    # slice_points alone on an already-sliced (in-memory) block: pure CPU cost
    blk = mesh.slice_cells(idx)
    ref = torch.unique(blk.cells)
    t0 = time.perf_counter(); blk.slice_points(ref); rec["slice_points_warm_s"] = time.perf_counter() - t0

    # (2) the recipe reader end to end (one sample), separate cold case not needed:
    #     same case, block seed 2 -> mostly warm; reports pipeline overhead
    reader = MeshReaderWithGlobalData(ROOT, pattern=f"{os.path.basename(case_dir.rsplit('/',1)[0])}/*.pdmsh/_tensordict/boundaries/vehicle",
                                      subsample_n_cells=n_cells, merge_global_data_from="../../global_data")
    reader.set_generator(torch.Generator().manual_seed(2))
    t0 = time.perf_counter(); m, _ = reader[0]; rec["reader_getitem_s"] = time.perf_counter() - t0
    rec["peak_rss_mib"] = rss_mib()
    print(json.dumps(rec), flush=True)


def driver(out, code_main, code_pr, recipe_src):
    cases = sorted(d for d in os.listdir(ROOT) if d.startswith("geo_LHC"))
    # 12 well-separated cases; BENCH_CASE_OFFSET picks a disjoint dozen for a
    # second round, BENCH_FIRST=pr swaps which package goes first.
    start = int(os.environ.get("BENCH_CASE_OFFSET", "50"))
    picks = [cases[i] for i in range(start, start + 12 * 137, 137)]
    snaps = {"main": code_main, "pr": code_pr}
    order = [("main", "pr"), ("pr", "main")] * 3  # alternate first-mover
    if os.environ.get("BENCH_FIRST") == "pr":
        order = [(b, a) for a, b in order]
    with open(out, "a") as f:
        for n_cells in (10_000, 100_000):
            ci = 0
            for a, b in order:
                for impl in (a, b):
                    case = picks[ci % len(picks)]; ci += 1
                    case_dir = f"{ROOT}/{case}/domain_{case[4:]}.pdmsh"
                    env = dict(os.environ, PYTHONPATH=f"{snaps[impl]}:{recipe_src}", BENCH_IMPL=impl, OMP_NUM_THREADS="8")
                    r = subprocess.run([sys.executable, __file__, "worker", case_dir, str(n_cells)], env=env, capture_output=True, text=True)
                    line = (r.stdout.strip().splitlines() or [""])[-1]
                    if not line.startswith("{"):
                        line = json.dumps({"impl": impl, "case": case, "n_cells": n_cells, "error": r.stderr[-800:]})
                    print(line, flush=True); f.write(line + "\n"); f.flush()


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        driver(*sys.argv[2:6])
