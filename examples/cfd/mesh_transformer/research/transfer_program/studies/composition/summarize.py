"""Print the tables used in NOTEBOOK.md from results.json."""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results.json")))
print("checks:", {k: (f'{v["sym_rel"]:.1e}', f'{v["min_eig_over_norm"]:.1e}', f'{v["null_vector_residual"]:.1e}') for k, v in r["response_checks"].items()})
print("composition law", r["composition_law_rel_error"], "exact assembly", r["exact_assembly_error"], "seconds", round(r["seconds"]))
prop = r["propagation"]
for var in ("diffusion", "screened"):
    for lay in ("chain", "grid"):
        for N in (2, 4, 8, 16):
            rows = [x for x in prop if x["variant"] == var and x["layout"] == lay and x["N"] == N]
            if not rows:
                continue
            s = f"{var} {lay} N={N} n_I={rows[0]['n_I']} n_dof={rows[0]['n_dof']} cond={st.median(x['cond_S_II'] for x in rows):.1e}: "
            for eps in ("0.001", "0.01", "0.1"):
                s += f"eps={eps}: " + " ".join(f"{pk}={st.median([x['perturbed'][f'{pk}_{eps}']['energy'] for x in rows]):.4f}" for pk in ("random", "high", "low")) + " | "
            s += f"exact={st.median(x['exact']['energy'] for x in rows):.1e} nnzLU={rows[0]['nnz_LU_monolithic']} nI3/3={rows[0]['n_I']**3/3:.2e}"
            if "pcg_iters" in rows[0]:
                s += f" pcg bj/2lvl={st.median(x['pcg_iters']['block_jacobi'] for x in rows):.0f}/{st.median(x['pcg_iters']['two_level'] for x in rows):.0f}"
            print(s)
print("\nrank scan (median energy error), truncated-S:")
for label in ("library", "new_shapes"):
    for lay, N in (("chain", 8), ("grid", 8), ("grid", 16)):
        rows = [x for x in r["rank_scan"] if x["shapes"] == label and x["layout"] == lay and x["N"] == N]
        for name in ("transfer", "pod"):
            med = {int(k): st.median(row[name][k] for row in rows) for k in rows[0][name]}
            need = next((k for k in sorted(med) if med[k] <= 0.01), None); need5 = next((k for k in sorted(med) if med[k] <= 0.05), None)
            print(f"{label} {lay} N={N} {name}: rank for 1% {need}, for 5% {need5}; " + " ".join(f"r{k}={med[k]:.3f}" for k in (1, 2, 4, 6, 8, 10, 12, 15)))
        tr = {k: [row["truncated_S"][k] for row in rows if row["truncated_S"][k] is not None] for k in rows[0]["truncated_S"]}
        print("   truncated S:", {k: (f"{st.median(v):.2e}" if v else None) for k, v in tr.items()})
print("\ntransfer sv:", [f"{v:.2e}" for v in r["transfer_singular_values"]])
print("pod sv:", [f"{v:.2e}" for v in r["pod_singular_values"]])
