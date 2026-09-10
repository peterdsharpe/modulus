"""Clone the program's evaluation launchers into float32 variants (FP32-REEVAL, notebook
#sec-nb-fp32-reeval-prereg).

Each clone differs from its source in exactly two functional edits, so override parity with the
bf16 evaluation holds by construction:
  * the output root: hl_evals -> hl_evals_fp32, iw_evals -> iw_evals_fp32, v0_evals -> v0_evals_fp32
    (this also moves the per-run .done idempotency markers and logs into the fp32 tree);
  * `precision=float32` appended to the infer.py invocation, anchored on `output_dir="${OUTDIR}"`.
Job names, sbatch log names and STATUS markers get an fp32_ prefix so nothing collides with the
bf16 launchers. Everything else (code snapshot on PYTHONPATH, model flags, forward_kwargs, dataset,
sampling_resolution, the "not trained" guard, array-index mapping) is untouched.

Usage (on the cluster login node): python3 make_fp32_launchers_2026-09-09.py
Writes $T/fp32_eval/<basename> for every source in SOURCES and prints a manifest.
"""
import pathlib, re, sys

T = pathlib.Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0")
OUT = T / "fp32_eval"
OUT.mkdir(exist_ok=True)
SOURCES = [
    # HiLift surface families
    "highlift/hl_ladder_eval_aga.sbatch", "highlift/hl_a35_eval_aga.sbatch", "highlift/hl_t1_eval_aga.sbatch",
    "highlift/hl_udrv_eval_aga.sbatch", "highlift/hl_wave2_eval_aga.sbatch", "highlift/hl_ladder_ctrl_eval_aga.sbatch",
    "highlift/hl_eval_aga.sbatch", "highlift/hl_inv_eval_aga.sbatch", "highlift/hl_b1_eval_aga.sbatch",
    # DrivAerML surface
    "iw_eval_aga.sbatch", "uw_eval_aga.sbatch", "nowt_eval_aga.sbatch", "mom2t_eval_aga.sbatch",
    # interior
    "v0/v0_full_eval_aga.sbatch", "v0/v0_qt_eval_aga.sbatch", "v0/v0_qt2_eval_aga.sbatch", "v0/v0_qt3_eval_aga.sbatch",
    "v0/v0_qt4_eval_aga.sbatch", "v0/v0_surf10k_eval_aga.sbatch", "v0/v0_udrvint_eval_aga.sbatch", "v0/v0_ladder_eval_aga.sbatch",
]
ROOTS = ("hl_evals", "iw_evals", "v0_evals")
manifest = []
for rel in SOURCES:
    src = T / rel
    if not src.exists():
        print(f"SKIP missing {rel}"); continue
    s = src.read_text()
    n_out = s.count('output_dir="${OUTDIR}"')
    if n_out != 1:
        print(f"SKIP {rel}: expected one output_dir anchor, found {n_out}"); continue
    if "precision=" in s:
        print(f"SKIP {rel}: already sets precision"); continue
    for r in ROOTS:
        s = re.sub(rf"\b{r}/", f"{r}_fp32/", s)
    s = s.replace('output_dir="${OUTDIR}"', 'output_dir="${OUTDIR}" precision=float32', 1)
    # job name, sbatch logs, STATUS/RUN markers
    s = re.sub(r"(#SBATCH -J )(\S+)", r"\1fp32-\2", s, count=1)
    s = re.sub(r"(#SBATCH -[oe] sbatch_logs/)", r"\1fp32_", s)
    s = re.sub(r'(STATUS_)([A-Za-z0-9]+)', r'\1FP32\2', s)
    s = re.sub(r'(-TASK-DONE)', r'-FP32\1', s)
    dst = OUT / src.name
    dst.write_text(s)
    manifest.append((rel, str(dst), s.count("_fp32/"), "precision=float32" in s))
for m in manifest:
    print("WROTE", m)
