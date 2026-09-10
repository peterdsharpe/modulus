#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J camp-e32sf
#SBATCH --time=01:30:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-6%7
#SBATCH -o sbatch_logs/campe32sf_%A_%a.log
#SBATCH -e sbatch_logs/campe32sf_%A_%a.log
#SBATCH --open-mode=append
# Campaign E, SAMPLE-FRAME convention (2026-09-10): the program's canonical sample-frame probes
# recipe/datasets/drivaer_probe_{unif2,biased3}_sf.yaml (CenterMesh AFTER the draw, so absolute-coordinate
# models see the frame a biased mesher would deliver). Same float32 instrument, code snapshot and
# overrides as campaign_e_density_fp32_aga.sbatch. The four runs the generalization session already
# evaluated under this convention (uw_gt_unit seeds 42/43, uw_transolver seeds 42/43 -> $T/mw_evals_probe_postcenter)
# are read from there by the reducer and are not repeated here.
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=600
export PYTHONPATH="${TASK}/code_eval:${TASK}/recipe/src"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_CAMPE32SF_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' EXIT

RUNS=(uw_gt_unit_lr1e3_seed44 iw_mt2_lr1e3_seed42 iw_mt2_lr1e3_seed43 iw_mt2_gauge_seed42 iw_mt2_gauge_seed43 now_mt2_lr1e3_seed42 now_mt2_lr1e3_seed43)
RUN="${RUNS[${SLURM_ARRAY_TASK_ID}]}"
case "$RUN" in
  uw_gt_*)         MODEL=geotransolver_surface; EXTRA="forward_kwargs.global_embedding=global_data.U_inf_dir" ;;
  uw_transolver_*) MODEL=transolver_surface;    EXTRA="forward_kwargs.fx.source=global_data.U_inf_dir" ;;
  iw_mt2_lr1e3_*)  MODEL=mt2_surface;           EXTRA="" ;;
  iw_mt2_gauge_*)  MODEL=mt2_surface;           EXTRA="+model.similarity_gauge=true" ;;
  now_mt2_*)       MODEL=mt2_surface;           EXTRA="+model.use_measure_weights=false" ;;
  *) echo "unknown run ${RUN}" >&2; exit 1 ;;
esac
grep -q "Training completed" "${TASK}/runs/${RUN}/train.log" || { echo "${RUN} not trained"; exit 1; }

unit() {
  local DATASET="$1" OUTDIR="$2" GPU="$3"
  [ -f "${OUTDIR}/.done" ] && return 0
  mkdir -p "${OUTDIR}"
  timeout 4800 env CUDA_VISIBLE_DEVICES="${GPU}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$((21000 + SLURM_JOB_ID % 30000 + GPU)) \
    "${VENV}/bin/python" "${TASK}/recipe/src/infer.py" \
    model="${MODEL}" ${EXTRA} dataset="${DATASET}" run_id="${RUN}" infer_split=val \
    sampling_resolution=10000 precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${OUTDIR}" \
    >> "${OUTDIR}.log" 2>&1 && touch "${OUTDIR}/.done" || echo "=== ${RUN} ${DATASET} FAILED rc=$?" >> "${OUTDIR}.log"
}
OUT="${TASK}/transfer/campaign_e_fp32_sf/${RUN}"
mkdir -p "${OUT}"; cd "${TASK}/recipe"
( unit drivaer_probe_unif2_sf   "${OUT}/unif"   0 ) & P0=$!
( unit drivaer_probe_biased3_sf "${OUT}/biased" 1 ) & P1=$!
rc_all=0; for p in $P0 $P1; do wait "$p" || rc_all=1; done
n_done=$(find "${OUT}" -name .done | wc -l)
echo "CAMPE-SF-DONE ${RUN} ${n_done}/2"
[ "${n_done}" = "2" ] && touch "${TASK}/transfer/campaign_e_fp32_sf/EVAL_DONE_${RUN}"
exit "${rc_all}"
