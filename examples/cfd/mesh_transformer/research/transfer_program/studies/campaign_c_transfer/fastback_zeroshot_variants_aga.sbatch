#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J fbzs-variants
#SBATCH --time=00:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1%2
#SBATCH -o sbatch_logs/fbzs_variants_%A_%a.log
#SBATCH -e sbatch_logs/fbzs_variants_%A_%a.log
#SBATCH --open-mode=append
# Zero-shot fastback (99 SHIFT-SUV validation cases) in float32 for the DrivAerML-trained ISLA variants that the
# Generalization Story fork's 4b hypotheses need (2026-09-10): similarity gauge (H1), 2560 query-independent anchors (H2),
# measure weights off (H4). Legacy MeshTransformer2 checkpoints -> evaluated under the snapshot that wrote them (`code`).
# Task 0: four units on four GPUs; task 1: two units. Outputs $T/transfer/fastback_zeroshot_fp32/<run>/fastback/<run>/.
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=600
export PYTHONPATH="${TASK}/code:${TASK}/recipe/src"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_FBZS_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' EXIT
unit() {  # RUN EXTRA GPU
  local RUN="$1" EXTRA="$2" GPU="$3"
  local OUTDIR="${TASK}/transfer/fastback_zeroshot_fp32/${RUN}/fastback"
  [ -f "${OUTDIR}/.done" ] && return 0
  mkdir -p "${OUTDIR}"
  timeout 3000 env CUDA_VISIBLE_DEVICES="${GPU}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$((26000 + SLURM_JOB_ID % 30000 + GPU)) \
    "${VENV}/bin/python" "${TASK}/recipe/src/infer.py" model=mt2_surface ${EXTRA} dataset=shift_suv_surface_fastback run_id="${RUN}" infer_split=val \
    sampling_resolution=10000 precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${OUTDIR}" \
    >> "${OUTDIR}.log" 2>&1 && touch "${OUTDIR}/.done" || echo "=== ${RUN} FAILED rc=$?" >> "${OUTDIR}.log"
}
cd "${TASK}/recipe"
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  ( unit iw_mt2_gauge_seed42 "+model.similarity_gauge=true" 0 ) & P0=$!
  ( unit iw_mt2_gauge_seed43 "+model.similarity_gauge=true" 1 ) & P1=$!
  ( unit now_mt2_lr1e3_seed42 "+model.use_measure_weights=false" 2 ) & P2=$!
  ( unit now_mt2_lr1e3_seed43 "+model.use_measure_weights=false" 3 ) & P3=$!
  for p in $P0 $P1 $P2 $P3; do wait "$p" || true; done
else
  ( unit iw_mt2_v5a4_seed42 "+model.query_independent=true +model.n_anchors=2560" 0 ) & P0=$!
  ( unit iw_mt2_v5a4_seed43 "+model.query_independent=true +model.n_anchors=2560" 1 ) & P1=$!
  for p in $P0 $P1; do wait "$p" || true; done
fi
echo "FBZS-DONE task ${SLURM_ARRAY_TASK_ID}: $(ls ${TASK}/transfer/fastback_zeroshot_fp32/*/fastback/.done 2>/dev/null | wc -l)/6"
