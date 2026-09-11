#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J v0wk-probe
#SBATCH --time=00:30:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/v0wkprobe_%j.log
#SBATCH -e sbatch_logs/v0wkprobe_%j.log
#SBATCH --open-mode=append
# Routing-mass probe (2026-09-10): did the wake / latent-volume context tokens receive slice mass, and where do
# the slice anchors sit? One validation case per run, float32, code_wake (the snapshot that trained them).
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TASK}/code_wake:${TASK}/recipe/src"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_V0WKPROBE_${SLURM_JOB_ID}"' EXIT
BASE="+model.query_tokens=true +model.similarity_gauge=true +model.n_query_scalars=1 model.hidden=344 +forward_kwargs.query_normals=interior.point_data.sdf_normals +forward_kwargs.query_scalars=interior.point_data.sdf"
cd "${TASK}/recipe"
probe() {
  local RUN="$1" GPU="$2" EXTRA="$3"
  CUDA_VISIBLE_DEVICES="${GPU}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((33100 + GPU)) \
    "${VENV}/bin/python" "${TASK}/transfer/probe_routing_mass.py" "${TASK}/transfer/routing_mass/${RUN}.json" \
    model=mt2_volume ${BASE} ${EXTRA} dataset=drivaer_ml_volume_mt run_id="${RUN}" infer_split=val \
    sampling_resolution=10000 precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${TASK}/transfer/routing_mass/tmp_${RUN}" \
    force_coefficients.enabled=false > "${TASK}/transfer/routing_mass/${RUN}.log" 2>&1 || echo "=== ${RUN} FAILED rc=$?" >> "${TASK}/transfer/routing_mass/${RUN}.log"
}
mkdir -p "${TASK}/transfer/routing_mass"
( probe v0_isla_qtsdf_h344_seed42 0 "" ) &
( probe v0_isla_qtsdf_h344_wake_seed42 1 "+model.wake_tokens=true" ) &
( probe v0_isla_qtsdf_h344_lvt_seed42 2 "+model.latent_volume_tokens=true" ) &
( probe v0_isla_qtsdf_h344_wake_seed43 3 "+model.wake_tokens=true" ) &
wait
echo "V0WKPROBE-DONE $(ls ${TASK}/transfer/routing_mass/*.json 2>/dev/null | wc -l)/4"
