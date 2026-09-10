#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla-snapladder
#SBATCH --time=01:30:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/sbatch_logs/isla_snapladder_%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/sbatch_logs/isla_snapladder_%j.log
# SNAPSHOT-LADDER (notebook #sec-nb-snapshot-ladder-prereg): the fixed checkpoint iw_mt2_lr1e3_seed42 evaluated with the
# reference kernel under each code snapshot with the same recipe copy; the shadow checkpoint dir carries both class file names.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'echo "rc=$?" > ${TASK}/STATUS_SNAPLADDER' EXIT
PY="${VENV}/bin/python"
run_one() {  # $1 snapshot dir, $2 recipe dir, $3 label, $4 gpu
  local OUTDIR=${TASK}/iw_evals/snapladder_$3
  rm -rf "${OUTDIR}"; mkdir -p "${OUTDIR}"
  ( cd "$2" && PYTHONPATH="${TASK}/$1:$2/src" CUDA_VISIBLE_DEVICES=$4 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((32800 + $4)) \
    "${PY}" "$2/src/infer.py" model=mt2_surface dataset=drivaer_ml_surface run_id=iw_mt2_lr1e3_seed42_shadow \
    infer_split=val sampling_resolution=10000 checkpoint_dir="${TASK}/runs" output_dir="${OUTDIR}" force_coefficients.enabled=false \
    > "${OUTDIR}.log" 2>&1 ) && echo "DONE $3 loaded=$(grep -c 'Loaded model state dictionary' "${OUTDIR}.log") $(grep -A3 'pressure_l2' "${OUTDIR}.log" | grep '|' | head -1)" || echo "FAILED $3 (see ${OUTDIR}.log)"
}
run_one code ${TASK}/recipe code 0 &
run_one code_isla2 ${TASK}/recipe code_isla2 1 &
run_one code_isla3 ${TASK}/recipe code_isla3 2 &
run_one code_isla4 ${TASK}/recipe code_isla4 3 &
wait
run_one code_isla5 ${TASK}/recipe code_isla5 0 &
run_one code_perf ${TASK}/recipe code_perf 1 &
if [ "$(diff -rq ${TASK}/recipe ${TASK}/recipe_support 2>/dev/null | grep -c ' differ')" != "0" ]; then
  run_one code ${TASK}/recipe_support code_recipe_support 2 &
fi
wait
echo SNAPLADDER_DONE
