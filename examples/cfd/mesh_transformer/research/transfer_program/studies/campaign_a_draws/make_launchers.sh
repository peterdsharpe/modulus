#!/bin/bash
# Campaign A launchers: clones of highlift/hl_udrv_aga.sbatch and hl_udrv_eval_aga.sbatch that read
# $T/transfer/campaign_a_lanes.tsv, use the recipe copy $T/recipe_support and the code snapshot
# $T/code_support, and carry job names camp-a-<lane>. Run on the AGA login node.
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
cd $T
sed -e 's/#SBATCH -J hl-udrv$/#SBATCH -J camp-a/' \
    -e 's/--array=0-17%18/--array=0-71%72/' \
    -e 's|sbatch_logs/hludrv_%A_%a.log|sbatch_logs/campa_%A_%a.log|g' \
    -e 's|CODE="${TASK}/code"|CODE="${TASK}/code_support"   # campaign A (transfer program): defaults bitwise equal to the reference ISLA|' \
    -e 's|RECIPE="${TASK}/recipe"|RECIPE="${TASK}/recipe_support"   # copy of the frozen recipe; carries the draw dataset variants|' \
    -e 's/TAG="udrv_/TAG="campa_/' \
    -e 's|"${TASK}/highlift/udrv_lanes.tsv"|"${TASK}/transfer/campaign_a_lanes.tsv"|' \
    -e 's/not in udrv_lanes.tsv/not in campaign_a_lanes.tsv/' \
    -e 's|sbatch_logs/hludrv_${TAG}.log|sbatch_logs/campa_${TAG}.log|' \
    -e 's|sbatch -J "hl-udrv-${SLURM_ARRAY_TASK_ID}" --array="${SLURM_ARRAY_TASK_ID}" --dependency=singleton "${TASK}/highlift/hl_udrv_aga.sbatch" >> "${TASK}/sbatch_logs/udrv_resubmit.log"|sbatch -J "camp-a-${SLURM_ARRAY_TASK_ID}" --array="${SLURM_ARRAY_TASK_ID}" --dependency=singleton "${TASK}/transfer/campaign_a_aga.sbatch" >> "${TASK}/sbatch_logs/campa_resubmit.log"|' \
    -e 's/run highlift\/build_ladder_manifests.py first/run build_campaign_a.py first/' \
    highlift/hl_udrv_aga.sbatch > transfer/campaign_a_aga.sbatch
sed -e 's/#SBATCH -J hl-udrv-eval$/#SBATCH -J camp-a-eval/' \
    -e 's/--array=0-17%18/--array=0-71%72/' \
    -e 's|sbatch_logs/hludrveval_%A_%a.log|sbatch_logs/campaeval_%A_%a.log|g' \
    -e 's|export PYTHONPATH="${TASK}/code:${TASK}/recipe/src"|export PYTHONPATH="${TASK}/code_support:${TASK}/recipe_support/src"|' \
    -e 's/STATUS_HLUDRVEVAL_/STATUS_CAMPAEVAL_/' \
    -e 's|"${TASK}/highlift/udrv_lanes.tsv"|"${TASK}/transfer/campaign_a_lanes.tsv"|' \
    -e 's/not in udrv_lanes.tsv/not in campaign_a_lanes.tsv/' \
    -e 's|"${TASK}/recipe/datasets/${DATASET}.yaml"|"${TASK}/recipe_support/datasets/${DATASET}.yaml"|' \
    -e 's|cd "${TASK}/recipe"|cd "${TASK}/recipe_support"|' \
    -e 's|"${TASK}/recipe/src/infer.py"|"${TASK}/recipe_support/src/infer.py"|' \
    -e 's/HLUDRVEVAL-DONE/CAMPAEVAL-DONE/' \
    highlift/hl_udrv_eval_aga.sbatch > transfer/campaign_a_eval_aga.sbatch
bash -n transfer/campaign_a_aga.sbatch && bash -n transfer/campaign_a_eval_aga.sbatch && echo syntax-ok
grep -nE '^#SBATCH -J|^CODE=|^RECIPE=|campaign_a_lanes|camp-a-\$|campa_resubmit' transfer/campaign_a_aga.sbatch | head -8
grep -nE '^#SBATCH -J|PYTHONPATH|campaign_a_lanes|recipe_support' transfer/campaign_a_eval_aga.sbatch | head -6
grep -c support_tokens code_support/physicsnemo/experimental/nn/isla/model.py
