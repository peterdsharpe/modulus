"""Deploy campaign C on AGA (no submission): recipe copy with the two hooks, manifests,
dataset variants, lane table, launcher and eval clones.

Run on the cluster login node with the recipe venv python:
    $V/bin/python $T/transfer/build_campaign_c.py
Idempotent. Writes under $T/recipe_campc (copy of $T/recipe_support plus patches),
$T/transfer/manifests_c/, $T/transfer/campaign_c_lanes.tsv, $T/transfer/campaign_c_aga.sbatch,
$T/transfer/campaign_c_eval_aga.sbatch.
"""
import json, os, random, re, shutil, sys
from pathlib import Path

T = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0")
DRIVAER = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML")
SHIFT = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-ShiftSUV")
SRC_RECIPE = T / "recipe_support"
RECIPE = T / "recipe_campc"
MAN = T / "transfer" / "manifests_c"
MAN.mkdir(parents=True, exist_ok=True)
PATCH = T / "transfer" / "campc_patch"   # local copies of the patched repo files (scp'd beforehand)

# ---------------------------------------------------------------- recipe copy + hooks
if not RECIPE.exists():
    shutil.copytree(SRC_RECIPE, RECIPE, symlinks=True)
    print("created", RECIPE)
dt = RECIPE / "src" / "domain_transforms.py"
s = dt.read_text()
if "class SetConstantCellField" not in s:
    block = (PATCH / "domain_transforms.py").read_text()
    i = block.index("@register()\nclass SetConstantCellField")
    s += "\n\n" + block[i:]
    dt.write_text(s); print("patched domain_transforms.py")
ut = RECIPE / "src" / "utils.py"
s = ut.read_text()
if "def initialize_from_checkpoint" not in s:
    block = (PATCH / "utils.py").read_text()
    i = block.index("def initialize_from_checkpoint"); j = block.index('    return {"file": str(path), "epoch": epoch}', i) + len('    return {"file": str(path), "epoch": epoch}')
    fn = block[i:j]
    if "from pathlib import Path" not in s:
        s = s.replace("import torch\n", "import torch\nfrom pathlib import Path\n", 1)
    if "from typing import" in s and "Any" not in s.split("from typing import")[1].split("\n")[0]:
        s = s.replace("from typing import", "from typing import Any,", 1)
    s += "\n\n" + fn + "\n"
    ut.write_text(s); print("patched utils.py")
tr = RECIPE / "src" / "train.py"
s = tr.read_text()
if "init_from" not in s:
    anchor = "    model.to(device)\n\n    if dist_manager.world_size > 1:\n"
    assert anchor in s, "train.py anchor not found; inspect the cluster recipe's model setup"
    hook = '''    model.to(device)

    ### Fine-tuning (transfer program, campaign C): initialize weights from another
    ### run's checkpoint directory; optimizer, scheduler and epoch start fresh. Skipped
    ### when this run already has its own checkpoints (chained links resume normally).
    init_from = cfg.training.get("init_from", None)
    if init_from:
        own_ckpt_dir = os.path.join(checkpoint_dir, cfg.run_id, "checkpoints")
        if os.path.isdir(own_ckpt_dir) and any(f.endswith(".mdlus") for f in os.listdir(own_ckpt_dir)):
            logger.info(f"init_from={init_from!r} ignored: run has its own checkpoints in {own_ckpt_dir}")
        else:
            init_report = initialize_from_checkpoint(model, init_from, device=device)
            logger.info(f"Initialized weights from {init_report['file']} (epoch {init_report['epoch']})")

    if dist_manager.world_size > 1:
'''
    s = s.replace(anchor, hook, 1)
    assert "from utils import (" in s
    s = s.replace("from utils import (", "from utils import (\n    initialize_from_checkpoint,", 1)
    assert "import os" in s
    tr.write_text(s); print("patched train.py")

# ---------------------------------------------------------------- manifests
rng = random.Random(2026)
dm = json.load(open(DRIVAER / "manifest.json"))
em = json.load(open(SHIFT / "estate" / "manifest.json"))
fm = json.load(open(SHIFT / "fastback" / "manifest.json"))
d_train = sorted(dm["train"]); e_train = sorted(em["train"]); f_train = sorted(fm["train"])
assert len(d_train) == 435 and len(em["val"]) == 99 and len(fm["val"]) == 99
d218 = sorted(rng.sample(d_train, 218)); e217 = sorted(rng.sample(e_train, 217))
f30 = rng.sample(f_train, 30); f20, f10 = sorted(f30[:20]), sorted(f30[20:])
json.dump({"train": d218, "val": dm["val"]}, open(MAN / "drivaer_n218_manifest.json", "w"), indent=1)
json.dump({"train": e217, "val": em["val"]}, open(MAN / "estate_n217_manifest.json", "w"), indent=1)
json.dump({"train": f20, "val": f10, "fewshot_val10": f10, "test99": fm["val"]}, open(MAN / "fastback_fewshot20_manifest.json", "w"), indent=1)
print("manifests: drivaer 218, estate 217, fastback 20+10 (test = fastback val 99)")

# ---------------------------------------------------------------- dataset yaml variants
DS = RECIPE / "datasets"
def variant(base, name, manifest=None, cond=None):
    y = (DS / f"{base}.yaml").read_text()
    if manifest is not None:
        # replace an existing manifest: line or add one after train_datadir
        if re.search(r"^manifest:.*$", y, flags=re.M):
            y = re.sub(r"^manifest:.*$", f"manifest: {manifest}", y, flags=re.M)
        else:
            y = re.sub(r"^(train_datadir:.*)$", rf"\1\nmanifest: {manifest}", y, count=1, flags=re.M)
    if cond is not None:
        # insert the constant condition field right before ComputeSurfaceNormals (cell_data, before MeshToDomainMesh)
        marker = "    - _target_: ${dp:ComputeSurfaceNormals}"
        assert marker in y, f"{base}: ComputeSurfaceNormals not found"
        y = y.replace(marker, f"    - _target_: ${{dp:SetConstantCellField}}\n      field_name: cond\n      value: {cond}\n{marker}", 1)
    (DS / f"{name}.yaml").write_text(f"# GENERATED by build_campaign_c.py from {base}.yaml (transfer program, campaign C)\n" + y)
variant("drivaer_ml_surface", "campc_drivaer_cond0", cond=0.0)
variant("shift_suv_surface", "campc_estate_cond1", cond=1.0)
variant("shift_suv_surface_fastback", "campc_fastback_cond1", cond=1.0)
variant("drivaer_ml_surface", "campc_drivaer_n218", manifest=str(MAN / "drivaer_n218_manifest.json"))
variant("shift_suv_surface", "campc_estate_n217", manifest=str(MAN / "estate_n217_manifest.json"))
variant("shift_suv_surface_fastback", "campc_fastback_fewshot20", manifest=str(MAN / "fastback_fewshot20_manifest.json"))
print("dataset variants written")

# ---------------------------------------------------------------- lane table
GTU = "forward_kwargs.global_embedding=global_data.U_inf_dir"
GTC = GTU + " forward_kwargs.local_embedding=[interior.points,boundaries.vehicle.cell_data.normals,boundaries.vehicle.cell_data.cond] model.functional_dim=7"
ISC = "+model.n_boundary_scalars=1 +forward_kwargs.boundary_scalars=boundaries.vehicle.cell_data.cond"
# NOTE: no ISLA output-head override here. The udrv lane table's "model.out_scalars=3
# model.out_vectors=2" is HiLift's target set; DrivAerML/SHIFT-SUV targets are pressure +
# wall shear (4 channels), which mt2_surface.yaml's defaults already produce. The first
# acceptance submission carried the override and failed at step 0 with a 9-vs-4 channel error.
rows = []
def lane(arm, model, extra, dataset, lr, seed, run):
    rows.append((len(rows), arm, model, extra, dataset, lr, seed, 10000, run))
for s_ in (42, 43):
    lane("T1", "mt2_surface", f"extra_datasets=[shift_suv_surface]", "drivaer_ml_surface", "1.0e-3", s_, f"campC_cond0_isla_seed{s_}")
    lane("T1", "geotransolver_surface", f"{GTU} extra_datasets=[shift_suv_surface]", "drivaer_ml_surface", "1.0e-3", s_, f"campC_cond0_gt_seed{s_}")
    lane("T1", "mt2_surface", f"{ISC} extra_datasets=[campc_estate_cond1]", "campc_drivaer_cond0", "1.0e-3", s_, f"campC_cond1_isla_seed{s_}")
    lane("T1", "geotransolver_surface", f"{GTC} extra_datasets=[campc_estate_cond1]", "campc_drivaer_cond0", "1.0e-3", s_, f"campC_cond1_gt_seed{s_}")
for s_ in (42, 43):
    lane("T2", "mt2_surface", f"extra_datasets=[campc_estate_n217]", "campc_drivaer_n218", "1.0e-3", s_, f"campC_mix435_isla_seed{s_}")
    lane("T2", "geotransolver_surface", f"{GTU} extra_datasets=[campc_estate_n217]", "campc_drivaer_n218", "1.0e-3", s_, f"campC_mix435_gt_seed{s_}")
for s_ in (42, 43):
    lane("T3", "mt2_surface", f"training.init_from={T}/runs/mt2_v3c_seed42/checkpoints training.num_epochs=200 training.optimizer.lr=1.0e-4", "campc_fastback_fewshot20", "1.0e-4", s_, f"campC_ft20_isla_seed{s_}")
    lane("T3", "mt2_surface", f"training.num_epochs=200", "campc_fastback_fewshot20", "1.0e-3", s_, f"campC_scratch20_isla_seed{s_}")
# GT few-shot lanes (16-19) wait for uw_gt_unit_lr1e3_seed42 to finish; listed so the launcher can run them later
for s_ in (42, 43):
    lane("T3gt", "geotransolver_surface", f"{GTU} training.init_from={T}/runs/uw_gt_unit_lr1e3_seed42/checkpoints training.num_epochs=200 training.optimizer.lr=1.0e-4", "campc_fastback_fewshot20", "1.0e-4", s_, f"campC_ft20_gt_seed{s_}")
    lane("T3gt", "geotransolver_surface", f"{GTU} training.num_epochs=200", "campc_fastback_fewshot20", "1.0e-3", s_, f"campC_scratch20_gt_seed{s_}")
with open(T / "transfer" / "campaign_c_lanes.tsv", "w") as f:
    f.write("idx\tarm\tmodel\textra\tdataset\tlr\tseed\ttokens\trun_id\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(f"lane table: {len(rows)} lanes (0-15 launch now; 16-19 after uw_gt_unit_lr1e3_seed42)")

# ---------------------------------------------------------------- launcher + eval clones
src = (T / "highlift" / "hl_udrv_aga.sbatch").read_text()
src = src.replace("udrv_lanes.tsv", "../transfer/campaign_c_lanes.tsv").replace('RECIPE="${TASK}/recipe"', 'RECIPE="${TASK}/recipe_campc"')
src = re.sub(r'CODE="\$\{TASK\}/[^"]*"', 'CODE="${TASK}/code_support"', src)
src = src.replace("hl-udrv", "camp-c").replace("hludrv_", "campc_").replace("hl_udrv_aga.sbatch", "../transfer/campaign_c_aga.sbatch")
# The udrv launcher passes training.num_epochs="${EPOCHS}" AFTER ${EXTRA}; parse a lane-level
# epoch count out of EXTRA (T3 uses 200) so the fixed value does not override it.
old_epochs = 'EPOCHS=500\n'
assert old_epochs in src
src = src.replace(old_epochs, 'EPOCHS=500\ncase "${EXTRA}" in *training.num_epochs=*) EPOCHS=$(echo "${EXTRA}" | grep -oE "training.num_epochs=[0-9]+" | cut -d= -f2); EXTRA=$(echo "${EXTRA}" | sed -E "s/training.num_epochs=[0-9]+//") ;; esac\n', 1)
assert "${EXTRA}" in src
(T / "transfer" / "campaign_c_aga.sbatch").write_text(src)

# Eval launcher (own body): per lane, evaluate on the fastback validation set (99 cases) and,
# for arms trained on DrivAerML, on the DrivAerML validation set (48 cars); the dataset variant
# carries the condition field for the conditioned arms. training.* tokens are stripped from EXTRA.
ev = f'''#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J camp-c-eval
#SBATCH --time=01:30:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-0%1
#SBATCH -o sbatch_logs/campceval_%A_%a.log
#SBATCH -e sbatch_logs/campceval_%A_%a.log
#SBATCH --open-mode=append
# Campaign C evaluation: submit with --array=<lane idx>. Idempotent per (run, target).
set -euo pipefail
ulimit -s 8192
TASK={T}
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RECIPE="${{TASK}}/recipe_campc"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=600
export PYTHONPATH="${{TASK}}/code_support:${{RECIPE}}/src"
trap 'rc=$?; printf "rc=%s\\n" "$rc" > "${{TASK}}/STATUS_CAMPCEVAL_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"' EXIT
LANE=$(awk -F"\\t" -v i="${{SLURM_ARRAY_TASK_ID}}" 'NR>1 && $1==i {{print; exit}}' "${{TASK}}/transfer/campaign_c_lanes.tsv")
[ -z "${{LANE}}" ] && {{ echo "lane ${{SLURM_ARRAY_TASK_ID}} not in campaign_c_lanes.tsv"; exit 1; }}
ARM=$(echo "${{LANE}}" | cut -f2); MODEL=$(echo "${{LANE}}" | cut -f3); EXTRA=$(echo "${{LANE}}" | cut -f4)
DATASET=$(echo "${{LANE}}" | cut -f5); TOKENS=$(echo "${{LANE}}" | cut -f8); RUN=$(echo "${{LANE}}" | cut -f9)
EXTRA=$(echo "${{EXTRA}}" | sed -E "s/training\\.[a-z_.]+=[^ ]+//g; s/extra_datasets=\\[[^]]*\\]//g")
grep -q "Training completed" "${{TASK}}/runs/${{RUN}}/train.log" 2>/dev/null || {{ echo "${{RUN}} not trained; no-op"; exit 0; }}
case "${{RUN}}" in
  *cond1*) FAST=campc_fastback_cond1; DRIV=campc_drivaer_cond0 ;;
  *) FAST=shift_suv_surface_fastback; DRIV=drivaer_ml_surface ;;
esac
unit() {{
  local DS="$1" OUTDIR="$2" GPU="$3"
  [ -f "${{OUTDIR}}/.done" ] && return 0
  mkdir -p "${{OUTDIR}}"
  timeout 5000 env CUDA_VISIBLE_DEVICES="${{GPU}}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((32600 + GPU)) \\
    "${{VENV}}/bin/python" "${{RECIPE}}/src/infer.py" model="${{MODEL}}" ${{EXTRA}} dataset="${{DS}}" run_id="${{RUN}}" infer_split=val \\
    sampling_resolution="${{TOKENS}}" checkpoint_dir="${{TASK}}/runs" output_dir="${{OUTDIR}}" >> "${{OUTDIR}}.log" 2>&1 \\
    && touch "${{OUTDIR}}/.done" || echo "=== ${{RUN}} ${{DS}} FAILED rc=$?" >> "${{OUTDIR}}.log"
}}
mkdir -p "${{TASK}}/campc_evals/${{RUN}}"; cd "${{RECIPE}}"
( unit "${{FAST}}" "${{TASK}}/campc_evals/${{RUN}}/fastback" 0 ) &
P0=$!
if [ "${{ARM}}" = "T1" ] || [ "${{ARM}}" = "T2" ]; then ( unit "${{DRIV}}" "${{TASK}}/campc_evals/${{RUN}}/drivaer" 1 ) & P1=$!; wait "$P1" || true; fi
wait "$P0" || true
echo "CAMPCEVAL-DONE ${{RUN}}"
'''
(T / "transfer" / "campaign_c_eval_aga.sbatch").write_text(ev)
print("launcher and eval written; inspect with: grep -nE 'RECIPE=|CODE=|lanes.tsv|EPOCHS' campaign_c_aga.sbatch")
