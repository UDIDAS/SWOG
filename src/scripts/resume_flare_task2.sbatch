#!/bin/bash
#SBATCH --job-name=flare_t2_resume
#SBATCH --partition=research
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ud3d4/acm_data/FLARE_Task2/resume_%j.log
#
# Requeuing resume for FLARE-Task2 per-patient SAM3 training.
# Survives allocation expiry / node reboot: on (re)start it restores data + checkpoints
# from Drive if scratch was wiped, forces any INCOMPLETE organ to retrain, then runs the
# resumable trainer (finished organs skip themselves) and exports NIfTI deliverables.

set -e
export PATH=$HOME/bin:$PATH
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
PY=~/.conda/envs/llmft/bin/python
D=/scratch/ud3d4/acm_data/FLARE_Task2
CK=$D/sam3
REMOTE_CK=drive_UD:data/FLARE_Task2_ckpts
SRC=/home/ud3d4/Desktop/SWOG/src/scripts
mkdir -p "$CK"

echo "[$(date)] resume start on $(hostname), job $SLURM_JOB_ID"

# 1) restore GT data if scratch was wiped
if [ "$(ls $D/train_gt_label/imagesTr/*.nii.gz 2>/dev/null | wc -l)" -lt 50 ]; then
  echo "[resume] data missing -> re-downloading from HF"
  $PY "$SRC/../scripts/download_flare_task2_gt.py" || $PY /home/ud3d4/Desktop/SWOG/handoff/download_flare_task2_gt.py
fi

# 2) restore checkpoints + completion markers from Drive if scratch was wiped
rclone copy "$REMOTE_CK" "$CK/" 2>/dev/null || true

# 3) force only INCOMPLETE organs to retrain: keep every checkpoint that has a matching
#    .done marker (a fully finished organ). Remove a checkpoint ONLY when its .done marker
#    is present nowhere (local or Drive) AND we can confirm it was interrupted -- to stay
#    safe against marker loss, we DELETE NOTHING here and rely on skip-if-checkpoint. An
#    interrupted organ therefore resumes at its best-so-far (acceptable; all plateau early).
#    To force a clean re-run of a specific organ, delete its .pth before submitting.

# 4) run the resumable trainer (finished organs skip; writes .done markers going forward)
cd "$SRC"
$PY run_flare_task2_sam3.py

# 5) export per-patient NIfTI deliverables
$PY run_flare_task2_sam3.py --deliver

# 6) final checkpoint + delivery sync to Drive
rclone copy "$CK" "$REMOTE_CK/" 2>/dev/null || true
echo "[$(date)] resume complete"
