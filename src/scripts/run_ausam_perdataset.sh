#!/bin/bash
# Per-dataset AUSAM (single GT-box, supervised control), one model per source dataset.
# Each training uses BOTH GPUs (DDP), so we run them sequentially, waiting for the GPUs to free
# between runs. LiTS is launched separately and is running now; this queues the rest behind it.
# Skips a dataset whose result JSON already exists (resume-safe).
cd /home/ud3d4/Desktop/SWOG
PY=~/.conda/envs/llmft/bin/python
for DS in kits msd flare_task2 flare23; do
  RES=/home/ud3d4/Desktop/SWOG/results/organ_generic_ausam_${DS}.json
  if [ -f "$RES" ]; then echo "[$DS] result exists — skip"; continue; fi
  # wait until GPU0 is essentially free (previous run finished)
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -gt 2000 ]; do sleep 60; done
  echo "=== [$DS] launching AUSAM $(date '+%H:%M') ==="
  $PY src/scripts/train_organ_generic.py --ausam --dataset "$DS" \
      > /scratch/ud3d4/acm_data/ausam_${DS}.log 2>&1
  echo "=== [$DS] finished $(date '+%H:%M') ==="
  sleep 30
done
echo "ALL per-dataset AUSAM done"
