#!/bin/bash
# Wait for ablation training to finish, then run disentanglement metrics.
set -euo pipefail
cd /home/chyanin/robotpaper

LOG="/home/chyanin/robotpaper/logs/ablation_10seed.log"

echo "$(date) Waiting for ablation training to complete..."
while true; do
    DONE_COUNT=$(grep -c 'DONE:' "$LOG" 2>/dev/null || echo "0")
    if [[ "$DONE_COUNT" -ge 20 ]]; then
        echo "$(date) Ablation training complete ($DONE_COUNT/20 done)"
        break
    fi
    echo "$(date) Ablation: $DONE_COUNT/20 done, waiting 60s..."
    sleep 60
done

# Wait a bit more for eval phases to finish (they run after training)
echo "$(date) Waiting 30min for ablation eval phases..."
sleep 1800

echo "$(date) Starting disentanglement metrics..."
chmod +x scripts/run_disentanglement.sh
PYTHON_CMD=/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3 bash scripts/run_disentanglement.sh 200
echo "$(date) Disentanglement metrics COMPLETE"
