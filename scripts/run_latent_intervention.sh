#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# run_latent_intervention.sh — Run latent clamping for multiple seeds
# Isaac Lab can only create one SimulationApp per process,
# so we launch a fresh Python process per seed.
# ══════════════════════════════════════════════════════════════
set -euo pipefail

ISAAC_PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
export PYTHON_CMD="$ISAAC_PYTHON"

SEEDS=(42 43 44)
EPISODES=30
OUT_DIR="results/latent_intervention"

mkdir -p "$OUT_DIR"
echo "Latent intervention — ${#SEEDS[@]} seeds × $EPISODES episodes each"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "═══ SEED $seed ═══"
    json_file="${OUT_DIR}/intervention_seed${seed}.json"
    if [[ -f "$json_file" ]]; then
        echo "  [SKIP] Already exists: $json_file"
        continue
    fi
    $ISAAC_PYTHON scripts/latent_intervention.py \
        --seeds "$seed" \
        --num_episodes "$EPISODES" \
        --output_dir "$OUT_DIR" 2>&1 | tee "${OUT_DIR}/log_seed${seed}.txt"
    echo "  Done: seed $seed"
done

echo ""
echo "═══ All seeds complete ═══"

# Merge per-seed JSONs into one
$ISAAC_PYTHON -c "
import json, glob
from pathlib import Path

out_dir = Path('$OUT_DIR')
all_results = {}
for f in sorted(out_dir.glob('intervention_seed*.json')):
    seed = f.stem.replace('intervention_seed', '')
    with open(f) as fh:
        all_results[f'seed_{seed}'] = json.load(fh)

combined = {
    'seeds': [int(s.replace('seed_', '')) for s in all_results],
    'factors': ['friction', 'mass', 'motor', 'contact', 'delay'],
    'results': all_results,
}
with open(out_dir / 'intervention_results.json', 'w') as fh:
    json.dump(combined, fh, indent=2)
print(f'Merged {len(all_results)} seed files → {out_dir / \"intervention_results.json\"}')
"
