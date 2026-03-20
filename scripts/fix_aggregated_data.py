#!/usr/bin/env python3
"""Rebuild results/aggregated/main_comparison.json from 5-seed eval_recomputed data."""

import json
import glob
import os
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working dir: {os.getcwd()}")

tasks = ["flat", "push", "randomized", "terrain"]
models = ["mlp", "lstm", "transformer", "dynamite"]

out = {}
for task in tasks:
    out[task] = {}
    for model in models:
        pattern = os.path.join("outputs", task, f"{model}_full", "seed_*", "*", "eval_recomputed", "eval_metrics.json")
        files = sorted(glob.glob(pattern))
        print(f"  {task}/{model}: pattern={pattern} -> {len(files)} files")
        rewards = []
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            r = data.get("episode_reward/mean", data.get("reward_mean", data.get("mean_reward")))
            if r is not None:
                rewards.append(r)
        if rewards:
            m = round(float(np.mean(rewards)), 2)
            s = round(float(np.std(rewards, ddof=1)), 2)
            out[task][model] = {"mean": m, "std": s}
            print(f"    -> {m} +/- {s} (n={len(rewards)})")

os.makedirs("results/aggregated", exist_ok=True)
with open("results/aggregated/main_comparison.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved results/aggregated/main_comparison.json")
print(json.dumps(out, indent=2))
