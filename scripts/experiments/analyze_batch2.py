#!/usr/bin/env python3
"""Batch 2 Confirmation Analysis: V2 (wider_all) vs Baseline DynaMITE
Computes per-seed metrics, 3-seed means, RobustScore, and paired t-tests."""

import json
import os
import sys
import numpy as np
from scipy import stats

# === Configuration ===
SEEDS = [42, 43, 44]
SWEEPS = ["push_magnitude", "friction", "action_delay"]

# Baseline sweep results directory
BASELINE_DIR = "results/sweeps_multiseed"
# V2 wider_all confirmation results
V2_DIR = "results/batch2_confirm/wider_all"

# Baseline per-seed eval (for nominal reward)
BASELINE_EVAL_DIRS = {
    42: "outputs/randomized/dynamite_full/seed_42",
    43: "outputs/randomized/dynamite_full/seed_43",
    44: "outputs/randomized/dynamite_full/seed_44",
}

# Latent scores (baseline only; V2 uses same architecture so latent=0.5 assumed)
BASELINE_LATENT = 0.500
V2_LATENT = 0.500  # Same architecture, assumed same latent quality

def load_sweep(path):
    """Load sweep JSON and return list of (value, mean_reward) tuples."""
    with open(path) as f:
        data = json.load(f)
    return [(r["value"], r["episode_reward/mean"]) for r in data["results"]]

def compute_sweep_metrics(results):
    """From sweep results, compute nominal, worst_case, sensitivity."""
    rewards = [r for _, r in results]
    nominal = rewards[0]  # First level is always easiest/nominal
    worst_case = min(rewards)
    sensitivity = abs(worst_case - nominal)
    return nominal, worst_case, sensitivity

def compute_robustscore(nominal, worst_case, neg_sensitivity, ood_avg, latent):
    """Composite RobustScore: 0.30*nom + 0.30*worst + 0.20*(-sens) + 0.10*ood + 0.10*lat"""
    return 0.30 * nominal + 0.30 * worst_case + 0.20 * (-neg_sensitivity) + 0.10 * ood_avg + 0.10 * latent

print("=" * 80)
print("BATCH 2 CONFIRMATION ANALYSIS: V2 (wider_all) vs Baseline DynaMITE")
print("=" * 80)

# === Load all results ===
baseline_data = {}  # {seed: {sweep: [(value, reward), ...]}}
v2_data = {}

for seed in SEEDS:
    baseline_data[seed] = {}
    v2_data[seed] = {}
    for sweep in SWEEPS:
        # Baseline
        bl_path = f"{BASELINE_DIR}/{sweep}/dynamite_seed{seed}/sweep_{sweep}.json"
        if os.path.exists(bl_path):
            baseline_data[seed][sweep] = load_sweep(bl_path)
        else:
            print(f"WARNING: Missing baseline sweep {bl_path}")
            
        # V2
        v2_path = f"{V2_DIR}/seed_{seed}/sweep_{sweep}.json"
        if os.path.exists(v2_path):
            v2_data[seed][sweep] = load_sweep(v2_path)
        else:
            print(f"WARNING: Missing V2 sweep {v2_path}")

# === Per-seed analysis ===
print("\n--- Per-Seed Push Magnitude Sweep ---")
print(f"{'Seed':>6} | {'Baseline Nom':>12} {'BL Worst':>10} {'BL Sens':>10} | {'V2 Nom':>10} {'V2 Worst':>10} {'V2 Sens':>10} | {'Δ Nom':>8} {'Δ Worst':>8} {'Δ Sens%':>8}")
print("-" * 115)

bl_push_metrics = []
v2_push_metrics = []

for seed in SEEDS:
    if "push_magnitude" in baseline_data[seed] and "push_magnitude" in v2_data[seed]:
        bl_nom, bl_worst, bl_sens = compute_sweep_metrics(baseline_data[seed]["push_magnitude"])
        v2_nom, v2_worst, v2_sens = compute_sweep_metrics(v2_data[seed]["push_magnitude"])
        bl_push_metrics.append((bl_nom, bl_worst, bl_sens))
        v2_push_metrics.append((v2_nom, v2_worst, v2_sens))
        
        d_nom = v2_nom - bl_nom
        d_worst = v2_worst - bl_worst
        d_sens_pct = (v2_sens - bl_sens) / bl_sens * 100
        
        print(f"{seed:>6} | {bl_nom:>12.4f} {bl_worst:>10.4f} {bl_sens:>10.4f} | {v2_nom:>10.4f} {v2_worst:>10.4f} {v2_sens:>10.4f} | {d_nom:>+8.4f} {d_worst:>+8.4f} {d_sens_pct:>+7.1f}%")

# === Per-seed RobustScore ===
print("\n--- Per-Seed RobustScore ---")
bl_scores = []
v2_scores = []

for i, seed in enumerate(SEEDS):
    # Aggregate across all sweeps for this seed
    bl_nominals = []
    bl_worsts = []
    bl_sensitivities = []
    bl_all_rewards = []
    
    v2_nominals = []
    v2_worsts = []
    v2_sensitivities = []
    v2_all_rewards = []
    
    for sweep in SWEEPS:
        if sweep in baseline_data[seed]:
            nom, worst, sens = compute_sweep_metrics(baseline_data[seed][sweep])
            bl_nominals.append(nom)
            bl_worsts.append(worst)
            bl_sensitivities.append(sens)
            bl_all_rewards.extend([r for _, r in baseline_data[seed][sweep]])
            
        if sweep in v2_data[seed]:
            nom, worst, sens = compute_sweep_metrics(v2_data[seed][sweep])
            v2_nominals.append(nom)
            v2_worsts.append(worst)
            v2_sensitivities.append(sens)
            v2_all_rewards.extend([r for _, r in v2_data[seed][sweep]])
    
    bl_nominal = np.mean(bl_nominals)
    bl_worst = min(bl_worsts)
    bl_sens = np.mean(bl_sensitivities)
    bl_ood = np.mean(bl_all_rewards)
    bl_rs = compute_robustscore(bl_nominal, bl_worst, bl_sens, bl_ood, BASELINE_LATENT)
    bl_scores.append(bl_rs)
    
    v2_nominal = np.mean(v2_nominals)
    v2_worst = min(v2_worsts)
    v2_sens = np.mean(v2_sensitivities)
    v2_ood = np.mean(v2_all_rewards)
    v2_rs = compute_robustscore(v2_nominal, v2_worst, v2_sens, v2_ood, V2_LATENT)
    v2_scores.append(v2_rs)
    
    print(f"  Seed {seed}: Baseline RS={bl_rs:.4f} | V2 RS={v2_rs:.4f} | Δ={v2_rs - bl_rs:+.4f}")
    print(f"    BL: nom={bl_nominal:.4f}, worst={bl_worst:.4f}, sens={bl_sens:.4f}, ood={bl_ood:.4f}")
    print(f"    V2: nom={v2_nominal:.4f}, worst={v2_worst:.4f}, sens={v2_sens:.4f}, ood={v2_ood:.4f}")

# === 3-Seed Aggregation ===
print("\n" + "=" * 80)
print("3-SEED AGGREGATED RESULTS")
print("=" * 80)

bl_mean_rs = np.mean(bl_scores)
bl_std_rs = np.std(bl_scores, ddof=1)
bl_ci = stats.t.ppf(0.975, df=2) * bl_std_rs / np.sqrt(3)

v2_mean_rs = np.mean(v2_scores)
v2_std_rs = np.std(v2_scores, ddof=1)
v2_ci = stats.t.ppf(0.975, df=2) * v2_std_rs / np.sqrt(3)

delta_rs = v2_mean_rs - bl_mean_rs

print(f"  Baseline RobustScore: {bl_mean_rs:.4f} ± {bl_ci:.4f} (95% CI)")
print(f"  V2 wider_all RS:     {v2_mean_rs:.4f} ± {v2_ci:.4f} (95% CI)")
print(f"  Δ RobustScore:       {delta_rs:+.4f}")

# Paired t-test
t_stat, p_value = stats.ttest_rel(v2_scores, bl_scores)
print(f"\n  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
print(f"  Significant at α=0.05? {'YES' if p_value < 0.05 else 'NO'}")

# === Push-specific metrics (3-seed mean) ===
print("\n--- Push Magnitude: 3-Seed Means ---")
if bl_push_metrics and v2_push_metrics:
    bl_nom_mean = np.mean([m[0] for m in bl_push_metrics])
    bl_worst_mean = np.mean([m[1] for m in bl_push_metrics])
    bl_sens_mean = np.mean([m[2] for m in bl_push_metrics])
    
    v2_nom_mean = np.mean([m[0] for m in v2_push_metrics])
    v2_worst_mean = np.mean([m[1] for m in v2_push_metrics])
    v2_sens_mean = np.mean([m[2] for m in v2_push_metrics])
    
    d_nom = v2_nom_mean - bl_nom_mean
    d_worst = v2_worst_mean - bl_worst_mean
    d_sens_pct = (v2_sens_mean - bl_sens_mean) / bl_sens_mean * 100
    
    print(f"  Baseline: nominal={bl_nom_mean:.4f}, worst={bl_worst_mean:.4f}, sensitivity={bl_sens_mean:.4f}")
    print(f"  V2:       nominal={v2_nom_mean:.4f}, worst={v2_worst_mean:.4f}, sensitivity={v2_sens_mean:.4f}")
    print(f"  Δ:        nominal={d_nom:+.4f}, worst={d_worst:+.4f}, sensitivity={d_sens_pct:+.1f}%")
    
    # Paired t-tests on push metrics
    bl_noms = [m[0] for m in bl_push_metrics]
    v2_noms = [m[0] for m in v2_push_metrics]
    t_nom, p_nom = stats.ttest_rel(v2_noms, bl_noms)
    
    bl_worsts = [m[1] for m in bl_push_metrics]
    v2_worsts = [m[1] for m in v2_push_metrics]
    t_worst, p_worst = stats.ttest_rel(v2_worsts, bl_worsts)
    
    bl_sens_vals = [m[2] for m in bl_push_metrics]
    v2_sens_vals = [m[2] for m in v2_push_metrics]
    t_sens, p_sens = stats.ttest_rel(v2_sens_vals, bl_sens_vals)
    
    print(f"\n  Paired t-tests (push):")
    print(f"    Nominal:     t={t_nom:.3f}, p={p_nom:.4f}")
    print(f"    Worst-case:  t={t_worst:.3f}, p={p_worst:.4f}")
    print(f"    Sensitivity: t={t_sens:.3f}, p={p_sens:.4f}")

# === Decision ===
print("\n" + "=" * 80)
print("DECISION")
print("=" * 80)

criteria_met = []
criteria_failed = []

# Check promotion criteria
if bl_push_metrics and v2_push_metrics:
    # Worst-case improves ≥0.15
    d_worst_mean = v2_worst_mean - bl_worst_mean
    if d_worst_mean >= 0.15:
        criteria_met.append(f"Worst-case improved by {d_worst_mean:+.4f} (≥0.15)")
    else:
        criteria_failed.append(f"Worst-case: {d_worst_mean:+.4f} (need ≥+0.15)")
    
    # Sensitivity reduces ≥15%
    if d_sens_pct <= -15:
        criteria_met.append(f"Sensitivity reduced by {-d_sens_pct:.1f}% (≥15%)")
    else:
        criteria_failed.append(f"Sensitivity: {d_sens_pct:+.1f}% (need ≤-15%)")
    
    # Nominal loss ≤0.10
    if abs(d_nom) <= 0.10:
        criteria_met.append(f"Nominal loss: {abs(d_nom):.4f} (≤0.10)")
    else:
        criteria_failed.append(f"Nominal loss: {abs(d_nom):.4f} (need ≤0.10)")

# RobustScore improvement
if delta_rs > 0:
    criteria_met.append(f"RobustScore improved: {delta_rs:+.4f}")
else:
    criteria_failed.append(f"RobustScore declined: {delta_rs:+.4f}")

print(f"\n  Criteria MET:")
for c in criteria_met:
    print(f"    ✓ {c}")
print(f"\n  Criteria FAILED:")
for c in criteria_failed:
    print(f"    ✗ {c}")

# Final decision
if p_value < 0.05 and delta_rs > 0 and any("Sensitivity reduced" in c or "Worst-case improved" in c for c in criteria_met):
    decision = "ACCEPT"
elif delta_rs > 0 and len(criteria_met) > len(criteria_failed):
    decision = "TENTATIVE ACCEPT (not statistically significant)"
else:
    decision = "REJECT"

print(f"\n  FINAL DECISION: {decision}")
print(f"  Statistical significance: p={p_value:.4f}")
print("=" * 80)
