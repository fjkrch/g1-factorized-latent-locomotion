#!/usr/bin/env python3
"""Quick verification of all code changes."""
import sys
sys.path.insert(0, "/home/chyanin/robotpaper")

# 1. g1_env imports
from src.envs.g1_env import G1EnvWrapper, init_sim, make_env, _GYM_ID_MAP
print("[1] g1_env imports OK")
print("    Gym IDs:", _GYM_ID_MAP)

# 2. Config loads correctly
from src.utils.config import load_config
cfg = load_config(
    "configs/base.yaml",
    "configs/task/flat.yaml",
    "configs/model/mlp.yaml",
    "configs/train/default.yaml",  # doesn't exist — should not crash
)
assert cfg["task"]["num_envs"] == 512, f"Expected 512, got {cfg['task']['num_envs']}"
print(f"[2] Config OK: num_envs={cfg['task']['num_envs']}")

# 3. Mock env works (no SimulationApp)
env = G1EnvWrapper(cfg, device="cpu")
env._create_isaac_env()  # should fall back to mock
assert env._env is None
data = env.reset()
assert "obs" in data and "cmd" in data
step = env.step(data["obs"][:, :cfg["task"]["observation"]["action_dim"]])
assert "obs" in step and "reward" in step and "done" in step
print("[3] Mock env OK")

# 4. train.py parses args
import subprocess, os
result = subprocess.run(
    [sys.executable, "scripts/train.py", "--help"],
    capture_output=True, text=True, cwd="/home/chyanin/robotpaper"
)
assert "--headless" in result.stdout, "Missing --headless in train.py"
assert "--variant" in result.stdout, "Missing --variant in train.py"
print("[4] train.py --help OK (has --headless, --variant)")

# 5. eval.py parses args
result = subprocess.run(
    [sys.executable, "scripts/eval.py", "--help"],
    capture_output=True, text=True, cwd="/home/chyanin/robotpaper"
)
assert "--headless" in result.stdout, "Missing --headless in eval.py"
print("[5] eval.py --help OK (has --headless)")

# 6. Shell script has conda activate
with open("scripts/run_train.sh") as f:
    sh = f.read()
assert "conda activate" in sh, "Missing conda activate in run_train.sh"
assert "--headless" in sh, "Missing --headless in run_train.sh"
print("[6] run_train.sh OK (conda activate, --headless)")

# 7. run_all_main.sh has conda
with open("scripts/run_all_main.sh") as f:
    sh = f.read()
assert "conda activate" in sh
print("[7] run_all_main.sh OK")

print("\n✓ ALL CHECKS PASSED")
