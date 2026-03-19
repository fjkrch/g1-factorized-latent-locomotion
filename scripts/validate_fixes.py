#!/usr/bin/env python3
"""
Validate all critical physics fixes in live Isaac Lab simulation.

Tests:
  1. Push velocity actually reaches PhysX (root_vel_w changes on push steps)
  2. Material properties are 3D tensors and set_material_properties works
  3. Mass randomization via set_masses works
  4. Per-env DR produces different physics per environment
  5. DR application runs without errors/device mismatches

Run:
    python scripts/validate_fixes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must init sim BEFORE any Isaac Lab imports
from src.envs.g1_env import init_sim
init_sim(headless=True)

import torch
from src.utils.config import load_config
from src.envs.g1_env import G1EnvWrapper


def make_env_no_close(cfg, device="cuda"):
    """Create env wrapper without closing SimulationApp."""
    env = G1EnvWrapper(cfg, device=device)
    env._create_isaac_env()
    return env


def test_push_velocity(env):
    """Test that push disturbance actually modifies robot velocity in PhysX."""
    print("\n" + "=" * 60)
    print("TEST 1: Push velocity reaches PhysX")
    print("=" * 60)

    env.reset()
    robot = env._rl_env.scene["robot"]
    push_steps = env._push_steps
    print(f"  Push steps: {push_steps}")
    print(f"  Push vel range: {env.task_cfg['domain_randomization']['push_vel_range']}")

    velocity_changed = False
    for step in range(max(push_steps) + 2):
        actions = torch.zeros(env.num_envs, env.act_dim, device=env.device)
        vel_before = robot.data.root_vel_w.clone()
        result = env.step(actions)
        vel_after = robot.data.root_vel_w.clone()

        if result["push_applied"].any():
            push_ids = result["push_applied"].nonzero(as_tuple=False).squeeze(-1)
            vel_diff = (vel_after[push_ids, :3] - vel_before[push_ids, :3]).abs()
            max_diff = vel_diff.max().item()
            print(f"  Step {step + 1}: PUSH to envs {push_ids.tolist()}, vel change: {max_diff:.4f} m/s")
            if max_diff > 0.01:
                velocity_changed = True
                print(f"    ✓ Velocity changed significantly")
            else:
                print(f"    ✗ Velocity change too small!")

    if velocity_changed:
        print("  >>> PASS: Push velocity reaches PhysX")
    else:
        print("  >>> FAIL: Push velocity NOT reaching PhysX")
    return velocity_changed


def test_material_properties(env):
    """Test that material properties tensor is 3D and set works."""
    print("\n" + "=" * 60)
    print("TEST 2: Material properties (friction/restitution)")
    print("=" * 60)

    env.reset()
    robot = env._rl_env.scene["robot"]
    view = robot.root_physx_view

    mat = view.get_material_properties()
    print(f"  Material tensor shape: {mat.shape} (device: {mat.device})")
    assert len(mat.shape) == 3, f"Expected 3D tensor, got {len(mat.shape)}D"
    print(f"  ✓ Tensor is 3D: [num_envs={mat.shape[0]}, num_shapes={mat.shape[1]}, 3]")

    orig_friction = mat[:, 0, 0].clone()
    print(f"  Original friction: {[f'{f:.3f}' for f in orig_friction.tolist()]}")

    # Set different friction per env
    test_frictions = [0.1, 0.5, 1.5, 2.5]
    for i, fric in enumerate(test_frictions):
        mat[i, :, 0] = fric
        mat[i, :, 1] = fric
    indices = torch.arange(env.num_envs, device=mat.device)
    view.set_material_properties(mat, indices)

    mat_after = view.get_material_properties()
    readback = mat_after[:, 0, 0].tolist()
    print(f"  After set: {[f'{f:.3f}' for f in readback]}")

    success = True
    for i, (expected, actual) in enumerate(zip(test_frictions, readback)):
        if abs(expected - actual) > 0.01:
            print(f"  ✗ Env {i}: expected {expected}, got {actual:.4f}")
            success = False
        else:
            print(f"  ✓ Env {i}: friction = {actual:.4f}")

    if success:
        print("  >>> PASS: Material properties write & readback correct")
    else:
        print("  >>> FAIL: Material properties NOT persisting")
    return success


def test_mass_randomization(env):
    """Test that mass randomization via set_masses works."""
    print("\n" + "=" * 60)
    print("TEST 3: Mass randomization")
    print("=" * 60)

    env.reset()
    robot = env._rl_env.scene["robot"]
    view = robot.root_physx_view

    try:
        masses = view.get_masses()
        print(f"  Masses tensor shape: {masses.shape} (device: {masses.device})")
        print(f"  Root body mass (env 0): {masses[0, 0].item():.3f} kg")
    except Exception as e:
        print(f"  ✗ get_masses() failed: {e}")
        return False

    if hasattr(env, '_base_masses'):
        print(f"  ✓ Base masses cached: {env._base_masses[0, 0].item():.3f} kg")
    else:
        print(f"  ✗ Base masses not cached")

    added_mass = env._dynamics_params["mass"][:, 0]
    print(f"  Added mass per env: {[f'{m:.3f}' for m in added_mass.tolist()]}")

    current = view.get_masses()
    success = True
    for i in range(min(4, env.num_envs)):
        if hasattr(env, '_base_masses'):
            expected = env._base_masses[i, 0].item() + added_mass[i].item()
            actual = current[i, 0].item()
            diff = abs(expected - actual)
            if diff > 0.1:
                print(f"  ✗ Env {i}: expected {expected:.3f}, got {actual:.3f}")
                success = False
            else:
                print(f"  ✓ Env {i}: mass {actual:.3f} = {env._base_masses[i, 0].item():.3f} + {added_mass[i].item():.3f}")
        else:
            print(f"  ? Env {i}: mass {current[i, 0].item():.3f} (no base reference)")
            success = False

    if success:
        print("  >>> PASS: Mass randomization works")
    else:
        print("  >>> FAIL: Mass randomization NOT working")
    return success


def test_per_env_dr(env):
    """Test that different envs get different DR parameters."""
    print("\n" + "=" * 60)
    print("TEST 4: Per-env domain randomization variety")
    print("=" * 60)

    env.reset()

    friction = env._dynamics_params["friction"][:, 0]
    mass = env._dynamics_params["mass"][:, 0]

    print(f"  Friction: {[f'{f:.3f}' for f in friction.tolist()]}")
    print(f"  Mass:     {[f'{m:.3f}' for m in mass.tolist()]}")

    fric_unique = len(set([round(f, 4) for f in friction.tolist()]))
    mass_unique = len(set([round(m, 4) for m in mass.tolist()]))
    print(f"  Unique friction: {fric_unique}/{env.num_envs}")
    print(f"  Unique mass: {mass_unique}/{env.num_envs}")

    success = fric_unique > 1 and mass_unique > 1

    if success:
        print("  >>> PASS: Per-env DR produces varied parameters")
    else:
        print("  >>> FAIL: All envs have identical parameters")
    return success


def test_dr_application(env):
    """Test that _apply_dynamics_to_env runs without errors."""
    print("\n" + "=" * 60)
    print("TEST 5: DR application (no device errors)")
    print("=" * 60)

    env.reset()

    try:
        env._apply_dynamics_to_env()
        print("  ✓ _apply_dynamics_to_env() ran without exception")

        robot = env._rl_env.scene["robot"]
        view = robot.root_physx_view
        mat = view.get_material_properties()
        expected_fric = env._dynamics_params["friction"][0, 0].item()
        actual_fric = mat[0, 0, 0].item()
        print(f"  Expected friction env[0]: {expected_fric:.4f}")
        print(f"  Readback friction env[0]: {actual_fric:.4f}")

        if abs(expected_fric - actual_fric) < 0.05:
            print("  ✓ Friction correctly applied to PhysX")
            print("  >>> PASS: DR applies without errors")
            return True
        else:
            print(f"  ✗ Friction mismatch: {abs(expected_fric - actual_fric):.4f}")
            print("  >>> FAIL: DR values not persisting in PhysX")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        print("  >>> FAIL: DR application raised exception")
        return False


if __name__ == "__main__":
    cfg = load_config(
        base_path="configs/base.yaml",
        task_path="configs/task/push.yaml",
    )
    cfg["task"]["num_envs"] = 4

    env = make_env_no_close(cfg, device="cuda")

    results = {}
    results["push_velocity"] = test_push_velocity(env)
    results["material_properties"] = test_material_properties(env)
    results["mass_randomization"] = test_mass_randomization(env)
    results["per_env_dr"] = test_per_env_dr(env)
    results["dr_application"] = test_dr_application(env)

    # Print summary BEFORE close (SimulationApp.close() does a hard exit)
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s}: {status}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED — physics fixes validated!")
    else:
        print("SOME TESTS FAILED — review output above")
    print("=" * 60)
    sys.stdout.flush()
    sys.stderr.flush()

    # Now close (this may hard-exit the process)
    env.close()
