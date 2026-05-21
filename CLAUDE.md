# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Sim-to-real robot controller for a robotics master's application portfolio. A wheeled racecar navigates to random goal positions on a flat plane using PPO, benchmarked against a classical PID baseline across clean and degraded conditions to quantify and close the sim-to-real gap.

## Stack

- **PyBullet** — physics simulation (racecar URDF with Ackermann steering)
- **Stable-Baselines3** — PPO implementation
- **Gymnasium** — RL environment interface
- **NumPy**, **matplotlib** — numerics and plotting

## Commands

```bash
# Train
python train_ppo.py                         # standard PPO, 500k steps (default)
python train_ppo.py --timesteps 1000000     # train longer
python train_ppo_robust.py                  # domain-randomized PPO, 1M steps

# Evaluate
python evaluate.py                          # PID vs PPO, 50 episodes headless
python evaluate.py --episodes 100           # smoother stats
python sim2real.py                          # three-way noise analysis (50 eps x 8 conditions)

# Debug / visualize
python debug.py                             # random actions (env smoke test)
python debug.py --model                     # watch trained PPO drive (GUI)
python debug.py --manual                    # keyboard control, arrow keys (GUI)
python pid_controller.py --headless --episodes 20

# Monitor training
tensorboard --logdir logs/
```

## Architecture

**`robot_env.py`** is the core Gymnasium environment:
- **Observation (7-dim):** `[dx, dy, yaw, dist_to_goal, angle_to_goal, linear_vel, angular_vel]`
- **Action (2-dim, continuous [-1, 1]):** `[throttle, steering]` — scaled to ±10 rad/s and ±0.785 rad
- **Joint wiring:** drive wheels are joints `[2, 3, 5, 7]` (velocity control); steering hinges are joints `[4, 6]` (position control)
- **Sub-stepping:** `self.substeps = 4` — each `env.step()` runs 4 physics ticks, giving ~8 seconds of sim time at 500 max steps
- **Curriculum:** goal spawn range starts at 1.5 m, expands by 0.5 m when 60% success rate over 50 episodes, caps at 5.0 m
- **Reward:** `progress_reward + heading_reward + forward_bonus + time_penalty`; +20 on reaching goal (dist < 0.5 m)

**`pid_controller.py`** — PD steering on `angle_to_goal`, throttle scaled by alignment (`cos(angle_to_goal)`). Gains: `kp=1.5`, `kd=0.5`. Close-range slowdown at dist < 0.6 m.

**`train_ppo_robust.py`** wraps `RobotEnv` in `DomainRandomizedEnv(gym.Env)`, injecting per-step sensor noise (0–0.15), motor noise (0–0.15), and per-episode friction randomization (0.4–1.3) during training. Saves to `ppo_robot_robust.zip`.

**`sim2real.py`** wraps `RobotEnv` in `NoisyEnvWrapper` (not a `gym.Env` — used for evaluation only, not training) to test all three controllers across 8 noise conditions with fixed seeds.

## Key details

- Trained models are gitignored (`*.zip`) — run `train_ppo.py` and `train_ppo_robust.py` to regenerate.
- `comparison.png` and `sim2real.png` are committed output artifacts from the last evaluation run.
- Windows console is cp1252 — avoid Unicode box-drawing characters (`─`, `→`, `—`) in print statements.
- `joints.py` and `teleop.py` are untracked utility scripts for URDF inspection and manual driving — not part of the training pipeline.
