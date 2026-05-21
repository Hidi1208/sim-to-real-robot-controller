# Sim-to-Real Robot Controller

A reinforcement learning project training a PPO agent to navigate a wheeled racecar to random goal positions in PyBullet simulation. The project benchmarks PPO against a classical PID baseline across clean and degraded conditions (sensor noise, motor noise, friction variation) to quantify the sim-to-real gap — then closes that gap using domain randomization during training.

## Results

Three-way comparison across 8 noise conditions (50 episodes each, fixed seeds):

| Condition | PID | PPO | PPO (robust) |
|---|---|---|---|
| Clean (baseline) | 70% | **88%** | 70% |
| Sensor noise (low) | 72% | 84% | 74% |
| Sensor noise (high) | 74% | 70% | **78%** |
| Motor noise (low) | 70% | **88%** | 76% |
| Motor noise (high) | 70% | **78%** | 68% |
| Friction variation | 70% | **86%** | 72% |
| Combined (realistic) | 72% | **86%** | 72% |
| Combined (harsh) | 74% | 64% | **72%** |

See `comparison.png` for PID vs PPO reward distributions and distance trajectories, and `sim2real.png` for the full three-way noise analysis bar chart.

## Project Structure

| File | Description |
|---|---|
| `robot_env.py` | Gymnasium environment — racecar URDF, Ackermann steering, curriculum learning |
| `train_ppo.py` | Train a standard PPO agent (500k–1M steps) |
| `train_ppo_robust.py` | Train PPO with domain randomization (noise + friction variation during training) |
| `pid_controller.py` | Classical PID baseline — steers toward goal, throttles on alignment |
| `evaluate.py` | Head-to-head PID vs PPO comparison with plots |
| `sim2real.py` | Three-way noise robustness analysis across 8 degraded conditions |
| `debug.py` | Visual inspection — random actions, trained model, or manual keyboard control |

## How to Run

**Install:**
```
pip install pybullet gymnasium stable-baselines3 matplotlib numpy
```

**Train:**
```bash
python train_ppo.py                        # standard PPO, 500k steps
python train_ppo.py --timesteps 1000000    # train longer
python train_ppo_robust.py                 # domain-randomized PPO, 1M steps
```

**Evaluate:**
```bash
python evaluate.py                         # PID vs PPO, 50 episodes headless
python evaluate.py --episodes 100          # smoother stats
python sim2real.py                         # three-way noise analysis
```

**Debug / visualize:**
```bash
python debug.py                            # random actions (env smoke test)
python debug.py --model                    # watch trained PPO drive
python debug.py --manual                   # keyboard control (arrow keys)
python pid_controller.py                   # watch PID drive (GUI)
python pid_controller.py --headless --episodes 20
```

**Monitor training:**
```
tensorboard --logdir logs/
```

## Key Findings

- **Standard PPO leads in clean conditions (+18% over PID) but collapses under high sensor noise**, dropping to parity with PID at `obs_noise=0.3` and falling 10% below PID under combined harsh conditions.
- **Domain randomization restores robustness** — the robust model recovers the sensor noise (high) gap (78% vs 70%) and holds under combined harsh conditions (72% vs 64%), directly fixing the failure cases of the standard model.
- **PID is the most stable baseline** — success rate varies only 70–74% across all conditions, never collapsing. Classical controllers don't degrade gracefully; they just don't degrade.
- **Robustness has a cost** — the domain-randomized model matches PID in clean conditions (70%) rather than matching standard PPO (88%), illustrating the standard sim-to-real tradeoff between peak performance and noise tolerance.

## Stack

- [PyBullet](https://pybullet.org) — physics simulation (racecar URDF with Ackermann steering)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io) — PPO implementation
- [Gymnasium](https://gymnasium.farama.org) — RL environment interface
- [NumPy](https://numpy.org) — numerics
- [matplotlib](https://matplotlib.org) — plots
