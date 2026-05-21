"""
Sim-to-Real gap analysis: test PID and PPO under degraded conditions.

Applies three types of real-world noise independently and combined:
  1. Observation noise (noisy sensors)
  2. Action noise (imprecise motors)
  3. Friction variation (different floor surfaces)

Usage:
  python sim2real.py                → full analysis, 50 episodes per condition
  python sim2real.py --episodes 100 → more episodes for smoother results
  python sim2real.py --render       → watch a noisy run
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from robot_env import RobotEnv
from pid_controller import PIDController
from stable_baselines3 import PPO
import pybullet as p


class NoisyEnvWrapper:
    """Wraps RobotEnv to inject real-world noise."""

    def __init__(self, env, obs_noise=0.0, action_noise=0.0, friction_range=None):
        self.env = env
        self.obs_noise = obs_noise
        self.action_noise = action_noise
        self.friction_range = friction_range  # (low, high) or None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Randomize floor friction on each episode
        if self.friction_range is not None:
            friction = np.random.uniform(*self.friction_range)
            p.changeDynamics(
                0, -1,  # plane body, base link
                lateralFriction=friction,
                physicsClientId=self.env.client
            )

        return self._add_obs_noise(obs), info

    def step(self, action):
        # Add motor noise
        if self.action_noise > 0:
            noise = np.random.normal(0, self.action_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_obs_noise(obs), reward, terminated, truncated, info

    def _add_obs_noise(self, obs):
        if self.obs_noise > 0:
            noise = np.random.normal(0, self.obs_noise, size=obs.shape)
            return obs + noise.astype(np.float32)
        return obs

    def close(self):
        self.env.close()


def run_condition(controller_name, act_fn, episodes, seeds, render=False, **noise_kwargs):
    """Evaluate a controller under a specific noise condition."""
    import time
    env = RobotEnv(render=render)
    wrapped = NoisyEnvWrapper(env, **noise_kwargs)

    results = []
    for seed in seeds:
        obs, _ = wrapped.reset(seed=int(seed))
        total_reward = 0

        for step in range(500):
            action = act_fn(obs)
            obs, reward, terminated, truncated, _ = wrapped.step(action)
            total_reward += reward
            if render:
                time.sleep(1 / 60)
            if terminated or truncated:
                break

        results.append({
            "success": terminated,
            "steps": step + 1,
            "reward": total_reward,
            "final_dist": obs[3]
        })

    wrapped.close()

    success_rate = 100 * sum(r["success"] for r in results) / len(results)
    avg_reward = np.mean([r["reward"] for r in results])
    return success_rate, avg_reward, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    seeds = np.random.RandomState(42).randint(0, 100000, size=args.episodes)

    # Load controllers
    model = PPO.load("ppo_robot")
    model_robust = PPO.load("ppo_robot_robust")
    pid = PIDController()

    def ppo_act(obs):
        action, _ = model.predict(obs)
        return action

    def ppo_robust_act(obs):
        action, _ = model_robust.predict(obs)
        return action

    def pid_act(obs):
        return pid.act(obs)

    # Define noise conditions: (label, kwargs)
    conditions = [
        ("Clean (baseline)",      {}),
        ("Sensor noise (low)",    {"obs_noise": 0.1}),
        ("Sensor noise (high)",   {"obs_noise": 0.3}),
        ("Motor noise (low)",     {"action_noise": 0.1}),
        ("Motor noise (high)",    {"action_noise": 0.3}),
        ("Friction variation",    {"friction_range": (0.3, 1.5)}),
        ("Combined (realistic)",  {"obs_noise": 0.1, "action_noise": 0.1, "friction_range": (0.5, 1.2)}),
        ("Combined (harsh)",      {"obs_noise": 0.3, "action_noise": 0.2, "friction_range": (0.3, 1.5)}),
    ]

    # Run all conditions
    pid_rates = []
    ppo_rates = []
    robust_rates = []
    labels = []

    print("\n" + "=" * 70)
    print("  Sim-to-Real Gap Analysis: PID vs PPO vs PPO (robust)")
    print("=" * 70)

    for label, kwargs in conditions:
        print(f"\n  Condition: {label}")
        print(f"  {'-' * 50}")

        pid.reset()
        pid_rate, pid_reward, _       = run_condition("PID",        pid_act,        args.episodes, seeds, args.render, **kwargs)
        ppo_rate, ppo_reward, _       = run_condition("PPO",        ppo_act,        args.episodes, seeds, args.render, **kwargs)
        robust_rate, robust_reward, _ = run_condition("PPO robust", ppo_robust_act, args.episodes, seeds, args.render, **kwargs)

        print(f"  PID:         {pid_rate:.0f}% success | avg reward: {pid_reward:.1f}")
        print(f"  PPO:         {ppo_rate:.0f}% success | avg reward: {ppo_reward:.1f}")
        print(f"  PPO (robust):{robust_rate:.0f}% success | avg reward: {robust_reward:.1f}")

        pid_rates.append(pid_rate)
        ppo_rates.append(ppo_rate)
        robust_rates.append(robust_rate)
        labels.append(label)

    # Plot
    plot_sim2real(labels, pid_rates, ppo_rates, robust_rates)

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Condition':<25} {'PID':>6} {'PPO':>6} {'Robust':>8}")
    print(f"  {'-' * 47}")
    for i, label in enumerate(labels):
        print(f"  {label:<25} {pid_rates[i]:>5.0f}% {ppo_rates[i]:>5.0f}% {robust_rates[i]:>7.0f}%")

    harsh_ppo_gap    = ppo_rates[-1]    - pid_rates[-1]
    harsh_robust_gap = robust_rates[-1] - pid_rates[-1]
    print(f"\n  Harsh condition - PPO vs PID:         {harsh_ppo_gap:+.0f}%")
    print(f"  Harsh condition - PPO (robust) vs PID:{harsh_robust_gap:+.0f}%")


def plot_sim2real(labels, pid_rates, ppo_rates, robust_rates, save_path="sim2real.png"):
    """Bar chart comparing PID vs PPO vs PPO (robust) across noise conditions."""
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    bars1 = ax.bar(x - width, pid_rates,    width, label="PID",         color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x,         ppo_rates,    width, label="PPO",         color="#2ecc71", alpha=0.85)
    bars3 = ax.bar(x + width, robust_rates, width, label="PPO (robust)", color="#3498db", alpha=0.85)

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Sim-to-Real Gap: Controller Performance Under Noise")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    for bar in [*bars1, *bars2, *bars3]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
