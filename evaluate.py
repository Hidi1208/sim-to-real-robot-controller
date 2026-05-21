"""
Head-to-head comparison: PID vs PPO on identical goal sets.
Produces success rates, average rewards, and a comparison plot.

Usage:
  python evaluate.py                → 50 episodes, headless
  python evaluate.py --episodes 100 → more episodes for smoother stats
  python evaluate.py --render       → watch both controllers (slower)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from robot_env import RobotEnv
from pid_controller import PIDController
from stable_baselines3 import PPO


def evaluate_controller(env, act_fn, episodes, seeds, render=False):
    """Run a controller over fixed seeds and collect metrics."""
    import time
    results = []

    for ep, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        total_reward = 0
        trajectory = []

        for step in range(500):
            action = act_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            trajectory.append(obs[3])  # track distance over time

            if render:
                time.sleep(1 / 60)
            if terminated or truncated:
                break

        results.append({
            "success": terminated,
            "steps": step + 1,
            "reward": total_reward,
            "final_dist": obs[3],
            "trajectory": trajectory
        })

    return results


def print_comparison(pid_results, ppo_results):
    """Print a side-by-side summary table."""
    def stats(results, label):
        successes = sum(r["success"] for r in results)
        n = len(results)
        avg_reward = np.mean([r["reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])
        avg_dist = np.mean([r["final_dist"] for r in results])
        success_steps = [r["steps"] for r in results if r["success"]]
        avg_success_steps = np.mean(success_steps) if success_steps else float("nan")

        print(f"\n  {label}")
        print(f"  {'-' * 40}")
        print(f"  Success rate:     {successes}/{n} ({100*successes/n:.0f}%)")
        print(f"  Avg reward:       {avg_reward:.1f}")
        print(f"  Avg steps:        {avg_steps:.0f}")
        print(f"  Avg final dist:   {avg_dist:.2f}m")
        print(f"  Avg steps (wins): {avg_success_steps:.0f}")

    print("\n" + "=" * 50)
    print("  PID vs PPO — Head-to-Head Comparison")
    print("=" * 50)
    stats(pid_results, "PID Controller")
    stats(ppo_results, "PPO Agent")


def plot_comparison(pid_results, ppo_results, save_path="comparison.png"):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Success rate bar chart
    pid_rate = 100 * sum(r["success"] for r in pid_results) / len(pid_results)
    ppo_rate = 100 * sum(r["success"] for r in ppo_results) / len(ppo_results)
    axes[0].bar(["PID", "PPO"], [pid_rate, ppo_rate], color=["#e74c3c", "#2ecc71"])
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Goal Reach Rate")
    axes[0].set_ylim(0, 100)

    # 2. Reward distribution
    pid_rewards = [r["reward"] for r in pid_results]
    ppo_rewards = [r["reward"] for r in ppo_results]
    axes[1].boxplot([pid_rewards, ppo_rewards], tick_labels=["PID", "PPO"])
    axes[1].set_ylabel("Episode Reward")
    axes[1].set_title("Reward Distribution")

    # 3. Distance over time (average trajectory)
    max_len = 500
    def pad_trajectories(results):
        padded = []
        for r in results:
            t = r["trajectory"]
            if len(t) < max_len:
                t = t + [t[-1]] * (max_len - len(t))
            padded.append(t[:max_len])
        return np.mean(padded, axis=0)

    pid_traj = pad_trajectories(pid_results)
    ppo_traj = pad_trajectories(ppo_results)
    axes[2].plot(pid_traj, label="PID", color="#e74c3c", alpha=0.8)
    axes[2].plot(ppo_traj, label="PPO", color="#2ecc71", alpha=0.8)
    axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Goal threshold")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Distance to Goal (m)")
    axes[2].set_title("Avg Distance Over Time")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Fixed seeds so both controllers face identical goals
    seeds = np.random.RandomState(42).randint(0, 100000, size=args.episodes)

    # --- PID ---
    print("  Running PID controller...")
    env = RobotEnv(render=args.render)
    pid = PIDController()

    def pid_act(obs):
        return pid.act(obs)

    pid_results = evaluate_controller(env, pid_act, args.episodes, seeds, args.render)
    env.close()

    # --- PPO ---
    print("  Running PPO agent...")
    env = RobotEnv(render=args.render)
    model = PPO.load("ppo_robot")

    def ppo_act(obs):
        action, _ = model.predict(obs)
        return action

    ppo_results = evaluate_controller(env, ppo_act, args.episodes, seeds, args.render)
    env.close()

    # --- Compare ---
    print_comparison(pid_results, ppo_results)
    plot_comparison(pid_results, ppo_results)

    # Reset PID state between episodes
    pid_original_act = pid_act
    def pid_act_with_reset(obs):
        return pid_original_act(obs)


if __name__ == "__main__":
    main()
