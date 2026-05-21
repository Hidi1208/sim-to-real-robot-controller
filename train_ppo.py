"""
Train a PPO agent on the racecar navigation task.

Usage:
  python train_ppo.py                    → train 500k steps
  python train_ppo.py --timesteps 1000000 → train longer
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from robot_env import RobotEnv


class CurriculumLogger(BaseCallback):
    """Logs curriculum progress and training stats."""

    def __init__(self, env, print_freq=10000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.print_freq = print_freq

    def _on_step(self):
        if self.num_timesteps % self.print_freq == 0:
            success_count = sum(self.env.recent_outcomes)
            total = len(self.env.recent_outcomes)
            rate = success_count / total if total > 0 else 0

            print(f"  Step {self.num_timesteps:>7d} | "
                  f"goal_range: {self.env.goal_range:.1f}m | "
                  f"success: {success_count}/{total} ({100*rate:.0f}%)")
        return True


def train(timesteps=500_000):
    env = RobotEnv(render=False)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )

    callback = CurriculumLogger(env, print_freq=10000)

    print(f"\n  Training PPO for {timesteps:,} steps...")
    print(f"  Starting goal range: {env.goal_range}m\n")

    model.learn(total_timesteps=timesteps, callback=callback)
    model.save("ppo_robot")

    print(f"\n  Training complete. Model saved as 'ppo_robot.zip'")
    print(f"  Final goal range: {env.goal_range}m")
    print(f"  Run 'python debug.py --model' to watch it drive.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()

    train(timesteps=args.timesteps)
