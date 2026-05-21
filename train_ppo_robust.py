"""
Train a PPO agent WITH domain randomization (noise during training).
Produces a robust model that handles sim-to-real transfer better.

Usage:
  python train_ppo_robust.py
"""

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from robot_env import RobotEnv
import pybullet as p


class DomainRandomizedEnv(gym.Env):
    """Wraps RobotEnv with randomized noise during training."""

    def __init__(self):
        super().__init__()
        self.env = RobotEnv(render=False)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.recent_outcomes = self.env.recent_outcomes

    @property
    def goal_range(self):
        return self.env.goal_range

    @goal_range.setter
    def goal_range(self, val):
        self.env.goal_range = val

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Randomize friction each episode
        friction = np.random.uniform(0.4, 1.3)
        p.changeDynamics(0, -1, lateralFriction=friction,
                         physicsClientId=self.env.client)

        return self._noisy_obs(obs), info

    def step(self, action):
        # Motor noise: random perturbation to actions
        noise_level = np.random.uniform(0.0, 0.15)
        action = action + np.random.normal(0, noise_level, size=action.shape)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(obs), reward, terminated, truncated, info

    def _noisy_obs(self, obs):
        # Sensor noise: varies per episode
        noise_level = np.random.uniform(0.0, 0.15)
        return obs + np.random.normal(0, noise_level, size=obs.shape).astype(np.float32)

    def close(self):
        self.env.close()


class CurriculumLogger(BaseCallback):
    def __init__(self, env, print_freq=10000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.print_freq = print_freq

    def _on_step(self):
        if self.num_timesteps % self.print_freq == 0:
            outcomes = self.env.recent_outcomes
            success_count = sum(outcomes)
            total = len(outcomes)
            rate = success_count / total if total > 0 else 0
            print(f"  Step {self.num_timesteps:>7d} | "
                  f"goal_range: {self.env.goal_range:.1f}m | "
                  f"success: {success_count}/{total} ({100*rate:.0f}%)")
        return True


def train(timesteps=1_000_000):
    env = DomainRandomizedEnv()

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

    print(f"\n  Training ROBUST PPO (with domain randomization)")
    print(f"  Noise: sensor 0-0.15, motor 0-0.15, friction 0.4-1.3")
    print(f"  Timesteps: {timesteps:,}\n")

    model.learn(total_timesteps=timesteps, callback=callback)
    model.save("ppo_robot_robust")

    print(f"\n  Model saved as 'ppo_robot_robust.zip'")
    print(f"  Final goal range: {env.goal_range}m")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()
    train(timesteps=args.timesteps)
