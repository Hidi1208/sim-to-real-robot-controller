"""
Debug / visual inspection tool.
Run this whenever you need to check what the robot is actually doing.

Modes:
  python debug.py              → random actions (test env works)
  python debug.py --model      → watch trained PPO model
  python debug.py --manual     → keyboard control (arrow keys)
"""

import argparse
import time
import numpy as np
import pybullet as p


def run_random(env, episodes=3):
    """Take random actions — just verify the env loads and physics work."""
    print("\n=== RANDOM ACTIONS ===")
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if step % 50 == 0:
                print(f"  Step {step:3d} | dist: {obs[3]:.2f} | angle: {obs[4]:.2f} | "
                      f"vel: {obs[5]:.2f} | reward: {reward:.3f}")

            time.sleep(1 / 60)
            if terminated or truncated:
                break

        print(f"  Episode {ep+1} done | steps: {step+1} | total reward: {total_reward:.1f}\n")


def run_model(env):
    """Watch a trained PPO model drive."""
    from stable_baselines3 import PPO

    try:
        model = PPO.load("ppo_robot")
    except FileNotFoundError:
        print("No trained model found. Run train_ppo.py first.")
        return

    print("\n=== TRAINED MODEL ===")
    for ep in range(5):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(500):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(1 / 60)
            if terminated or truncated:
                break

        result = "REACHED GOAL" if terminated else "timed out"
        print(f"  Episode {ep+1} | {result} | steps: {step+1} | reward: {total_reward:.1f}")


def run_manual(env):
    """Drive with arrow keys to build intuition."""
    print("\n=== MANUAL CONTROL ===")
    print("  UP/DOWN   = throttle")
    print("  LEFT/RIGHT = steer")
    print("  Q = quit\n")

    obs, _ = env.reset()
    while True:
        keys = p.getKeyboardEvents()
        throttle = 0.0
        steer = 0.0

        if p.B3G_UP_ARROW in keys:
            throttle = 1.0
        if p.B3G_DOWN_ARROW in keys:
            throttle = -1.0
        if p.B3G_LEFT_ARROW in keys:
            steer = 0.7
        if p.B3G_RIGHT_ARROW in keys:
            steer = -0.7
        if ord('q') in keys:
            break

        action = np.array([throttle, steer], dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(1 / 120)

        if terminated or truncated:
            print(f"  dist: {obs[3]:.2f} | {'GOAL!' if terminated else 'timeout'}")
            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true", help="Watch trained PPO model")
    parser.add_argument("--manual", action="store_true", help="Keyboard control")
    args = parser.parse_args()

    from robot_env import RobotEnv
    env = RobotEnv(render=True)

    try:
        if args.model:
            run_model(env)
        elif args.manual:
            run_manual(env)
        else:
            run_random(env)
    finally:
        env.close()
