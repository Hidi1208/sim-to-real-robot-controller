"""
Classical PID controller baseline for racecar goal navigation.
Steers toward the goal, throttles based on alignment.

Usage:
  python pid_controller.py              → run with GUI
  python pid_controller.py --episodes 20 → run multiple episodes headless
"""

import argparse
import time
import numpy as np
from robot_env import RobotEnv


class PIDController:
    """PID controller that outputs [throttle, steering] actions."""

    def __init__(self):
        # Steering PID gains
        self.kp_steer = 1.5
        self.ki_steer = 0.0
        self.kd_steer = 0.5

        # Throttle gains
        self.base_throttle = 0.6
        self.alignment_scale = 0.4   # reduce throttle when misaligned

        # PID state
        self.prev_angle_error = 0.0
        self.integral_error = 0.0

    def reset(self):
        self.prev_angle_error = 0.0
        self.integral_error = 0.0

    def act(self, obs):
        """Given observation, return [throttle, steering] in [-1, 1]."""
        angle_to_goal = obs[4]   # already normalized to [-pi, pi]
        dist = obs[3]

        # --- Steering (PID on angle_to_goal) ---
        error = angle_to_goal
        self.integral_error += error
        self.integral_error = np.clip(self.integral_error, -5.0, 5.0)
        derivative = error - self.prev_angle_error
        self.prev_angle_error = error

        steer = (self.kp_steer * error
                 + self.ki_steer * self.integral_error
                 + self.kd_steer * derivative)
        steer = np.clip(steer, -1.0, 1.0)

        # --- Throttle (based on alignment) ---
        alignment = np.cos(angle_to_goal)  # 1.0 = facing goal, -1.0 = facing away

        if alignment > 0.3:
            # Reasonably aligned — drive forward, faster when well aligned
            throttle = 0.7 + self.alignment_scale * alignment
        elif alignment > -0.3:
            # Sideways — slow forward while turning
            throttle = 0.3
        else:
            # Goal is behind — still go forward and steer around
            # (reversing a car toward a goal is harder than doing a wide turn)
            throttle = 0.25

        # Slow down when very close to avoid overshooting
        if dist < 0.6:
            throttle *= 0.7

        throttle = np.clip(throttle, -1.0, 1.0)

        return np.array([throttle, steer], dtype=np.float32)


def run_pid(episodes=5, render=True):
    env = RobotEnv(render=render)
    controller = PIDController()

    results = []
    for ep in range(episodes):
        obs, _ = env.reset()
        controller.reset()
        total_reward = 0

        for step in range(500):
            action = controller.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if render:
                time.sleep(1 / 60)
            if terminated or truncated:
                break

        success = terminated
        results.append({
            "episode": ep + 1,
            "success": success,
            "steps": step + 1,
            "final_dist": obs[3],
            "reward": total_reward
        })

        status = "GOAL!" if success else f"timeout (dist: {obs[3]:.2f})"
        print(f"  Episode {ep+1:3d} | {status:25s} | steps: {step+1:3d} | reward: {total_reward:.1f}")

    # Summary
    successes = sum(r["success"] for r in results)
    print(f"\n  PID Results: {successes}/{episodes} goals reached "
          f"({100*successes/episodes:.0f}% success rate)")

    env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    run_pid(episodes=args.episodes, render=not args.headless)
