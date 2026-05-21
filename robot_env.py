import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from collections import deque


class RobotEnv(gym.Env):
    """PyBullet racecar environment with Ackermann steering and curriculum learning."""

    metadata = {"render_modes": ["human", "headless"], "render_fps": 60}

    def __init__(self, render=False):
        super().__init__()

        # Connect to PyBullet
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Action space: [throttle, steering_angle], both in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [dx, dy, yaw, dist, angle_to_goal, linear_vel, angular_vel]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Robot joint indices (from racecar URDF inspection)
        self.drive_joints = [2, 3, 5, 7]       # all four wheels
        self.steer_joints = [4, 6]              # front steering hinges

        # Physics parameters
        self.max_speed = 10.0       # wheel velocity (rad/s)
        self.max_steer = 0.785      # ~45 degrees in radians
        self.drive_force = 10.0
        self.steer_force = 10.0

        # Episode parameters
        self.max_steps = 500
        self.step_count = 0
        self.goal_reach_dist = 0.5

        # Curriculum learning
        self.goal_range = 1.5               # start with close goals
        self.max_goal_range = 5.0
        self.recent_outcomes = deque(maxlen=50)
        self.curriculum_threshold = 0.6     # expand when 60% success rate

        # Physics sub-stepping
        self.substeps = 4

        # State
        self.robot = None
        self.goal_position = None
        self.prev_dist = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)

        # Load ground and robot
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot = p.loadURDF(
            "racecar/racecar.urdf",
            basePosition=[0, 0, 0.1],
            physicsClientId=self.client
        )

        # Curriculum: expand goal range if doing well
        if len(self.recent_outcomes) == 50:
            success_rate = sum(self.recent_outcomes) / 50
            if success_rate >= self.curriculum_threshold and self.goal_range < self.max_goal_range:
                self.goal_range = min(self.goal_range + 0.5, self.max_goal_range)
                self.recent_outcomes.clear()

        # Spawn random goal
        angle = self.np_random.uniform(0, 2 * np.pi)
        dist = self.np_random.uniform(0.5, self.goal_range)
        self.goal_position = np.array([np.cos(angle) * dist, np.sin(angle) * dist])

        # Visual goal marker (only in GUI mode)
        if p.getConnectionInfo(self.client)["connectionMethod"] == p.GUI:
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.2, length=0.01,
                rgbaColor=[1, 0, 0, 0.7],
                physicsClientId=self.client
            )
            p.createMultiBody(
                baseVisualShapeIndex=visual,
                basePosition=[self.goal_position[0], self.goal_position[1], 0.01],
                physicsClientId=self.client
            )

        self.step_count = 0
        obs = self._get_obs()
        self.prev_dist = obs[3]

        return obs, {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        dx = self.goal_position[0] - pos[0]
        dy = self.goal_position[1] - pos[1]
        dist = np.sqrt(dx**2 + dy**2)

        # Angle from robot's heading to the goal
        goal_angle = np.arctan2(dy, dx)
        angle_to_goal = goal_angle - yaw
        # Normalize to [-pi, pi]
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi

        # Velocity projected onto robot's forward axis
        vel, ang_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client)
        forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
        linear_vel = vel[0] * forward_dir[0] + vel[1] * forward_dir[1]
        angular_vel = ang_vel[2]

        return np.array([dx, dy, yaw, dist, angle_to_goal, linear_vel, angular_vel], dtype=np.float32)

    def step(self, action):
        # Apply throttle to all drive wheels
        throttle = float(action[0]) * self.max_speed
        for joint in self.drive_joints:
            p.setJointMotorControl2(
                self.robot, joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=throttle,
                force=self.drive_force,
                physicsClientId=self.client
            )

        # Apply steering to front hinges
        steering = float(action[1]) * self.max_steer
        for joint in self.steer_joints:
            p.setJointMotorControl2(
                self.robot, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=steering,
                force=self.steer_force,
                physicsClientId=self.client
            )

        for _ in range(self.substeps):
            p.stepSimulation(physicsClientId=self.client)

        # Get new state
        obs = self._get_obs()
        dist = obs[3]
        angle_to_goal = obs[4]
        linear_vel = obs[5]

        # --- Reward ---
        # 1. Progress toward goal (biggest signal)
        progress_reward = (self.prev_dist - dist) * 5.0

        # 2. Heading alignment (scaled to matter)
        heading_reward = np.cos(angle_to_goal) * 0.3

        # 3. Forward motion bonus
        forward_bonus = 0.1 if linear_vel > 0.1 else -0.05

        # 4. Small time penalty to encourage efficiency
        time_penalty = -0.01

        reward = progress_reward + heading_reward + forward_bonus + time_penalty
        self.prev_dist = dist

        # Goal reached
        terminated = False
        if dist < self.goal_reach_dist:
            reward += 20.0
            terminated = True
            self.recent_outcomes.append(1)

        # Episode timeout
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        if truncated and not terminated:
            self.recent_outcomes.append(0)

        return obs, reward, terminated, truncated, {}

    def close(self):
        p.disconnect(physicsClientId=self.client)
