from typing import Dict, Tuple
import argparse

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

import sys
sys.path.append('/home/prakyath/github/personal/test_walking_with_palm')

from utils.integration import AccIntegration


# Skiping default camera config

class WalkerStraight(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
            
        ],
    }
    def __init__(
        self, 
        model_path: str = "./models/quad.xml", 
        frame_skip: int = 4, 
        include_position: bool = True,
        include_velocity: bool = True,
        include_acc: bool = True,
        include_gyro: bool = True,
        forward_reward_weight: float = 0.99,
        health_reward: float = 0.99, 
        ctrl_cost_weight: float = 1e-3, 
        weight_smooth_phase_plane: float = 1e-3,
        healthy_state_range: Tuple[float, float]= (0.3, -0.3), 
        healthy_z_range: Tuple[float, float] = (0.03, 0.2),
        side_cost_weight: float = 1e-3,
        max_episode_steps: int = 1000,
        ) -> None:
        print(os.getcwd())

        utils.EzPickle.__init__(
        self, 
        model_path,
        frame_skip,
        include_position,
        include_velocity,
        include_acc,
        include_gyro,
        forward_reward_weight,
        health_reward,
        ctrl_cost_weight,
        weight_smooth_phase_plane,
        healthy_state_range,
        healthy_z_range,
        )

        # saving the parameters
        self._model_path = model_path
        self._frame_skip = frame_skip
        self._include_position = include_position
        self._include_velocity = include_velocity
        self._include_acc = include_acc
        self._include_gyro = include_gyro
        self._foward_reward_weight = forward_reward_weight
        self._health_reward = health_reward
        self._ctrl_cost_weight = ctrl_cost_weight
        self._weight_smooth_phase_plane = weight_smooth_phase_plane
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._side_cost_weight = side_cost_weight
        self._reset_noise_scale = 1e-3
        self.max_episode_steps = max_episode_steps


        # starting the mujoco env
        # TODO: Need to understand the camera position and how to change it
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip,
            observation_space=None,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        
        # start with ACC and GYRO and then if selected add the position and velocity
        obs_size = 0
        if self._include_acc:
            obs_size += 3 # hardcoding this but in the future it should be dynamic

        if self._include_gyro:
            obs_size += 3 # hardcoding this but in the future it should be dynamic

        #if self._include_position:
            #obs_size += self.data.qpos.size

        #if self._include_velocity:
            #obs_size += self.data.qvel.size



        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
                    )

        self.observation_struction = {
            "acc": 3,
            "gyro": 3,
            "position": self.data.qpos.size,
            "velocity": self.data.qvel.size
        }

        self.acc_to_pos = AccIntegration()
        self.prev_com_pos = self.get_body_com("torso").copy()
        self.i = 0

    def _get_obs(self) -> np.ndarray:
        obs = []
        if self._include_acc:
            obs += self.data.sensordata[:3].tolist()
        if self._include_gyro:
            obs += self.data.sensordata[3:6].tolist()
        if self._include_position:
            obs += self.data.qpos.tolist()
        if self._include_velocity:
            obs += self.data.qvel.tolist()
        return np.array(obs)

    def _calculate_position(self, acc: np.ndarray) -> np.ndarray:
        acc = acc.reshape(3,-1) # make sure it is 3xn
        acc = acc * 10e9
        acc[2,:] -= 9.81 * 10e9
        return self.acc_to_pos.predict(acc)

    def _get_reward(self) -> float:

        # forward reward
        com = self.get_body_com("torso")
        forward_reward = (com[0] - self.prev_com_pos[0])/self.dt
        forward_reward *= self._foward_reward_weight

        return forward_reward

    def _get_obs(self) -> np.ndarray:
        return self.data.sensordata

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.i = 0
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    @property
    def terminate(self) -> bool:
        com = self.get_body_com("torso")
        if com[2] < self._healthy_z_range[0]:
            return True
        if com[1] < self._healthy_state_range[1] and com[1] > self._healthy_state_range[0]:
            return True
        return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.i += 1
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self._get_reward()
        com = self.get_body_com("torso")
        cost = self._ctrl_cost_weight * np.sum(np.square(action))
        cost_sideway_stepping = self._side_cost_weight * np.square(com[1])
        reward = reward - cost_sideway_stepping - cost
        terminate = self.terminate
        if self.i > self.max_episode_steps:
            terminate = True
            reward += 1000

        info = {
            "reward_forward": reward,
            "reward_ctrl": -cost,
            "com": com}

        self.prev_com_pos = com.copy()

        return obs, reward, terminate, False, info



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()


    walk_env = WalkerStraight(model_path=args.model_path)
    walk_env.reset()
    for _ in range(100):
        action = walk_env.action_space.sample()
        obs, reward, terminate, _, info = walk_env.step(action)
        print(info["com"], terminate)
