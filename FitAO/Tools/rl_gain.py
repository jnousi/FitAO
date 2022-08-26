import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np
from FitAO.Tools.mat import do_cmat

# Simple showcase how the gym enviroments can be used to generate a general
# class used for RL problems. __init__ env input can be any of the AO gym
# enviroments
class AoGain:
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, env, param_file):
        self.n_obs = 20  # history of measurements to use

        self.env = env
        self.env.__init__()
        self.env.set_params_file(param_file)
        self.S2V = do_cmat(env, 0.05)
        self._def_spaces()
        self.obs = np.zeros(self.n_obs)
        self.gain = 0

    def step(self, action, showAtmos=True):
        self.steps = self.steps + 1

        self.gain = np.clip(self.gain + action, 0, 1)

        self.action = self.last_action - self.gain * np.matmul(self.S2V, self.slopes)
        self.last_action = self.action

        # Add used gain to observations
        self.obs = np.roll(self.obs, 1)
        self.obs[0] = self.gain
        self.slopes, self.reward, self.done, self.info = self.env.step(self.action)

        # Add measured Strehl to observations
        obs = self._get_reward(action)
        self.obs = np.roll(self.obs, 1)
        self.obs[0] = obs

        # Limit each episode to 5000 steps
        if self.steps == 5000:
            self.done = True

        return self.obs, self._get_reward(action), self.done, self.info

    def reset(self, seed=-1):
        self.steps = 0

        self.obs = np.zeros(self.n_obs)

        self.slopes = self.env.reset(seed=seed)
        self.last_action = 0 * np.matmul(self.S2V, self.slopes)

        obs = np.sum(np.abs(np.matmul(self.S2V, self.slopes)))
        self.obs = np.roll(self.obs, 1)
        self.obs[0] = self._get_reward(0)

        self.gain = 0

        return self.obs

    def render(self, mode="rgb_array"):
        self.env.render()

    def _def_spaces(self):  # Defining the sizes and min/max of action/observation spaces
        self.action_space = spaces.Box(low=np.asarray([-0.5]), high=np.asarray([0.5]))
        self.observation_space = spaces.Box(
            low=np.asarray(np.zeros(self.n_obs)), high=np.asarray(np.ones(self.n_obs))
        )

    def _get_reward(self, action):
        return np.exp(self.env.get_strehl()) - np.abs(action)

    def close(self):
        self.env.close()
