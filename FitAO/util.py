import torch
import numpy as np
from torch import nn
import gym
from gym import spaces
from gym.spaces import Box, Tuple
import tqdm
import Tools.mat as tm
import matplotlib.pyplot as plt


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bs = x.shape[0]
        return x.view(bs, -1)


class BaseNet(nn.Module):
    def __init__(self, n_prev_actions=1):
        super().__init__()

        n_input_channels = 2
        k_size = 3

        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Dynamics(nn.Module):
    def __init__(self, xvalid, yvalid):
        super().__init__()

        n_input_channels = 3
        k_size = 3
        self.xvalid = xvalid
        self.yvalid = yvalid

        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, k_size, padding=(k_size // 2)),
            nn.LeakyReLU(),
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            # nn.Tanh()
        )

    # TODO preprocess mask
    def forward(self, state, action):
        if action.ndim == 4:
            action_img = action
        else:
            action_img = torch.zeros(len(action), 1, 17, 17)
            action_img[..., self.xvalid, self.yvalid] = action.view(
                action.shape[0], 1, 220
            )
        feats = self.net(torch.cat([state, action_img], dim=1))
        mean = self.output(feats)
        mask = torch.zeros_like(mean, dtype=torch.bool)
        mask[:, :, self.xvalid, self.yvalid] = 1
        return mean * mask


class Policy(nn.Module):
    def __init__(self, env, xvalid, yvalid, stochastic=True):
        super().__init__()

        self.base_net = BaseNet()
        self.policy_actuators = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1), nn.Tanh())
        self.xvalid = torch.from_numpy(xvalid)
        self.yvalid = torch.from_numpy(yvalid)

    def forward(self, state, sigma=0.0):
        feats = self.base_net(state)
        actuators = self.policy_actuators(feats)

        actuators = actuators[:, :, self.xvalid, self.yvalid].squeeze(1)

        return actuators + sigma * torch.randn_like(actuators)  # .clamp(-1, 1)


class TorchWrapperNoDelay(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env = env

        # Calib DM 0
        self._env.set_noise(-1)
        self._env.set_V2S(None)
        self._env.set_S2V(None)
        self._env.reset()

        n_act0 = self._env.get_valid(0)
        n_act1 = self._env.get_valid(1)

        calibDmCommands = self._env.get_calibConst() * np.identity(n_act0)
        calibDmCommands = np.pad(calibDmCommands, ((0, 0), (0, n_act1)), "constant")

        D = np.zeros(
            (self._env.observation_space.sample().shape[0], calibDmCommands.shape[0])
        )
        print("Doing imat...")
        self._env.step(0 * calibDmCommands[0, :], showAtmos=False)
        for i in tqdm.tqdm(
            range(calibDmCommands.shape[0])
        ):  # tqdm used to create progress bar
            for j in range(2):  # To compensate 1 frame lag of wfs sensor
                slopes_push = self._env.step(calibDmCommands[i, :], showAtmos=False)[0]
            for j in range(2):
                slopes_pull = self._env.step(-1 * calibDmCommands[i, :], showAtmos=False)[
                    0
                ]
            D[:, i] = 0.5 * (np.asarray(slopes_push) - np.asarray(slopes_pull))
            # print(D[:,i])
            # env.render()

        # plt.plot(np.linalg.norm(D,axis=0))
        # plt.show()

        self.V2S_0 = D / self._env.get_calibConst()

        self._env.step(0 * calibDmCommands[0, :], showAtmos=False)

        self._env.reset_custom_params()

        # Calib DM 1
        self._env.set_noise(-1)
        self._env.set_V2S(None)
        self._env.set_S2V(None)
        self._env.reset()

        calibDmCommands = self._env.get_calibConst() * np.identity(n_act1)
        calibDmCommands = np.pad(calibDmCommands, ((0, 0), (n_act0, 0)), "constant")

        D = np.zeros(
            (self._env.observation_space.sample().shape[0], calibDmCommands.shape[0])
        )
        print("Doing imat...")
        self._env.step(0 * calibDmCommands[0, :], showAtmos=False)
        for i in tqdm.tqdm(
            range(calibDmCommands.shape[0])
        ):  # tqdm used to create progress bar
            for j in range(2):  # To compensate 1 frame lag of wfs sensor
                slopes_push = self._env.step(calibDmCommands[i, :], showAtmos=False)[0]
            for j in range(2):
                slopes_pull = self._env.step(-1 * calibDmCommands[i, :], showAtmos=False)[
                    0
                ]
            D[:, i] = 0.5 * (np.asarray(slopes_push) - np.asarray(slopes_pull))
            # print(D[:,i])
            # env.render()

        # plt.plot(np.linalg.norm(D,axis=0))
        # plt.show()

        self.V2S_1 = D / self._env.get_calibConst()

        self._env.step(0 * calibDmCommands[0, :], showAtmos=False)

        self._env.reset_custom_params()

        self.S2V_0 = tm.do_cmat(self._env, 0.05, self.V2S_0)
        self.S2V_1 = tm.do_cmat(self._env, 0.05, self.V2S_1)

        self.action_space = spaces.Box(
            -5,
            5,
            shape=(
                self._env.get_total(1),
                self._env.get_total(1),
            ),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1, self._env.get_total(1), self._env.get_total(1)),
            dtype=np.float32,
        )
        # self.normalizer = np.load('int_std.npy').mean()* 2.5/8
        self.normalizer = 1
        self.F = np.matmul(self.S2V_1, self.V2S_1)  # np.identity(self._env.get_valid(1))

        self.last_action = self._env.img_to_vec(np.zeros(self.action_space.shape))
        self.last_action_dm0 = np.zeros((self._env.get_valid(0)))
        # self.last_action_dm0 = 0 * np.matmul(self.S2V_0, self._env.get_slopes())
        self.last_action_dm1 = np.zeros((self._env.get_valid(1)))

        self.flip = 1
        self.m1_gain = 0.8

        self._env.reset()

    def step(self, action):
        action_to_numpy = action.numpy()
        action_vec = self._env.img_to_vec(action_to_numpy)
        a1 = action_vec
        # action_vec = np.clip(self._env.F @ (action_vec * self.normalizer + self.last_action),
        if self.m1_gain > 0:
            action_vec = np.clip(
                (action_vec * self.normalizer + self.last_action), -0.5, 0.5
            )
        else:
            action_vec = action_vec + self.last_action

        # print(action_vec.shape)
        # if self.flip:
        # print(self.last_action_dm0.shape)
        action_dm0 = self.last_action_dm0 - self.m1_gain * np.matmul(
            self.S2V_0, self._env.get_slopes()
        )
        # print(np.matmul(self.S2V_0, self._env.get_slopes()).shape)
        self.last_action_dm0 = action_dm0
        # self.flip = 0
        # else:
        # action_dm0 = self.last_action_dm0
        # self.flip = 1
        # print(action_dm0.shape)

        # action_dm1 = self.last_action_dm1 - 0.8 * np.matmul(self.S2V_1, self._env.get_slopes())
        # self.last_action_dm1 = action_dm1

        # plt.subplot(2,2,1)
        # plt.imshow(self._env.vec_to_img(action_dm1))
        # plt.subplot(2,2,2)
        # plt.imshow(self._env.vec_to_img(action_vec))
        # plt.subplot(2,2,3)
        # plt.imshow(self._env.vec_to_img(action_vec)/self._env.vec_to_img(action_dm1))
        # plt.colorbar()
        # plt.subplot(2,2,4)
        # dsf = np.arange(1,self._env.get_valid(1)+1)
        # asdf = self._env.vec_to_img(dsf)
        # asdf2 = self._env.img_to_vec(asdf)
        #
        # print(dsf)
        # print(asdf2)
        # plt.imshow(asdf)
        # plt.show()

        # asdf = self._env.vec_to_img(action_dm1)
        # asdf2 = self._env.img_to_vec(asdf)

        # print("Vec diff:",np.sum(np.abs(action_vec-action_dm1)))
        # print("Im diff:",np.sum(np.abs(self._env.vec_to_img(action_vec)-self._env.vec_to_img(action_dm1))))
        # print("Self diff:",np.sum(np.abs(asdf2-action_dm1)))

        # plt.show()

        total_act = np.concatenate((action_dm0, action_vec))

        # print(total_act.shape)
        # print(self._env.action_space.shape)

        next_obs, reward, done, info = self._env.step(total_act)
        reward = self._get_reward(self._env.get_slopes(), a1)
        # self._env.render()
        next_obs = self.process_obs(next_obs, action_dm0)

        self.last_action = action_vec

        # self.last_action = -1 * np.clip(action_vec * self.normalizer + self.last_action,-170,170)

        return (
            torch.tensor(next_obs, dtype=torch.float32),
            reward,
            done,
            [(k, torch.tensor(v, dtype=torch.float32)) for k, v in info.items()],
        )

    def reset(self, new_atmos=True, seed=-1):
        obs = self._env.reset(seed)
        obs = self.process_obs(obs, self.last_action_dm0)

        self.last_action = self._env.img_to_vec(np.zeros(self.action_space.shape))
        self.last_action_dm0 = np.zeros((self._env.get_valid(0)))

        return torch.tensor(obs, dtype=torch.float32)

    def _get_reward(self, slopes, action):
        res_volt1 = np.matmul(self.S2V_1, slopes)
        reward = -1 * (np.mean(np.power(res_volt1, 2)))

        # reward = self._env.get_strehl()

        return reward

    def reset_soft(self):
        obs = self._env.reset(soft=True)
        obs = self.process_obs(obs, self.last_action_dm0)

        return torch.tensor(obs, dtype=torch.float32)

    def process_obs(self, obs, action_dm0):
        obs = self._env.vec_to_img(
            -1 * np.matmul(self.S2V_1, obs)
        )  # -np.matmul(self.V2S_0,action_dm0)))
        obs = np.expand_dims(obs, 0)  # / self.normalizer

        return obs

    def set_m1_gain(self, m1_gain):
        self.m1_gain = m1_gain


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
