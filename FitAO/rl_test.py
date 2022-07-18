import gym
from CompassEnv.CompassEnv import CompassEnv
from OOMAOEnv.PyOOMAO import PyOOMAO
from SoapyEnv.SoapyEnv import SoapyEnv
from Tools.rl_gain import AoGain
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import seaborn

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

param_file = "./Conf/sh_16x8.py"

# RL env used to learn adaptive gain, you can replace the CompassEnv with both
# soapyEnv and PyOOMAO
env = AoGain(CompassEnv(),param_file)

# ----------------- Train model ------------------------------------------------

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard_gain/")
model.learn(total_timesteps=int(3e5))
model.save("ppo2_AO")

del model # remove to demonstrate saving and loading

# ----------------- Load and test model ----------------------------------------

model = PPO2.load("ppo2_AO")

# Test trained agent
# rew = 0
# obs = env.reset(1234)
# for i in range(100):
#     action, _states = model.predict(obs)
#     print("Gain: "+str(action))
#     print("Strehl: "+str(env._get_reward(0)))
#     obs, rewards, dones, info = env.step(action)
#     rew = rew + rewards
#     env.render()
#
# print("Avg Strehl: "+str(rew/4000))

# -------- Parameters for comparison ------------------------------------------

n = 500 #frames to compare

# ---------- Optimaze static gain ----------------------------------------------
# opt_gain = 0
# max_rew = 0
# for j in range(5,101,5):
#     env.reset(-1,baseline_mag)
#     temp_rew = 0
#     env.step(j/100)
#     for a in range(n):
#         obs, rewards, dones, info = env.step(0)
#         temp_rew = rewards + temp_rew
#
#     if temp_rew > max_rew:
#         max_rew = temp_rew
#         opt_gain = j/100

opt_gain = 0.45

# -------- Test the RL against optimized static gain ---------------------------

gains = []

cur_gain = 0

seed = np.random.randint(1,1000000)

tmp_s_str = []

#Static gain
env.reset(seed)
static_str = 0
obs, rewards, dones, info = env.step(opt_gain)
tmp_s_str.append(env.env.get_strehl())
for o in range(n):
    obs, rewards, dones, info = env.step(0)
    tmp_s_str.append(env.env.get_strehl())
    if o == 40:
        env.env.render()
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        plt.savefig("fitao_render.png", dpi=500,bbox_inches='tight')
        plt.close()


tmp_a_str = []

#Adaptive gain
obs = env.reset(seed)
adpt_str = 0
for o in range(n):
    action, _states = model.predict(obs)
    cur_gain = np.clip(cur_gain + action,0,1)
    gains.append(cur_gain)
    obs, rewards, dones, info = env.step(action)
    tmp_a_str.append(env.env.get_strehl())

# ---------------  Plot the results --------------------------------------------

plt.clf()
plt.axhline(np.mean(tmp_s_str[40:]), label="Average static gain", color="r", linestyle="--")
plt.plot(tmp_a_str, label="Adaptive gain")

plt.legend(loc="lower right")
plt.xlabel('Frame number')
plt.ylabel("Strehl ratio")

plt.savefig("fitao_strehl.png", dpi=1200)

plt.clf()
seaborn.histplot(np.asarray(gains),stat="probability",legend=False)
plt.xlabel("Chosen gain")
plt.ylabel("Probability")
plt.savefig("fitao_gain.png", dpi=1200)

print(opt_gain)
