from CompassEnv.CompassEnv import CompassEnv
from OOMAOEnv.OOMAOEnv import OOMAOEnv
from SoapyEnv.SoapyEnv import SoapyEnv
import numpy as np
import time
import matplotlib.pyplot as plt
import Tools.mat as tm

param_file = "./Conf/sh_16x8.py"  # Path to Compass param file,

envs = []
render = False

n = 2000  # number of iterations per r0
g = 0.6  # gain used for demonstration integrator controller
r0_min = 3  # r0 limits to loop trough in cm
r0_max = 15

env1 = CompassEnv()  # Compass enviroment object
envs.append(env1)

env2 = OOMAOEnv()  # OOMAO enviroment object
envs.append(env2)

env3 = soapyEnv()  # Soapy enviroment object
envs.append(env3)

# Set range of values to go trough
x = np.arange(r0_min, r0_max + 1, 1)  # / 100

# Main loop trough all enviroments
sr_list = []
performance = []
for env in envs:
    sr_temp_list = []
    performance_tmp = []
    env.set_params_file(param_file)
    S2V = tm.do_cmat(env, 0.05)
    for r0 in x:
        env.set_gsmag(float(r0))
        obs = env.reset()
        last_action = 0 * np.matmul(S2V, obs)
        sr = 0  # used for counting avg Strehl ratio
        # time.sleep(5)
        start_time = time.time()  # Timing run

        for i in range(n):
            # Rendering WFS phase and DM shape
            if render:
                env.render()

            action = last_action - g * np.matmul(S2V, obs)

            last_action = action

            # Applying command and after that moving atmos/updating WFS. Returns slopes as obs,
            # squared norm of residual voltages as reward, done as False (no current rule
            # for done), and empty info (can add any extra information we might want to use)
            obs, reward, done, info = env.step(action)
            tmp = []
            sr += env.get_strehl()

        performance_tmp.append(((time.time() - start_time) / n))
        sr_temp_list.append(sr / n)

    sr_list.append(sr_temp_list)
    performance.append(np.mean(performance_tmp))
    env.close()
    # file.close()

# Reporting performance of enviroments
for i in range(0, len(envs)):
    print(
        envs[i].__class__.__name__ + " avg per step time: " + str(performance[i]) + " s"
    )

# Reporting results of avg Strehl ratios

for i in range(0, len(sr_list)):
    plt.plot(x, sr_list[i], label=envs[i].__class__.__name__)

plt.legend(loc="upper left")
plt.xlabel("GS mag")
plt.ylabel("Avg. Strehl ratio on " + str(n) + " steps")
# plt.xscale("log")

plt.show()
