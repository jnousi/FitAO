envs = []

try:
    from CompassEnv.CompassEnv import CompassEnv

    envs.append(CompassEnv())
except ModuleNotFoundError:
    print(
        "\033[93m"
        + "Shesa modules not found. Try running '. ./env_var.sh' first!"
        + "\x1b[0m"
    )

try:
    from OOMAOEnv.PyOOMAO import PyOOMAO

    envs.append(PyOOMAO())
except ModuleNotFoundError:
    print("\033[93m" + "PyOOMAO not found." + "\x1b[0m")

try:
    from SoapyEnv.SoapyEnv import SoapyEnv

    envs.append(SoapyEnv())
except ModuleNotFoundError:
    print("\033[93m" + "Soapy not found." + "\x1b[0m")

import Tools.mat as tm
import Tools.control as ctr
import numpy
import time
import tqdm


param_files = ["./Conf/sh_16x8.py", "./Conf/sh_16x8_lgs.py", "./Conf/pyrhr_16x16.py"]

controller = ctr.Control()

for env in envs:
    for param_file in param_files:
        valid = env.set_params_file(param_file)

        if valid:
            controller.do_matrices(env, forceNew=True)
            obs = env.reset(2)

            for i in tqdm.tqdm(range(100)):

                # Rendering WFS phase and DM shape
                env.render()

                # Integrator controller
                action = controller.closed_loop(obs, gain=0.6)

                # Applying command and after that moving atmos/updating WFS. Returns slopes as obs,
                # squared norm of residual voltages as reward, done as False (no current rule
                # for done), and empty info (can add any extra information we might want to use)
                obs, reward, done, info = env.step(action)

                # print(env.get_strehl())

            controller.reset()

    # Clean enviroment of after use
    env.close()
