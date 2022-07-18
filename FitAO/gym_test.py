try:
    from CompassEnv.CompassEnv import CompassEnv
except ModuleNotFoundError:
    print("\033[93m"+"Shesa modules not found. Try running '. ./env_var.sh' first!"+'\x1b[0m')
from OOMAOEnv.PyOOMAO import PyOOMAO
from SoapyEnv.SoapyEnv import SoapyEnv
import Tools.mat as tm
import Tools.control as ctr
import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm
# Example script for working with individual enviroments

param_file = "./Conf/sh_16x8.py"  #Path to Compass param file, . in front means relative import.

env = PyOOMAO()  #AO enviroment object
controller = ctr.control(skip_WFS=False)
env.set_params_file(param_file)   #sets env parameter file

controller.do_matrices(env, forceNew = False)

obs = env.reset(2)     #initialize sim and get current wfs slopes

#print("Sample: "+str(env.action_space.sample()))    #Random sample from action space

print("Running loop...")
for i in tqdm.tqdm(range(500)):

    #Rendering WFS phase and DM shape
    env.render()

    #Integrator controller
    action = controller.closed_loop(obs, gain = 0.6)

    #Applying command and after that moving atmos/updating WFS. Returns slopes as obs,
    #squared norm of residual voltages as reward, done as False (no current rule
    #for done), and empty info (can add any extra information we might want to use)
    obs, reward, done, info = env.step(action, showAtmos = True)

    #print(env.get_strehl())

    #time.sleep(1) #Possibility to slow down simulation steps, time given as sec

#reset enviroment
env.close()
