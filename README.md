# FitAO

FitAO is a unified Python interface for multiple AO simulators. Currently it supports [Compass](https://anr-compass.github.io/compass/), [OOMAO](https://github.com/rconan/OOMAO) and [Soapy](https://github.com/AOtools/soapy). The interface follows [OpenAI Gyms](https://github.com/openai/gym) model for [enviroments](https://gym.openai.com/docs/#environments).

## TOC
  * [Installation](#installation)
    + [Compass](#compass)
    + [OOMAO](#oomao)
    + [Soapy](#soapy)
  * [Usage](#usage)
    + [Basic usage](#basic-usage)
    + [Multiple environments](#multiple-environments)
    + [Parameters](#parameters)
    + [Editable parameters](#editable-parameters)
  * [Interface](#interface)
    + [step](#step)
    + [reset](#reset)
    + [close](#close)
    + [set_params_file](#set_params_file)
    + [set_params](#set_params)
    + [Misc. functions](#Misc-functions)
  * [Good to know](#good-to-know)
    + [Compass](#compass)
    + [OOMAO](#oomao)
    + [Soapy](#soapy)


## Installation

### Compass

- Follow the [installation instructions](https://anr-compass.github.io/compass/install.html) of Compass.
- Compass v5.0.0 is the currently supported release (meaning CUDA 11.0 needs to be used)
- Always before first run add these enviroment variables (can be added by running `. ./env_var.sh` command on the terminal):
  - export SHESHA_ROOT=$HOME/shesha
  - export PYTHONPATH=$SHESHA_ROOT:$PYTHONPATH
  - export PYTHONDONTWRITEBYTECODE=1

### OOMAO

- Install [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
- Copy and paste oomao-master folder from [OOMAOs GitHub](https://github.com/rconan/OOMAO) to OOMAOEnv folder

### Soapy

- Install the [required libraries](https://github.com/AOtools/soapy) for Soapy
  - GUI libraries should not be necessary
- Copy the Soapy folder to SoapyEnv folder
- Replace the simulator.py found on soapy/soapy/sim with the one found on SoapyEnv folder
- Replace the wfs.py found on soapy/soapy/wfs with the one found on SoapyEnv folder

## Usage

### Basic usage

Start by importing your enviroment(s) of choice:

- For Compass use `from CompassEnv.compass_env import CompassEnv`
- For OOMAO use `from OOMAOEnv.oomao_gym import PyOOMAO`
- For Soapy use `from SoapyEnv.soapyEnv import soapyEnv`

Next it is recommended to define the path to the parameter file to be used. The Conf folder includes some examples to use, from which sh_16x8.py is the simplest. So:
`param_file = "/Conf/sh_16x8.py"`.

Now we can create our enviroment and set its parameter file. To create the enviroment you simply need to call the imported objects. For example: `env = CompassEnv()`. `PyOOMAO()` and `soapyEnv()` may also be used.

To set the enviroments parameters, we can call the function to do that `env.set_params_file(param_file)`. It returns a boolean that tells if the requested parameter file is supported (e.g. Soapy doesn't support a pyramid WFS). 

Now to initialize the enviroment we need to call `reset` function as such: `obs = env.reset()`. Note that we saved the return to a variable called `obs` as the reset function returns the first measured slopes.

To see the current visual representation of the current situation, we can call `env.render()` to create a plot of the current WFS phase, DM shape and atmospheric screen.

Now we could use some control algorithm to generate appropriate control commands for the DM. `gym_test.py` has an example integrator control implementation. It uses controllers found in `Tools/control.py`. First we create the controller class by `controller = ctr.control(skip_WFS=True/False)` where `ctr` is the imported file and `skip_WFS` allows us to directly project the wavefront to the DM without using WFS measurements. Then we call `controller.do_matrices(env,forceNew = False/True)` to create/load the required matrices for the controller (uses tools found in `Tools/mat.py`). Then to compute the command(s) we can simply call `action = controller.closed_loop(obs,gain = 0.6)`. Other option would be to replace `closed_loop` with `pseudo_open_loop` as it is the other implemented controller. Both use the `gain` parameter.

This can be then used to take a step in our simulation. By calling ` obs, reward, done, info = env.step(action)`, we send the action to an action buffer (with a delay of 1 by default), move the atmosphere and calculate the new slopes and reward. The slopes are then saved to `obs` and reward to `reward`. `done` returns always `false`, but it could be used to set a fail/pass condition for RL-training. `info` currently returns an empty dictionary, but it could be used to return interesting information from the simulation.

Now we can loop this to simulate the AO system. When we are finished, `env.close()` should be called to clear the simulation from memory. This can be really important when running big systems with multiple simulators, so that when we move from one simulator to the next one there is enough free memory to use.

The whole script could look something like this:

```python
from CompassEnv.compass_env import CompassEnv
import Tools.control as ctr

param_file = "/Conf/sh_16x8.py"
env = CompassEnv()
controller = ctr.control(skip_WFS=False)

valid = env.set_params_file(param_file)

if valid:
    obs = env.reset()
    controller.do_matrices(env,forceNew = False)

    for step in range(1000):
        env.render()
        action = controller.closed_loop(obs,gain = 0.6)
        obs, reward, done, info = env.step(action)

env.close()
```

### Multiple environments

It is recommended to save all environment objects to a list. Then you can simply loop trough the list and inside the loop do your simulations. `controller.reset()` is called to clear all past actions/observations from the memory.

```python
from CompassEnv.compass_env import CompassEnv
from OOMAOEnv.oomao_gym import PyOOMAO
from SoapyEnv.soapyEnv import soapyEnv
import Tools.control as ctr

param_file = "/Conf/sh_16x8.py"
envs = []
envs.append(CompassEnv())
envs.append(PyOOMAO())
envs.append(soapyEnv())

controller = ctr.control(skip_WFS=False)

for env in envs:
    valid = env.set_params_file(param_file,forceNew = False)
    if valid:
    	obs = env.reset()
    	controller.do_matrices(env)

    	for step in range(1000):
        	env.render()
    
        	action = controller.closed_loop(obs,gain = 0.6)
        	obs, reward, done, info = env.step(action)

    	controller.reset()

    	env.close()
```


### Parameters

FitAO uses native Compass parameter files, from which it parses information for other simulations. Currently FitAO supports only SCAO systems with Shack-Hartmann WFS. Parameters which you can safely edit include telescope size; obstrution ratio; all atmospheric parameters; WFS lenslet, pixel and noise amount.

### Editable parameters

Some special parameters can be adjusted outside the parameter files. Currently adjusting for Fried parameter @500nm (r0), wfs noise (noise) and guide star magnitude (gs\_mag) are implemented, but extending this support should be almost trivial. These can be used by `env.set_"param"("param value")` (for example `env.set_r0(0.15)`) and resetting the enviroment after to update the atmosphere accordingly.

## Interface

Since FitAO is based on OpenAi:s gym interface, you can see the [offical documentation](https://gym.openai.com/docs/#environments) for required functions. Here is a bit more in-depth explanation of how they are used here (to help adding new simulators/get a better understanding what functions should do)

### step

`step(self, action, showAtmos = True):`

The step function is used to move the simulation forward and interact with the telescope. `action` is the command given to the DM, which is delayed by a buffer inside this function and scaled such as with command of 1 creation of interaction matrix is reasonable. `showAtmos` controls whether the atmosphere is taken into account when looking at received the wavefronts. It is important we can turn the atmosphere off for calibrating the system. 

The step starts by setting the DM shape based on the delayed command. Then the the wavefront is propagated trough the possible atmosphere and DM. Note that the returned slopes from this propagation  step should always be one time-step behind the actual slopes to simulate the delays present in the system and scaled to match the commands.

The step function returns the scaled slopes, a reward, a done boolean and a dictionary of other important info (currently empty)

### reset

`reset(self, seed = -1):`

`seed` is used to set a seed for the atmosphere generation. This way it is possible to use the same atmosphere multiple times. Negative numbers denote random seed and positive are set seeds.

Reset should set DM commands to zero, reset atmosphere with current seed and return the scaled slopes after propagation. 

### render

`render(self, mode='rgb_array'):`

`mode` is currently unused, but it could be used to set some predefined modes of rendering (eg. only rendering certain things).

Render is used to visualize the current state in the system. Currently it plots WFS residual wavefront, the DM shape, combined atmosphere, target image, and raw WFS image. 

### close

`close(self):`

Close is used to clean up the simulation once it is no longer used.

### set_params_file

`set_params_file(self,param_file):`

`param_file` path to the parameter file to be used.

Used to tell the simulators what parameters to used. Uses Compass parameter files. Examples can be found under `Conf` folder. Parameters can be extracted from the parameter file by using `ParamParser.parse_file(file_path)` from `Tools` folder.  Also calls `set_params()` to set the parameters. Returns boolean if the config is valid (i.e. the simulator supports the requested parameter file).

### set_params

`set_params(self,seed=None):`

`seed` is the seed used for atmosphere generation.

First sets the parameters based on the parameter file. After that changes any parameters that might have been edited in the script (`seed`, `r0`, `noise`,...) and then calls for initialization of the simulation and defining of spaces. Spaces are the the sizes of DM commands and slope measurements. These are used by reinforcement learning and are set by `self.action_space = spaces.Box(low=action_l,high=action_h)` where `action_l` is a vector of minimum values for DM commands and `action_h` is a vector of maximum values for DM command. Observation spaces are done by setting `self.observation_space` similiarly.


### Misc functions

`reset_custom_params(self)`, clear all parameters set on script.

`set_"param"(self,"param")` set parameter from the script (`r0`,`noise`,`gsmag`)

`get_strehl(self)` return Strehl value

`get_wavelen(self)` returns GS wavelength

`get_valid(self)` returns amount of valid actuators on DM, used for interaction matrix generation


## Good to know

### Compass
- Nothing so far.

### OOMAO
- Direct projection to the DM (`skip_WFS` parameter on controller) can be really sensitive to gain. For closed loop integrator gain of around 0.3 seems to work ok.
- Laser guide starts do not seem to exhibit elongation, but are implemented as documentation shows.
- `get_KL_modes2volt()` basic idea written, but OOMAO has trouble running `KLBasisDecomposition` script.

### Soapy
- Needs to recreate internal interaction matrix with every reset, which can be slow.
- Fft oversampling for WFS matched to Compass, making the simulation much slower at a cost of producing more similar results (can be edited in `_param_file_gen()` function inside `SoapyEnv.py`).
- No support for laser guide stars (seems like broken implementation in latest release) and pyramid WFS.



