import matlab.engine
import gym
from gym import error, spaces
import numpy as np
import matlab
import matplotlib.pyplot as plt
import time
import os
import io
import sys
from Tools.ParamParser import *


class PyOOMAO(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    # --------------------------Core gym funtions--------------------------------

    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        if not os.path.isfile(os.getcwd() + "/env.m"):
            try:
                self.eng.cd(
                    os.getcwd() + "/OOMAOEnv"
                )  # Change matlab folder to OOMAO env folder
            except:
                print("PyOOMAO files not found!")

        self.env = self.eng.env(nargout=1, stdout=io.StringIO())

        self.action_buffer = []
        self.done = False
        self.param_file = ""
        self.delay = 1
        self.S2V = None
        self.V2S = None
        self.F = 1
        self.pmat = None
        self.infmat = None
        self.calibConst = 1
        self.name = "OOMAO"

    def step(self, action, showAtmos=True):
        # Single AO step. Action defines DM shape and function returns WFS
        # slopes, reward (-1*norm of slopes), done as false (no current condition)
        # and (currently empty) info dictionary, where one could store useful
        # data about the simulation
        action = action * (self.get_wavelen() / 1e6)
        if showAtmos:
            if len(self.action_buffer) < self.delay:
                self.action_buffer.append(action)
                action = 0 * action
            else:
                self.action_buffer.append(action)
                action = self.action_buffer.pop(0)

        action = self.list2matlab(action)
        self.eng.applyControl(self.env, action, nargout=0, stdout=io.StringIO())
        slopes, done, info = self.eng.step(
            self.env, showAtmos, nargout=3, stdout=io.StringIO()
        )
        slopes = self.list2python(slopes) / (self.get_wavelen() / 1e6)
        info = {}

        return slopes, self._get_reward(slopes), bool(done), info

    def render(self, mode="rgb_array"):
        dm, ngs, atmos, img, wfs_img = self.eng.render(
            self.env, nargout=5, stdout=io.StringIO()
        )

        try:
            plt.clf()

            plt.subplot(1, 5, 1)
            plt.imshow(np.asarray(ngs))
            plt.title("WFS residual phase")

            plt.subplot(1, 5, 2)
            plt.imshow(np.asarray(dm))
            plt.title("DM shape")

            plt.subplot(1, 5, 3)
            plt.imshow(np.asarray(atmos))
            plt.title("Atmos")

            plt.subplot(1, 5, 4)
            plt.imshow(np.asarray(img))
            plt.title("Image")

            plt.subplot(1, 5, 5)
            plt.imshow(np.asarray(wfs_img))
            plt.title("Raw WFS image")

            plt.setp(
                plt.gcf().get_axes(), xticks=[], yticks=[]
            )  # Remove axis ticks for readability
            plt.suptitle("OOMAO", fontsize=14)

            plt.draw()
            plt.pause(1e-3)  # Pause for plotting to work
        except KeyboardInterrupt:
            print("Ok ok, quitting")
            sys.exit(1)
        except:
            print("Cant render")

    def reset(self, seed=-1):  # -1 random seed (default), > 0 set seed
        self.eng.clr(self.env, nargout=0)
        self.set_params(seed)
        self.action_buffer = []
        slopes = self.list2python(
            self.eng.rst(self.env, nargout=1, stdout=io.StringIO())
        ) / (self.get_wavelen() / 1000000)

        return slopes

    def close(self):
        self.eng.quit()
        plt.close()
        del self.env, self.eng

    # ---------------- Simulation initialisation functions ---------------------

    def set_params_file(self, param_file, do_mat=False):
        # Set the Compass parameter file used by simulation and load it the first
        # time. Returns boolean if the config is valid.
        if param_file != self.param_file:
            self.param_file = param_file
            self.S2V = None
            self.V2S = None
            valid_config = self._param_file_parser()
            self.set_params()

        return valid_config

        # if do_mat:
        #     S2V = self.eng.get_S2V(self.env, nargout=1,stdout=io.StringIO())
        #     return np.asarray(S2V)

    def set_params(self, seed=-1):  # -1 random seed (default), > 0 set seeds
        print("Setting up sim params...")
        self.eng.set_tel_params(
            self.env,
            self.tel_params["diam"],
            self.tel_params["obstructionRatio"],
            self.tel_params["samplingTime"],
            self.tel_params["pupdiam"],
            nargout=0,
            stdout=io.StringIO(),
        )
        if ("alt" in self.gs_params) and (self.gs_params["alt"] > 0):  # LGS
            self.eng.set_lgs_params(
                self.env,
                self.gs_params["magnitude"],
                self.gs_params["wavelength"],
                self.gs_params["alt"],
                self.gs_params["lltx"],
                self.gs_params["llty"],
                self.gs_params["xpos"],
                self.gs_params["ypos"],
                nargout=0,
                stdout=io.StringIO(),
            )
            print("LGS")
        else:  # NGS
            self.eng.set_ngs_params(
                self.env,
                self.gs_params["magnitude"],
                self.gs_params["wavelength"],
                nargout=0,
                stdout=io.StringIO(),
            )
            print("NGS")
        self.eng.set_science_params(
            self.env,
            self.science_params["magnitude"],
            self.science_params["wavelength"],
            nargout=0,
            stdout=io.StringIO(),
        )
        if self.wfs_params["type"] == "sh":
            self.eng.set_sh_params(
                self.env,
                self.wfs_params["nxsub"],
                self.wfs_params["npix"],
                self.wfs_params["fracsub"],
                self.wfs_params["noise"],
                nargout=0,
                stdout=io.StringIO(),
            )
        else:
            self.eng.set_pyr_params(
                self.env,
                self.wfs_params["nxsub"],
                self.wfs_params["fssize"],
                self.wfs_params["fracsub"],
                self.wfs_params["noise"],
                self.wfs_params["pyr_amp"],
                self.wfs_params["pyr_pup_sep"],
                self.wfs_params["pyr_npts"],
                self.tel_params["pupdiam"],
                nargout=0,
                stdout=io.StringIO(),
            )
        self.eng.set_dm_params(
            self.env,
            self.wfs_params["nxsub"],
            self.tel_params["pupdiam"],
            self.dm_params["coupling"],
            nargout=0,
            stdout=io.StringIO(),
        )
        self.eng.set_atmos_params(
            self.env,
            self.atmos_params["windSpeed"],
            self.atmos_params["windDirection"],
            self.atmos_params["r0"],
            self.atmos_params["layeredL0"],
            self.atmos_params["fractionalR0"],
            self.atmos_params["altitude"],
            seed,
            nargout=0,
            stdout=io.StringIO(),
        )
        self.eng.set_cam(self.env, nargout=0, stdout=io.StringIO())
        self._def_spaces()
        # Finishes initilizing
        self.step(0 * self.action_space.sample(), showAtmos=False)

    def _param_file_parser(self):
        # Parse Compass parameter file and convert units to match OOMAO.
        # Returns boolean if config is valid
        print("Parsing param file...")

        (
            self.tel_params,
            self.atmos_params,
            self.gs_params,
            self.science_params,
            self.wfs_params,
            self.dm_params,
            self.delay,
        ) = parse_file(self.param_file, "matlab")

        # Unit conversion
        if "npix" in self.wfs_params:
            self.wfs_params["npix"] = self.wfs_params["npix"] * self.wfs_params["nxsub"]
        self.atmos_params["r0"] = (self.atmos_params["r0"] / (500 ** (6 / 5))) * (
            550 ** (6 / 5)
        )  # r0 from 500nm -> 550nm used by OOMAO

        self.wavelength_warning("Guidestar", self.gs_params["wavelength"])
        self.wavelength_warning("Science", self.science_params["wavelength"])

        # Update calibConst
        self.calibConst = self.gs_params["calibConst"]

        return True

    def _def_spaces(self):
        # Defining the shapes and limits of observations and actions for gym
        action, obs = self.eng.get_spaces(self.env, nargout=2, stdout=io.StringIO())
        action = np.asarray(action)
        action = sum(action.flat)
        obs = np.array(obs)

        action_h = 1000 * np.ones(action)  # .shape[0])
        action_l = -1 * action_h
        self.action_space = spaces.Box(low=action_l, high=action_h)

        obs_h = 1 * np.ones(obs.shape[0])
        obs_l = -1 * obs_h
        self.observation_space = spaces.Box(low=obs_l, high=obs_h)

        self.xvalid, self.yvalid = np.nonzero(self.get_valid())

        self.n_actu = int(self.wfs_params["nxsub"] + 1)

    # -------------------------- Updating params outside the sim ---------------

    def reset_custom_params(self):
        self._param_file_parser()

    def set_r0(self, r0):
        self.atmos_params["r0"] = float(r0)

    def set_noise(self, noise):
        self.wfs_params["noise"] = int(noise)

    def set_gsmag(self, mag):
        self.gs_params["magnitude"] = float(mag)

    def set_S2V(self, S2V):
        self.S2V = S2V

    def get_S2V(self):
        return self.S2V

    def set_V2S(self, V2S):
        self.V2S = V2S

    def get_V2S(self):
        return self.V2S

    def set_pmat(self, pmat):
        self.pmat = pmat

    def get_pmat(self):
        return self.pmat

    def set_infmat(self, infmat):
        self.infmat = infmat

    def get_infmat(self):
        return self.infmat

    def set_F(self, F):
        self.F = F

    def get_F(self):
        return self.F

    def get_calibConst(self):
        return self.calibConst

    # -------------------------- Misc. functions -------------------------------

    def get_strehl(self):
        return self.eng.get_strehl(self.env, nargout=1, stdout=io.StringIO())

    def get_valid(self):
        return self.eng.get_valid(self.env, nargout=1, stdout=io.StringIO())

    def get_wavelen(self):
        return self.gs_params["wavelength"]

    def get_imat(self):
        return self.eng.get_imat(self.env)

    def _get_reward(self, slopes, type="volt"):
        if self.S2V is not None and type is not "sh":
            res_volt = np.matmul(self.S2V, slopes)
            reward = -1 * np.linalg.norm(res_volt)
        else:
            reward = self.get_strehl()

        return reward

    def get_dm_shape(self, dm_idx=0):
        return np.asarray(
            self.eng.get_dm_shape(self.env, nargout=1, stdout=io.StringIO())
        )

    def get_phase_screen(self, target=0):
        return np.asarray(
            self.eng.get_phase_screen(self.env, nargout=1, stdout=io.StringIO())
        )

    def get_pupil(self):
        return np.asarray(self.eng.get_pupil(self.env, nargout=1, stdout=io.StringIO()))

    def get_KL_modes2volt(self, n_modes=None):
        mod2volt = np.asarray(
            self.eng.get_KL_modes2volt(self.env, nargout=1, stdout=io.StringIO())
        )

        if n_modes is not None:
            mod2volt = mod2volt[:, 0:n_modes]

        return mod2volt

    def _get_photonpsubap(self):
        total_photons = self.eng.get_photonpsubap(
            self.env, nargout=1, stdout=io.StringIO()
        )
        print(total_photons)

    # Functions to transform lists to and from matlab double list to numpy array
    def list2matlab(self, list):
        return matlab.double(list.reshape(len(list), 1).tolist())

    def list2python(self, list):
        return np.asarray([float(i[0]) for i in list])

    def wavelength_warning(self, txt, wavelength):
        # Used to warn the user if the chosen wavelength is not supported by
        # OOMAO.
        supported_wavelength = [
            0.36,
            0.44,
            0.55,
            0.64,
            0.79,
            1.22,
            1.65,
            2.18,
            3.55,
            4.77,
        ]

        closest = min(supported_wavelength, key=lambda x: abs(x - wavelength))

        if closest == wavelength:
            pass
        else:
            print(
                "\033[93m"
                + txt
                + " set wavelength of "
                + str(wavelength)
                + " micrometers is not supported by OOMAO!"
            )
            print(
                "Using the closest supperted wavelength of "
                + str(closest)
                + " for OOMAO. \n (Consider changing the simulation parameter file to match)"
                + "\x1b[0m"
            )
            time.sleep(5)

    # ---------------------- RL funtions ----------------------------------------

    def vec_to_img(self, action_vec):
        valid_actus = np.zeros_like(self.get_valid())
        valid_actus[self.xvalid, self.yvalid] = action_vec

        return valid_actus

    def img_to_vec(self, action):
        assert len(action.shape) == 2

        return action[self.xvalid, self.yvalid]

    def sample_binary_noise(self, sigma):
        noise = self.F @ (sigma * np.sign(np.random.normal(0, 1, size=(self.n_actions,))))
        return self.vec_to_img(noise)
