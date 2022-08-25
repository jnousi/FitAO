import gym
from gym import spaces

try:
    from .soapy2.soapy.simulation import Sim
except ImportError:
    from soapy2.soapy.simulation import Sim

import matplotlib.pyplot as plt
import numpy as np
import time
import os
from FitAO.Tools.ParamParser import parse_file


class SoapyEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    # --------------------------Core gym funtions--------------------------------

    def __init__(self):
        self.action_buffer = []  # used to add delay to commands
        self.done = False
        self.param_file = ""
        self.custom_params = {}
        self.delay = 1
        self.S2V = None
        self.V2S = None
        self.pmat = None
        self.infmat = None
        self.name = "Soapy"

        self.sim = Sim()

    def step(self, action, showAtmos=True):
        # Single AO step. Action defines DM shape and function returns WFS
        # slopes, reward (-1*norm of slopes), done as false (no current condition)
        # and (currently empty) info dictionary, where one could store useful
        # data about the simulation
        if showAtmos:
            if len(self.action_buffer) < self.delay:
                self.action_buffer.append(action)
                action = 0 * action
            else:
                self.action_buffer.append(action)
                action = self.action_buffer.pop(0)

        self.sim.setDmShape(action)  # Closed doesn't actually mean closed
        slopes = self.sim.moveAtmos(showAtmos=showAtmos)

        return slopes, self._get_reward(slopes), False, {}

    def render(self, mode="rgb_array"):
        # atmos = self.sim.wfss[0].wfsDetectorPlane
        atmos = np.sum(self.sim.get_scrns(), axis=0)
        wfs = self.sim.sciCams[0].residual
        dm = self.sim.dmShape[0]
        self.sim.doSci()
        psf = self.sim.sciImgs[0]

        plt.clf()

        plt.subplot(1, 5, 1)
        plt.imshow(np.asarray(wfs))
        plt.title("WFS residual phase")

        try:  # First step of the simulation dm shape doesn't exist
            plt.subplot(1, 5, 2)
            plt.imshow(dm)
            plt.title("DM shape")
        except:
            plt.subplot(1, 5, 2)
            plt.title("DM shape")

        plt.subplot(1, 5, 3)
        plt.imshow(np.squeeze(atmos))
        plt.title("Atmos")

        plt.subplot(1, 5, 4)
        plt.imshow(psf)
        plt.title("Image")
        plt.subplot(1, 5, 5)
        plt.imshow(self.sim.wfss[0].wfsDetectorPlane)
        plt.title("Raw WFS image")

        plt.setp(
            plt.gcf().get_axes(), xticks=[], yticks=[]
        )  # Remove axis ticks for readability
        plt.suptitle("Soapy", fontsize=14)

        plt.draw()
        plt.pause(1e-3)  # Pause for plotting to work

    def reset(self, seed=-1):
        self.set_params(seed=seed)
        return self.sim.moveAtmos(useDM=False)

    def close(self):
        self.sim.finishUp()
        plt.close()
        if os.path.exists("tmp_params.yaml"):
            os.remove("tmp_params.yaml")
        else:
            print("No tmp param file found to be cleared!")
        del self.sim

    # ---------------- Simulation initialisation functions ---------------------

    def set_params_file(self, param_file, do_mat=False):
        # Set the Compass parameter file used by simulation and load it the first
        # time. Returns boolean if the config is valid.
        if param_file != self.param_file:
            self.param_file = param_file
            self.S2V = None
            self.V2S = None
            valid_config = self._param_file_parser()
            if valid_config:
                self.set_params(make_mat=True)

                return True
                # if do_mat:
                #     return np.transpose(np.asarray(self.sim.get_cmat()))
            else:
                return False

    def set_params(self, seed=-1, make_mat=False):
        # Currently always random seed for atmosphere layers initial formation.
        if seed > 0:
            self.atmos_params["seed"] = seed
        self._param_file_gen()
        print("Setting sim up...")
        self.sim.readParams("tmp_params.yaml")
        self.sim.aoinit()
        self.sim.makeIMat(forceNew=True)
        self._def_spaces()
        # Finishes initilizing
        self.step(0 * self.action_space.sample(), showAtmos=True)

    def _def_spaces(self):  # Defining the sizes and min/max of action/observation spaces
        action_h = 1000 * np.ones(self.sim.dms[0].n_acts)
        action_l = -1 * action_h
        self.action_space = spaces.Box(low=action_l, high=action_h)

        obs_h = 1 * np.ones(self.sim.moveAtmos(useDM=False).shape)
        obs_l = -1 * obs_h
        self.observation_space = spaces.Box(low=obs_l, high=obs_h)

    def _param_file_parser(self):
        # Parse Compass parameter file and convert units to match Soapy. Returns
        # boolean if config is valid
        print("Parsing param file...")

        (
            self.tel_params,
            self.atmos_params,
            self.gs_params,
            self.science_params,
            self.wfs_params,
            self.delay,
        ) = parse_file(self.param_file)

        # Unit conversion
        self.gs_params["wavelength"] = 10 ** (-6) * self.gs_params["wavelength"]
        self.science_params["wavelength"] = 10 ** (-6) * self.science_params["wavelength"]
        self.tel_params["obsDiam"] = (
            self.tel_params["diam"] * self.tel_params["obstructionRatio"]
        )

        if self.wfs_params["type"] == "pyrhr":
            print(
                "\033[93m"
                + "Soapy doesnt support pyramid WFS. Use Compass or OOMAO instead!"
                + "\x1b[0m"
            )
            return False

        if ("alt" in self.gs_params) and (self.gs_params["alt"] > 0):
            print("\033[93m" + "LGS currently not supported by Soapy!" + "\x1b[0m")
            return False

        return True

    def _param_file_gen(self):
        # Generate a param file following Soapy format to load the simulation
        # parameters from
        print("Creating temporary param file...")
        f = open("tmp_params.yaml", "w")

        f.write("simName:\n")
        f.write("pupilSize: " + str(int(self.tel_params["pupdiam"])) + "\n")
        f.write("nGS:  1\n")
        f.write("nDM:  1\n")
        f.write("nSci:  1\n")
        f.write("nIters:  1000\n")  # Doesnt really matter
        f.write("loopTime:  " + str(self.tel_params["samplingTime"]) + "\n")
        f.write("threads:  1\n")

        f.write("\nverbosity: 1\n")

        f.write("\nsaveCMat:  False\n")
        f.write("saveSlopes:  False\n")
        f.write("saveDmCommands:  False\n")
        f.write("saveLgsPsf: False\n")
        f.write("saveSciPsf: False\n")

        f.write("\nAtmosphere:\n")
        f.write("  scrnNo:  " + str(int(self.atmos_params["nscreens"])) + "\n")
        f.write("  scrnHeights:  " + str(self.atmos_params["altitude"]) + "\n")
        f.write("  scrnStrengths:  " + str(self.atmos_params["fractionalR0"]) + "\n")
        f.write("  windDirs:  " + str(self.atmos_params["windDirection"]) + "\n")
        f.write("  windSpeeds:  " + str(self.atmos_params["windSpeed"]) + "\n")
        if "seed" in self.atmos_params:
            f.write("  randomSeed:  " + str(self.atmos_params["seed"]) + "\n")
        f.write("  r0:  " + str(self.atmos_params["r0"]) + "\n")
        f.write("  L0:  " + str(self.atmos_params["layeredL0"]) + "\n")
        f.write("  infinite: True\n")
        f.write("  wholeScrenSize:  2048\n")

        f.write("\nTelescope:\n")
        f.write("  telDiam:  " + str(self.tel_params["diam"]) + "\n")
        f.write("  obsDiam:  " + str(self.tel_params["obsDiam"]) + "\n")
        f.write("  mask: circle\n")

        f.write("\nWFS:\n")
        f.write("  0:\n")
        f.write("    type: ShackHartmann\n")
        f.write(
            "    GSPosition:  "
            + str([self.gs_params["xpos"], self.gs_params["ypos"]])
            + "\n"
        )
        if "alt" in self.gs_params:
            f.write("    GSHeight:  " + str(self.gs_params["alt"]) + "\n")
        f.write("    GSMag:  " + str(self.gs_params["magnitude"]) + "\n")
        f.write("    nxSubaps:  " + str(int(self.wfs_params["nxsub"])) + "\n")
        f.write("    pxlsPerSubap:  " + str(int(self.wfs_params["npix"])) + "\n")
        f.write("    subapFOV:  2.5\n")
        f.write("    wavelength:  " + str(self.gs_params["wavelength"]) + "\n")
        f.write("    subapThreshold:  " + str(self.wfs_params["fracsub"]) + "\n")
        f.write("    throughput:  " + str(self.wfs_params["opt_thr"]) + "\n")
        f.write("    propagationMode:  Physical\n")
        f.write("    fftOversamp:  6\n")
        f.write("    fftwThreads:  1\n")
        if self.wfs_params["noise"] < 0:
            f.write("    photonNoise:  False\n")
            f.write("    eReadNoise:  0\n")
        elif self.wfs_params["noise"] == 0:
            f.write("    photonNoise:  True\n")
            f.write("    eReadNoise:  0\n")
        else:
            f.write("    photonNoise:  True\n")
            f.write("    eReadNoise:  " + str(self.wfs_params["noise"]) + "\n")
            # f.write("    lgs:  True\n")
            # f.write("    lgs:\n")
            # f.write("      wavelength:  "+str(self.gs_params["wavelength"])+"\n")
            # f.write("      height:  "+str(self.gs_params["alt"])+"\n")
            # f.write("      elongationDepth: 12000\n")
            # f.write("      launchPosition:  "+str([self.gs_params["lltx"],self.gs_params["llty"]])+"\n")
            # f.write("      pupilDiam: 0.3\n")

        f.write("\nDM:\n")
        f.write("  0:\n")
        f.write("    type:  GaussStack\n")
        f.write("    closed:  True\n")
        f.write("    nxActuators:  " + str(int(self.wfs_params["nxsub"] + 1)) + "\n")
        f.write("    iMatValue:  500\n")

        f.write("\nReconstructor:\n")
        f.write("  type: MVM\n")
        f.write("  svdConditioning:  0.03\n")
        f.write("  gain:  0.4\n")

        f.write("\nScience:\n")
        f.write("  0:\n")
        f.write(
            "    position:  "
            + str([self.science_params["xpos"], self.science_params["ypos"]])
            + "\n"
        )
        f.write("    FOV:  1.0\n")
        f.write("    wavelength:  " + str(self.science_params["wavelength"]) + "\n")
        f.write("    pxls:  64\n")
        f.write("    propagationMode:  Geometric\n")
        f.write("    instStrehlWithTT:  True\n")
        f.write("    fftOversamp:  6\n")
        f.write("    fftwThreads:  1\n")

        f.close()

    # -------------------------- Updating params outside the sim ---------------

    def reset_custom_params(self):
        self._param_file_parser()

    def set_r0(self, r0):
        self.atmos_params["r0"] = r0

    def set_noise(self, noise):
        self.wfs_params["noise"] = noise

    def set_gsmag(self, mag):
        self.gs_params["magnitude"] = mag

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

    # -------------------------- Misc. functions -------------------------------

    def get_strehl(self):
        self.sim.doSci()
        return self.sim.sciCams[0].instStrehl

    def _get_reward(self, slopes, type="volt"):
        if self.S2V is not None and type != "sh":
            res_volt = np.matmul(self.S2V, slopes)
            reward = -1 * np.linalg.norm(res_volt)
        else:
            reward = self.get_strehl()

        return reward

    def get_wavelen(self):
        return self.gs_params["wavelength"] * (10 ** (6))

    def get_valid(self):
        return self.sim.dms[0].n_valid_actuators
        # return self.sim.dms[0].n_acts

    def get_imat(self):
        return self.sim.get_imat()

    def get_dm_shape(self, dm_idx=0):
        return self.sim.dmShape[dm_idx]

    def get_phase_screen(self, target=0):
        return self.sim.sciCams[target].residual

    def get_KL_modes2volt(self, n_modes=None):
        print("Modes not supported by Soapy!")
        time.sleep(2)

        return None
