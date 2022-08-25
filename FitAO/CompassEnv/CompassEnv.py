import gym
from gym import spaces
import os
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
from shesha.config import ParamConfig
from shesha.supervisor.optimizers.modalBasis import ModalBasis
import matplotlib.pyplot as plot
import numpy as np
from skimage.transform import resize


class CompassEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    # --------------------------Core gym funtions--------------------------------

    def __init__(self):
        self.supervisor = None
        self.action_buffer = []  # used to add delay to commands
        self.done = False
        self.param_file = ""
        self.custom_params = {}
        self.delay = 1
        self.S2V = None
        self.V2S = None
        self.pmat = None
        self.infmat = None
        self.F = None
        self.calibConst = None
        self.name = "Compass"

    def step(self, action, showAtmos=True):
        # A single step of an AO system. Action represents the DM command, that
        # is delayed by the action_buffer.

        if showAtmos:
            if len(self.action_buffer) < self.delay:
                self.action_buffer.append(action)
                action = 0 * action
            else:
                self.action_buffer.append(action)
                action = self.action_buffer.pop(0)

        self.supervisor.dms.set_command(action)
        self.supervisor.atmos.is_enable = showAtmos
        self.supervisor.next(move_atmos=showAtmos, do_control=True, apply_control=False)

        return self.get_slopes(), self._get_reward(self.get_slopes()), self.done, {}

    def reset(self, seed=-1, soft=False):
        # Resetting the AO system. Negative (default) seed represents random
        # atmos screens
        if soft:  # Reset DM shape and Strehl
            for i in range(0, int(self.delay) + 1):
                self.step(np.zeros((1, self.get_valid())))
            self.action_buffer = []
            self.supervisor.target.reset_strehl(0)

        else:  # Reset everything (new atmosphere etc.)
            self.supervisor.reset()
            self.set_params(seed=seed)
            self.action_buffer = []
            self.supervisor.next(do_control=False, apply_control=False)

        return self.get_slopes()

    def render(self, mode="rgb_array"):
        n_wfs = len(self.supervisor.config.p_wfss)
        n_dm = len(self.supervisor.config.p_dms)
        n_atm = self.supervisor.config.p_atmos.get_nscreens()

        if n_wfs + n_dm + 2 > 4:
            rows = 2
            col = (n_wfs + n_dm + 2) / 2 + 1
        else:
            rows = 1
            col = n_wfs + n_dm + 2 + 1

        rows = int(rows)
        col = int(col)

        plot.clf()

        for i in range(n_wfs):
            plot.subplot(rows, col, i + 1)
            plot.imshow(self.get_phase_screen(i))
            plot.title("WFS " + str(i) + " residual phase")

        for i in range(n_dm):
            plot.subplot(rows, col, i + n_wfs + 1)
            plot.imshow(self.get_dm_shape(i))
            plot.title("DM " + str(i) + " shape")

        for i in range(n_atm):
            if i == 0:
                atm_screen = resize(self.supervisor.atmos.get_atmos_layer(i), (128, 128))
            else:
                atm_screen += resize(self.supervisor.atmos.get_atmos_layer(i), (128, 128))
        plot.subplot(rows, col, n_wfs + n_dm + 1)
        plot.imshow(atm_screen)
        plot.title("Atmos")

        plot.subplot(rows, col, n_wfs + n_dm + 2)
        plot.imshow(
            self.supervisor.target.get_tar_image(0, expo_type="se")
        )  # expo_type = "le" for long exposure
        plot.xlim(465, 565)
        plot.ylim(465, 565)
        plot.title("Image")

        plot.subplot(rows, col, n_wfs + n_dm + 3)
        if self.supervisor.config.p_wfss[0].get_type() == "pyrhr":
            plot.imshow(self.supervisor.wfs.get_pyrhr_image(0))
        else:
            plot.imshow(self.supervisor.wfs.get_wfs_image(0))
        plot.title("Raw WFS image")

        plot.setp(
            plot.gcf().get_axes(), xticks=[], yticks=[]
        )  # Remove axis ticks for readability
        plot.suptitle("Compass", fontsize=14)

        plot.draw()
        plot.pause(1e-3)  # Pause for plotting to work

    def close(self):
        self.supervisor.reset()
        plot.close()
        del self.supervisor

    # ---------------- Simulation initialisation functions ---------------------

    def set_params_file(self, param_file, do_mat=False):
        # Sets parameters and returns Boolean if the config is valid.
        if param_file != self.param_file:
            self.param_file = param_file
            self.S2V = None
            self.V2S = None
            self.set_params()

        # if do_mat:
        #     return self.supervisor.rtc.get_interaction_matrix(0)

        return True

    def set_params(self, seed=None):
        config = ParamConfig(self.param_file)
        self.supervisor = Supervisor(config)
        # Generate a list of seeds for atmosphere if seeds given. If given seed
        # is < 0, all layers will get a random seed. Otherwise the the layer
        # seeds are calculated as: first layer = seed, second = seed + 1...
        if seed:
            seeds = []
            if seed < 0:
                for i in range(0, self.supervisor.config.p_atmos.get_nscreens()):
                    seeds.append(np.random.randint(10**9))
            else:
                for i in range(0, self.supervisor.config.p_atmos.get_nscreens()):
                    seeds.append(seed + i)
            self.supervisor.config.p_atmos.set_seeds(seeds)

        self.delay = self.supervisor.config.p_controllers[0].get_delay()
        self.calibConst = self.supervisor.config.p_dms[0].get_push4imat()

        self._init_supervisor()
        self._def_spaces()

        # For applying custom parameters
        if "r0" in self.custom_params:
            self.supervisor.config.p_atmos.set_r0(self.custom_params["r0"])
        if "noise" in self.custom_params:
            self.supervisor.config.p_wfss[0].set_noise(self.custom_params["noise"])
        if "gs_mag" in self.custom_params:
            self.supervisor.config.p_wfss[0].set_gsmag(self.custom_params["gs_mag"])
        if "wind_speed" in self.custom_params:
            self.supervisor.config.p_atmos.set_windspeed(self.custom_params["wind_speed"])

    def _def_spaces(self):
        # Defining the sizes and min/max of action/observation spaces
        n_dms = len(self.supervisor.config.p_dms)
        total_act = 0
        for i in range(0, n_dms):
            total_act += self.supervisor.config.p_dms[i].get_ntotact()
        action_h = 1000 * np.ones((total_act,))
        # print(self.supervisor.get_com(0).shape[0])
        action_l = -1 * action_h
        self.action_space = spaces.Box(low=action_l, high=action_h)

        obs_h = 1 * np.ones(self.get_slopes().shape[0])
        obs_l = -1 * obs_h
        self.observation_space = spaces.Box(low=obs_l, high=obs_h)

    def _init_supervisor(self):
        self.supervisor.reset()
        self.supervisor.next(do_control=True, apply_control=False)

        try:
            xvalid = self.supervisor.config.p_dms[1]._xpos
            yvalid = self.supervisor.config.p_dms[1]._ypos
            self.xvalid = (
                (xvalid - xvalid.min()) / self.supervisor.config.p_dms[1]._pitch
            ).astype(np.int)
            self.yvalid = (
                (yvalid - yvalid.min()) / self.supervisor.config.p_dms[1]._pitch
            ).astype(np.int)
            #
            self.n_actu = self.supervisor.config.p_dms[1].nact
            # valid_actus = np.zeros((self.n_actu, self.n_actu))
            #
            # print(valid_actus[self.xvalid, self.yvalid].shape)
            # print(self.xvalid)
            # print(self.yvalid)

            # + 2 for tip/tilt DM
            n_actions = self.get_valid(
                1
            )  # valid_actus[self.xvalid, self.yvalid].shape[1]# + 2
            self.n_actions = n_actions
        except:
            print("not rl")

    # -------------------------- Updating params outside the sim ---------------

    def reset_custom_params(self):
        self.custom_params.clear()

    def set_r0(self, r0):
        self.custom_params["r0"] = r0

    def set_noise(self, noise):
        self.custom_params["noise"] = noise

    def set_gsmag(self, mag):
        self.custom_params["gs_mag"] = mag

    def set_windspeed(self, wind_speed):
        self.custom_params["wind_speed"] = wind_speed

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

    def set_calibConst(self, calibConst):
        self.calibConst = calibConst

    def get_calibConst(self):
        return self.calibConst

    # -------------------------- Misc. functions -------------------------------

    def get_strehl(self):
        return self.supervisor.target.get_strehl(0)[0]

    def get_slopes(self):
        return self.supervisor.rtc.get_slopes(0)

    def _get_reward(self, slopes, type="volt"):
        if self.S2V is not None and type != "sh":
            res_volt = np.matmul(self.S2V, slopes)
            reward = -1 * np.linalg.norm(res_volt)
        else:
            reward = self.get_strehl()

        return reward

    def get_wavelen(self):
        return self.supervisor.config.p_wfss[0].get_Lambda()

    def get_valid(self, dm_idx=0):
        return self.supervisor.config.p_dms[dm_idx].get_ntotact()

    def get_total(self, dm_idx=0):
        return self.supervisor.config.p_dms[dm_idx].nact

    def get_influ(self, dm_idx=0):
        return self.supervisor.basis.compute_influ_basis(dm_idx)

    def get_dm_shape(self, dm_idx=0):
        return self.supervisor.dms.get_dm_shape(dm_idx)

    def get_raw_wfs_image(self, wfs_idx=0):
        if self.supervisor.config.p_wfss[wfs_idx].get_type() == "pyrhr":
            return self.supervisor.wfs.get_pyrhr_image(wfs_idx)
        else:
            return self.supervisor.wfs.get_wfs_image(wfs_idx)

    def get_phase_screen(self, target=0, pupil=True):
        return self.supervisor.target.get_tar_phase(target)

    def get_pupil(self):
        return self.supervisor.config.p_geom.get_spupil()

    def get_KL_modes2volt(self, n_modes=None):
        mb = ModalBasis(
            self.supervisor.config, self.supervisor.dms, self.supervisor.target
        )
        mod2volt, proj = mb.compute_modes_to_volts_basis(modal_basis_type="KL2V")

        if n_modes is not None:
            mod2volt = mod2volt[:, 0:n_modes]

        return mod2volt

    def _get_photonpsubap(self):
        return self.supervisor.config.p_wfss[0]._nphotons

    def get_imat(self):
        return self.supervisor.rtc.get_interaction_matrix(0)

    # ---------------------- RL funtions ----------------------------------------

    def vec_to_img(self, action_vec):
        valid_actus = np.zeros(
            (self.supervisor.config.p_dms[1].nact, self.supervisor.config.p_dms[1].nact)
        )
        valid_actus[self.xvalid, self.yvalid] = action_vec

        return valid_actus

    def img_to_vec(self, action):
        assert len(action.shape) == 2

        return action[self.xvalid, self.yvalid]

    def sample_binary_noise(self, sigma):
        noise = sigma * np.sign(np.random.normal(0, 1, size=(self.n_actions,)))
        return self.vec_to_img(noise)
