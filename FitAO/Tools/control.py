import time
from FitAO.Tools.mat import *


class Control:
    def __init__(self, skip_WFS=False):
        # skip_WFS allows reading wavefronts directly instead of approximating
        # it by using a WFS.
        self.wfs_mes = []
        self.actions = []
        self.env = None
        self.S2V = None
        self.V2S = None
        self.pmat = None
        self.infmat = None
        self.skip_WFS = skip_WFS

    def closed_loop(self, obs, gain=0.6):
        self.wfs_mes.append(obs)

        if not self.skip_WFS:
            if len(self.actions) == 0 and self.S2V is not None:
                self.actions.append(0 * np.matmul(self.S2V, obs))
            elif self.S2V is None:
                print("Control matrix missing! Please insert it using set_S2V")
                time.sleep(5)

            self.actions.append(self.actions[-1] - gain * np.matmul(self.S2V, obs))
        else:
            if len(self.actions) == 0 and self.pmat is not None:
                self.actions.append(
                    0 * np.matmul(self.pmat, self.env.get_phase_screen(0).flatten())
                )
            elif self.pmat is None:
                print("Projection matrix missing! Please insert it using set_pmat")
                time.sleep(5)

            if self.env.name == "OOMAO" or self.env.name == "Soapy":
                gain = -1 * gain

            self.actions.append(
                self.actions[-1]
                - gain * np.matmul(self.pmat, self.env.get_phase_screen(0).flatten())
            )

        return self.actions[-1]

    # def get_closed_gain(self,env):
    # TODO: Optimizer for closed loop gain

    def pseudo_open_loop(self, obs, gain=1):
        # Gain can be used to limit overshoots by errors in estimates.
        if not self.skip_WFS:
            self.wfs_mes.append(obs)

            if len(self.actions) == 0 and self.S2V is not None and self.V2S is not None:
                self.actions.append(0 * np.matmul(self.S2V, obs))
            elif self.S2V is None or self.V2S is None:
                print("Control/interaction matrix might be missing!")
                time.sleep(5)

            openloop_obs = obs - np.matmul(self.V2S, self.actions[-1])

            self.actions.append(-1 * gain * np.matmul(self.S2V, openloop_obs))
        else:
            self.wfs_mes.append(obs)

            if (
                len(self.actions) == 0
                and self.pmat is not None
                and self.infmat is not None
            ):
                self.actions.append(
                    0 * np.matmul(self.pmat, self.env.get_phase_screen(0).flatten())
                )
            elif self.pmat is None or self.infmat is None:
                print("Influence/projection matrix might be missing!")
                time.sleep(5)

            openloop_obs = self.env.get_phase_screen(0).flatten() - np.matmul(
                self.infmat, self.actions[-1]
            )

            if self.env.name == "OOMAO" or self.env.name == "Soapy":
                gain = -1 * gain

            self.actions.append(-1 * gain * np.matmul(self.pmat, openloop_obs))

        return self.actions[-1]

    def do_matrices(self, env, forceNew=True):
        self.env = env
        if forceNew:
            if not self.skip_WFS:
                print("Creating new interaction/control matrix...")
                self.V2S = do_imat(self.env)
                env.set_V2S(self.V2S)
                save_mats(env)
                self.S2V = do_cmat(self.env, 0.05, self.V2S)
                env.set_S2V(self.S2V)
                # self.F = do_F(self.env)
            else:
                print("Creating new influence/projection matrix...")
                self.infmat = make_influence_matrix(self.env)
                self.pmat = create_DM_projection_matrix(self.env, self.infmat)
            save_mats(env)
        else:
            try:
                if not self.skip_WFS:
                    load_mats(self.env, self.skip_WFS)
                    self.V2S = self.env.get_V2S()
                    self.S2V = self.env.get_S2V()
                    print("Loaded pre-existing interaction/control matrix.")
                else:
                    load_mats(self.env, self.skip_WFS)
                    self.pmat = self.env.get_pmat()
                    self.infmat = self.env.get_infmat()
                    print("Loaded pre-existing influence/projection matrix.")
            except Exception as e:
                print("Failed to load pre-existing matrices!")
                print("Reason: " + str(e))
                self.do_matrices(self.env)

    def reset(self):
        self.wfs_mes = []
        self.actions = []

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


# Backwards compatibility for refactored class name
control = Control
