from CompassEnv.CompassEnv import CompassEnv
from OOMAOEnv.PyOOMAO import PyOOMAO
from SoapyEnv.SoapyEnv import SoapyEnv
import Tools.mat as tm
import matplotlib.pyplot as plt
import numpy

param_file = "./Conf/sh_16x8.py"

# env3 = PyOOMAO()
# env3.set_params_file(param_file)
env2 = CompassEnv()
S2 = env2.set_params_file(param_file)
# env = soapyEnv()
# env.set_params_file(param_file)
# S = env.get_imat()

# #imat_soa = tm.do_imat(env)
# imat_com = tm.do_imat(env2)
S2V = tm.do_cmat(env2, 0.05)
# # imat_oom = tm.do_imat(env3)
# #
# # plt.imshow(imat_oom)
# # plt.colorbar()
# # plt.title("OOMAO")
#
# plt.figure()
# plt.imshow(imat_com)
# ##print(numpy.average(imat_com))
# plt.colorbar()
# plt.title("Compass")
#
# plt.figure()
# plt.imshow(S2)
# plt.colorbar()
# plt.title("Compass own")
#
# # plt.figure()
# # plt.imshow(S2 - imat_com)
# # plt.colorbar()
# # plt.title("Compass res")
#
plt.figure()
plt.imshow(S2V)
# plt.colorbar()
# plt.title("Compass control")
#
# # plt.figure()
# # plt.imshow(imat_soa)
# # #print(numpy.average(imat_soa))
# # plt.colorbar()
# # plt.title("Soapy")
# #
# # plt.figure()
# # plt.imshow(imat_soa - numpy.transpose(S))
# # print(numpy.amax(imat_soa - numpy.transpose(S)))
# # plt.colorbar()
# # plt.title("Soapy res")
#
# # res = imat_oom - imat_com
# # plt.figure()
# # plt.imshow(res)
# # plt.colorbar()
#
plt.show()
#
# #Using the generated command matrix
# obs = env2.reset()
# last_action = 0 * numpy.matmul(S2V,obs)
# g = 1
#
# for i in range(300):
#
#     #Rendering WFS phase and DM shape
#     env2.render()
#
#     #Integrator controller
#     action = last_action - g * numpy.matmul(S2V,obs)
#     #Pseudo open loop control
#     # obs = obs + numpy.matmul(Imat,last_action)
#     #
#     # action = -1 * numpy.matmul(S2V,obs)
#
#     last_action = action
#
#     #Applying command and after that moving atmos/updating WFS. Returns slopes as obs,
#     #squared norm of residual voltages as reward, done as False (no current rule
#     #for done), and empty info (can add any extra information we might want to use)
#     obs, reward, done, info = env2.step(action)
#     print(env2.get_strehl())
