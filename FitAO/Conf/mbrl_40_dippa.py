import shesha.config as conf

# simul_name="bench_scao_pyr40_8pix"

# loop
p_loop = conf.Param_loop()

p_loop.set_niter(10000000)
p_loop.set_ittime(0.001)  # =1/1000
p_loop.set_devices([0])
# geom
p_geom = conf.Param_geom()

p_geom.set_zenithangle(0.)
#p_geom.set_pupdiam(392)

# tel
p_tel = conf.Param_tel()

p_tel.set_diam(6.5)
p_tel.set_cobs(0.14)

# atmos
p_atmos = conf.Param_atmos()

p_atmos.set_r0(0.16)
p_atmos.set_nscreens(3)
p_atmos.set_frac([0.50, 0.35, 0.15])
p_atmos.set_alt([0, 5, 10])
#p_atmos.set_windspeed([20, 40, 40])
p_atmos.set_windspeed([10, 16, 35])
p_atmos.set_winddir([0, 20, 180])
p_atmos.set_L0([30, 30 ,30])

#p_atmos.set_r0(0.15)
#p_atmos.set_nscreens(1)
#p_atmos.set_frac([1])
#p_atmos.set_alt([0])
#p_atmos.set_windspeed([2])
#p_atmos.set_winddir([0])
#p_atmos.set_L0([100])

# target
p_target = conf.Param_target()
p_targets = [p_target]
p_target.set_xpos(0.)
p_target.set_ypos(0.)
p_target.set_Lambda(1.6) #microns
p_target.set_mag(0.)


p_wfs0 = conf.Param_wfs()
p_wfss = [p_wfs0]

p_wfs0.set_type("pyrhr")
p_wfs0.set_nxsub(49)
p_wfs0.set_fstop("round")
p_wfs0.set_fssize(2.5) # arcsec
p_wfs0.set_fracsub(0.5)
p_wfs0.set_xpos(0.) #TO-DO: check meaning
p_wfs0.set_ypos(0.)
p_wfs0.set_Lambda(0.85) #microns
p_wfs0.set_gsmag(9)
p_wfs0.set_optthroughput(0.15) #throughput + multip. L3 CCD noise
p_wfs0.set_zerop(1.7e10)
p_wfs0.set_noise(0)
#p_wfs0.set_fssize()
p_wfs0.set_pyr_npts(1)
p_wfs0.set_pyr_ampl(0)#1e-8) #0.00000001
p_wfs0.set_atmos_seen(1)
#p_wfs0.set_pyr_pup_sep(80)

# dm
p_dm0 = conf.Param_dm()
p_dm1 = conf.Param_dm()
p_dms = [p_dm0,p_dm1]

p_dm0.set_type("pzt")
nact = p_wfs0.nxsub + 1
p_dm0.set_nact(11)
p_dm0.set_alt(0.)
p_dm0.set_thresh(0.3)
p_dm0.set_coupling(0.3)
p_dm0.set_unitpervolt(1)
p_dm0.set_push4imat(0.01)

p_dm1.set_type("pzt")
p_dm1.set_nact(50)
p_dm1.set_alt(0.)
p_dm1.set_thresh(0.8)
p_dm1.set_coupling(0.3)
p_dm1.set_unitpervolt(1)
p_dm1.set_push4imat(0.01)
#p_dm0.set_margin_out(5)
#p_dm0.set_margin_in(5)


#p_dm1.set_type("tt")
#p_dm1.set_alt(0.)
#p_dm1.set_unitpervolt(0.0005)
#p_dm1.set_push4imat(100)


# centroiders
p_centroider0 = conf.Param_centroider()
p_centroiders = [p_centroider0]

p_centroider0.set_nwfs(0)
p_centroider0.set_type("maskedpix") #maskedpix/pyr


# controllers
p_controller0 = conf.Param_controller()
p_controllers = [p_controller0]


#p_controller0.do_kl_imat = True
#p_controller0.klpush = [0.04]
#p_controller0.nModesFilt = 1

p_controller0.set_type("generic")
p_controller0.set_nwfs([0])
p_controller0.set_ndm([0,1])
p_controller0.set_maxcond(1000)
p_controller0.set_delay(2)
p_controller0.set_gain(0.4)
#p_controller0.set_nmodes(900).

#p_controller0.set_modopti(0)
#p_controller0.set_nrec(2048)
#p_controller0.set_nmodes(1284)
#p_controller0.set_gmin(0.001)
#p_controller0.set_gmax(0.5)
#p_controller0.set_ngain(500)
