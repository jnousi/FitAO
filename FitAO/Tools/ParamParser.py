# Collection of functions to parse Compass parameter files
def parse_file(file_path,format='default'):
    file = open(file_path, 'r')

    tel_params = {}
    atmos_params = {}
    gs_params = {}
    science_params = {}
    wfs_params = {}
    dm_params = {}

    for line in file:
        if line[0] == "#" or line == "":   #Skip commented/empty lines
            continue
        else:
            if line.startswith("p_loop.set_ittime"):
                tel_params["samplingTime"] = parse_single(line)
            if line.startswith("p_tel.set_diam"):
                tel_params["diam"] = parse_single(line)
            if line.startswith("p_tel.set_cobs"):
                tel_params["obstructionRatio"] = parse_single(line)
            if line.startswith("p_geom.set_pupdiam"):   #number of phase points
                tel_params["pupdiam"] = parse_single(line)

            if line.startswith("p_dm0.set_coupling"):
                dm_params["coupling"] = parse_single(line)

            if line.startswith("p_atmos.set_nscreens"):
                atmos_params["nscreens"] = parse_single(line)
            if line.startswith("p_atmos.set_r0"):
                atmos_params["r0"] = parse_single(line)
            if line.startswith("p_atmos.set_frac"):
                atmos_params["fractionalR0"] = parse_list(line,format)
            if line.startswith("p_atmos.set_alt"):
                atmos_params["altitude"] = parse_list(line,format)
            if line.startswith("p_atmos.set_windspeed"):
                atmos_params["windSpeed"] = parse_list(line,format)
            if line.startswith("p_atmos.set_winddir"):
                atmos_params["windDirection"] = parse_list(line,format)
            if line.startswith("p_atmos.set_L0"):
                atmos_params["layeredL0"] = parse_list(line,format)


            if line.startswith("p_target.set_xpos"):
                science_params["xpos"] = parse_single(line)
            if line.startswith("p_target.set_ypos"):
                science_params["ypos"] = parse_single(line)
            if line.startswith("p_target.set_Lambda"):
                science_params["wavelength"] = parse_single(line)
            if line.startswith("p_target.set_mag"):
                science_params["magnitude"] = parse_single(line)


            if line.startswith("p_wfs0.set_type"):
                wfs_params["type"] = parse_single(line)
            #Generic properties
            if line.startswith("p_wfs0.set_nxsub"):
                wfs_params["nxsub"] = parse_single(line)
            if line.startswith("p_wfs0.set_fracsub"):
                wfs_params["fracsub"] = parse_single(line)
            if line.startswith("p_wfs0.set_noise"):
                wfs_params["noise"] = 0.8*parse_single(line)
            if line.startswith("p_wfs0.set_zerop"):
                wfs_params["zero_p"] = parse_single(line)
            if line.startswith("p_wfs0.set_optthroughput"):
                wfs_params["opt_thr"] = parse_single(line)
            if line.startswith("p_wfs0.set_xpos"):
                gs_params["xpos"] = parse_single(line)
            if line.startswith("p_wfs0.set_ypos"):
                gs_params["ypos"] = parse_single(line)
            if line.startswith("p_wfs0.set_Lambda"):
                gs_params["wavelength"] = parse_single(line)
            if line.startswith("p_wfs0.set_gsmag"):
                gs_params["magnitude"] = parse_single(line)
            if line.startswith("p_wfs0.set_gsalt"):
                gs_params["alt"] = parse_single(line)
            if line.startswith("p_wfs0.set_lltx"):
                gs_params["lltx"] = parse_single(line)
            if line.startswith("p_wfs0.set_llty"):
                gs_params["llty"] = parse_single(line)
            if line.startswith("p_dm0.set_push4imat"):
                gs_params["calibConst"] = parse_single(line)
            #Shack Hartmann spesific
            if line.startswith("p_wfs0.set_npix"):
                wfs_params["npix"] = parse_single(line)
            #Pyramid spesific
            if line.startswith("p_wfs0.set_fssize"):
                wfs_params["fssize"] = parse_single(line)
            if line.startswith("p_wfs0.set_pyr_ampl"):
                wfs_params["pyr_amp"] = parse_single(line)
            if line.startswith("p_wfs0.set_pyr_pup_sep"):
                wfs_params["pyr_pup_sep"] = parse_single(line)
            if line.startswith("p_wfs0.set_pyr_npts"):
                wfs_params["pyr_npts"] = parse_single(line)

            if line.startswith("p_controller0.set_delay"):
                delay = parse_single(line)
            if line.startswith("p_controller0.set_maxcond"):
                wfs_params["cmat_cond"] = parse_single(line)
    file.close()

    return tel_params,atmos_params,gs_params,science_params,wfs_params,dm_params,delay


def parse_single(line):
    try:    #Number
        return float(line.split("(")[1].split(")")[0])
    except ValueError:  #String
        return line.split('"')[1].split('"')[0]

def parse_list(line,format):
    str_list = line.split("[")[1].split("]")[0].split(",")
    float_list = []
    for element in str_list:
        float_list.append(float(element))
    if format is 'matlab':
        import matlab
        return matlab.double(float_list)
    else:
        return float_list
