from SUAVE.Core import Units, Data
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
import numpy as np


# Assumptions
# 1) Rated thrust per scaled engine and rated thurst for baseline are the same
# 2) Engine weight scaling parameter is 1.15
# 3) Enginge inlet weight scaling exponent is 1
# 4) Baseline inlet weight is 0 lbs as in example files FLOPS
# 5) Baseline nozzle weight is 0 lbs as in example files FLOPS

def nacelle_FLOPS(prop):
    NENG = prop.number_of_engines
    TNAC = NENG + 1. / 2 * (NENG - 2 * np.floor(NENG / 2.))
    DNAC = prop.nacelle_diameter / Units.ft
    XNAC = prop.engine_length / Units.ft
    FTHRST = prop.sealevel_static_thrust * 1 / Units.lbf
    WNAC = 0.25 * TNAC * DNAC * XNAC * FTHRST ** 0.36
    return WNAC * Units.lbs


def thrust_reverser_FLOPS(prop):
    NENG = prop.number_of_engines
    TNAC = NENG + 1. / 2 * (NENG - 2 * np.floor(NENG / 2.))
    THRUST = prop.sealevel_static_thrust * 1 / Units.lbf
    WTHR = 0.034 * THRUST * TNAC
    return WTHR * Units.lbs


def misc_engine_FLOPS(vehicle, prop):
    NENG = prop.number_of_engines
    THRUST = prop.sealevel_static_thrust * 1 / Units.lbf
    WEC = 0.26 * NENG * THRUST ** 0.5
    FNAC = prop.nacelle_diameter / Units.ft
    VMAX = vehicle.design_mach_number
    WSTART = 11.0 * NENG * VMAX ** 0.32 * FNAC ** 1.6
    return WEC * Units.lbs, WSTART * Units.lbs


def fuel_system_FLOPS(vehicle, NENG):
    VMAX = vehicle.design_mach_number
    FMXTOT = vehicle.mass_properties.max_zero_fuel / Units.lbs
    WFSYS = 1.07 * FMXTOT ** 0.58 * NENG ** 0.43 * VMAX ** 0.34
    return WFSYS * Units.lbs


def engine_FLOPS(vehicle, prop):
    EEXP = 1.15
    EINL = 1
    ENOZ = 1
    THRSO = prop.sealevel_static_thrust * 1 / Units.lbf
    THRUST = THRSO
    if vehicle.systems.accessories == "short-range" or vehicle.systems.accessories == "commuter":
        WENGB = THRSO/10.5
    else:
        WENGB = THRSO/5.5
    WINLB = 0 / Units.lbs
    WNOZB = 0 / Units.lbs
    WENGP = WENGB * (THRUST / THRSO) ** EEXP
    WINL = WINLB * (THRUST/THRSO) ** EINL
    WNOZ = WNOZB * (THRUST/THRSO) ** ENOZ
    WENG = WENGP + WINL + WNOZ
    return WENG * Units.lbs

def total_prop_flops(vehicle, prop):
    NENG = prop.number_of_engines
    WNAC = nacelle_FLOPS(prop)
    WFSYS = fuel_system_FLOPS(vehicle, NENG)
    WENG = engine_FLOPS(vehicle, prop)
    WEC, WSTART = misc_engine_FLOPS(vehicle, prop)
    WTHR = thrust_reverser_FLOPS(prop)
    WPRO = NENG * WENG + WFSYS + WEC + WSTART + WTHR + WNAC

    output = Data()
    output.wt_prop              = WPRO
    output.wt_thrust_reverser   = WTHR
    output.wt_starter           = WSTART
    output.wt_engine_controls   = WEC
    output.fuel_system          = WFSYS
    output.nacelle              = WNAC
    output.wt_eng               = WENG * NENG
    return output
