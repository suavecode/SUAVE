from SUAVE.Core import Units, Data
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
import numpy as np


def nacelle_Raymer(vehicle, prop, WENG):
    NENG = prop.number_of_engines
    Kng = 1
    Nlt = prop.engine_length / Units.ft
    Nw = prop.nacelle_diameter / Units.ft
    Wec = 2.331 * WENG ** 0.901 * 1.18
    Sn = 2 * np.pi * Nw/2 * Nlt + np.pi * Nw**2/4 * 2
    WNAC = 0.6724 * Kng * Nlt ** 0.1 * Nw ** 0.294 * vehicle.envelope.ultimate_load ** 0.119 \
           * Wec ** 0.611 * NENG * 0.984 * Sn ** 0.224
    return WNAC * Units.lbs


def misc_engine_Raymer(vehicle, prop, WENG):
    NENG = prop.number_of_engines
    Lec = NENG * vehicle.fuselages['fuselage'].lengths.total / Units.ft
    WEC = 5 * NENG + 0.8 * Lec
    WSTART = 49.19*(NENG*WENG/1000)**0.541
    return WEC * Units.lbs, WSTART * Units.lbs


def fuel_system_Raymer(vehicle, NENG):
    VMAX = vehicle.design_mach_number
    FMXTOT = vehicle.mass_properties.max_zero_fuel / Units.lbs
    WFSYS = 1.07 * FMXTOT ** 0.58 * NENG ** 0.43 * VMAX ** 0.34
    return WFSYS * Units.lbs

def engine_Raymer(vehicle, prop):
    EEXP = 1.15
    EINL = 1
    ENOZ = 1
    THRSO = prop.sealevel_static_thrust * 1 / Units.lbf
    THRUST = THRSO
    if vehicle.systems.accessories == "short-range" or vehicle.systems.accessories == "commuter":
        WENGB = THRSO / 10.5
    else:
        WENGB = THRSO / 5.5
    WINLB = 0 / Units.lbs
    WNOZB = 0 / Units.lbs
    WENGP = WENGB * (THRUST / THRSO) ** EEXP
    WINL = WINLB * (THRUST / THRSO) ** EINL
    WNOZ = WNOZB * (THRUST / THRSO) ** ENOZ
    WENG = WENGP + WINL + WNOZ
    return WENG * Units.lbs


def total_prop_Raymer(vehicle, prop):
    NENG = prop.number_of_engines

    WFSYS = fuel_system_Raymer(vehicle, NENG)
    WENG = engine_Raymer(vehicle, prop)
    WNAC = nacelle_Raymer(vehicle, prop, WENG)
    WEC, WSTART = misc_engine_Raymer(vehicle, prop, WENG)
    WTHR = 0
    WPRO = NENG * WENG + WFSYS + WEC + WSTART + WTHR + WNAC

    output = Data()
    output.wt_prop = WPRO
    output.wt_thrust_reverser = WTHR
    output.wt_starter = WSTART
    output.wt_engine_controls = WEC
    output.fuel_system = WFSYS
    output.nacelle = WNAC
    output.wt_eng = WENG * NENG
    return output
