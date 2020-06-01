from SUAVE.Core import Units, Data
## @ingroup Methods-Weights-Correlations-Raymer
# prop_system.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

def total_prop_Raymer(vehicle, prop):
    """ Calculate the weight of propulsion system using Raymer method, including:
        - dry engine weight
        - fuel system weight
        - thurst reversers weight
        - electrical system weight
        - starter engine weight
        - nacelle weight
        - cargo containers

        Assumptions:
            1) Rated thrust per scaled engine and rated thurst for baseline are the same
            2) Engine weight scaling parameter is 1.15
            3) Enginge inlet weight scaling exponent is 1
            4) Baseline inlet weight is 0 lbs as in example files FLOPS
            5) Baseline nozzle weight is 0 lbs as in example files FLOPS

        Source:
            Aircraft Design: A Conceptual Approach

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
            prop    - data dictionary for the specific propulsor that is being estimated [dimensionless]

        Outputs:
            output - data dictionary with weights                               [kilograms]
                    - output.wt_prop: total propulsive system weight
                    - output.wt_thrust_reverser: thurst reverser weight
                    - output.starter: starter engine weight
                    - output.wt_engine_controls: engine controls weight
                    - output.fuel_system: fuel system weight
                    - output.nacelle: nacelle weight
                    - output.wt_eng: dry engine weight

        Properties Used:
            N/A
    """
    NENG            = prop.number_of_engines
    WFSYS           = fuel_system_Raymer(vehicle, NENG)
    WENG            = engine_Raymer(vehicle, prop)
    WNAC            = nacelle_Raymer(vehicle, prop, WENG)
    WEC, WSTART     = misc_engine_Raymer(vehicle, prop, WENG)
    WTHR            = 0
    WPRO = NENG * WENG + WFSYS + WEC + WSTART + WTHR + WNAC

    output                      = Data()
    output.wt_prop              = WPRO
    output.wt_thrust_reverser   = WTHR
    output.wt_starter           = WSTART
    output.wt_engine_controls   = WEC
    output.fuel_system          = WFSYS
    output.nacelle              = WNAC
    output.wt_eng               = WENG * NENG
    return output

def nacelle_Raymer(vehicle, prop, WENG):
    NENG    = prop.number_of_engines
    Kng     = 1 # assuming the engine is not pylon mounted
    Nlt     = prop.engine_length / Units.ft
    Nw      = prop.nacelle_diameter / Units.ft
    Wec     = 2.331 * WENG ** 0.901 * 1.18
    Sn      = 2 * np.pi * Nw/2 * Nlt + np.pi * Nw**2/4 * 2
    WNAC = 0.6724 * Kng * Nlt ** 0.1 * Nw ** 0.294 * vehicle.envelope.ultimate_load ** 0.119 \
           * Wec ** 0.611 * NENG * 0.984 * Sn ** 0.224
    return WNAC * Units.lbs


def misc_engine_Raymer(vehicle, prop, WENG):
    NENG    = prop.number_of_engines
    Lec     = NENG * vehicle.fuselages['fuselage'].lengths.total / Units.ft
    WEC     = 5 * NENG + 0.8 * Lec
    WSTART  = 49.19*(NENG*WENG/1000)**0.541
    return WEC * Units.lbs, WSTART * Units.lbs


def fuel_system_Raymer(vehicle, NENG):
    VMAX    = vehicle.design_mach_number
    FMXTOT  = vehicle.mass_properties.max_zero_fuel / Units.lbs
    WFSYS = 1.07 * FMXTOT ** 0.58 * NENG ** 0.43 * VMAX ** 0.34
    return WFSYS * Units.lbs

def engine_Raymer(vehicle, prop):
    EEXP    = 1.15
    EINL    = 1
    ENOZ    = 1
    THRSO   = prop.sealevel_static_thrust * 1 / Units.lbf
    THRUST  = THRSO
    if vehicle.systems.accessories == "short-range" or vehicle.systems.accessories == "commuter":
        WENGB = THRSO / 10.5
    else:
        WENGB = THRSO / 5.5
    WINLB   = 0 / Units.lbs
    WNOZB   = 0 / Units.lbs
    WENGP   = WENGB * (THRUST / THRSO) ** EEXP
    WINL    = WINLB * (THRUST / THRSO) ** EINL
    WNOZ    = WNOZB * (THRUST / THRSO) ** ENOZ
    WENG    = WENGP + WINL + WNOZ
    return WENG * Units.lbs
