## @ingroup Methods-Weights-Correlations-Raymer
# systems.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units, Data
import numpy as np


def systems_Raymer(vehicle):
    """ Calculates the system weight based on the Raymer method

        Assumptions:
            Number of flight control systems = 4
            Max APU weight = 70 lbs
            Assuming not a reciprocating engine and not a turboprop
            System Electrical Rating: 60 kv · A (typically 40-60 for transports, 110-160 for fighters & bombers)
            Uninstalled Avionics weight: 1400 lb (typically= 800-1400 lb)

        Source:
            Aircraft Design: A Conceptual Approach

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]

        Outputs:
            output - a data dictionary with fields:
               wt_flt_ctrl - weight of the flight control system                                [kilograms]
               wt_apu - weight of the apu                                                       [kilograms]
               wt_hyd_pnu - weight of the hydraulics and pneumatics                             [kilograms]
               wt_instruments - weight of the instruments and navigational equipment            [kilograms]
               wt_avionics - weight of the avionics                                             [kilograms]
               wt_elec - weight of the electrical items                                         [kilograms]
               wt_ac - weight of the air conditioning and anti-ice system                       [kilograms]
               wt_furnish - weight of the furnishings in the fuselage                           [kilograms]
               wt_anti_ice - weight of anti-ice system                                          [kilograms]

        Properties Used:
            N/A
    """
    if vehicle.passengers >= 150:
        NFLCR = 3 # number of flight crew
    else:
        NFLCR = 2
    Ns      = 4  # Number of flight control systems (typically 4)
    Kr      = 1  # assuming not a reciprocating engine
    Ktp     = 1  # assuming not a turboprop
    Nf      = 7  # number of functions performed by controls (typically 4-7)
    Rkva    = 60  # system electrical rating
    Wuav    = 1400  # uninstalled avionics weight

    L   = vehicle.fuselages['fuselage'].lengths.total / Units.ft
    Bw  = vehicle.wings['main_wing'].spans.projected / Units.ft
    DG  = vehicle.mass_properties.max_takeoff / Units.lbs
    Scs = vehicle.flap_ratio * vehicle.reference_area / Units.ft**2

    WSC = 36.28 * vehicle.design_mach_number**0.003 * Scs**0.489 * Ns**0.484 * NFLCR**0.124

    if vehicle.passengers >= 6.:
        apu_wt = 7.0 * vehicle.passengers
    else:
        apu_wt = 0.0  # no apu if less than 9 seats
    WAPU            = max(apu_wt, 70./Units.lbs)
    propulsor_name  = list(vehicle.propulsors.keys())[0]
    propulsors      = vehicle.propulsors[propulsor_name]
    NENG            = propulsors.number_of_engines
    WIN = 4.509 * Kr * Ktp * NFLCR ** 0.541 * NENG * (L + Bw) ** 0.5
    WHYD = 0.2673 * Nf * (L + Bw) ** 0.937
    WELEC = 7.291 * Rkva ** 0.782 * (2*L) ** 0.346 * NENG ** 0.1
    WAVONC = 1.73 * Wuav ** 0.983

    L   = vehicle.fuselages['fuselage'].lengths.total / Units.ft
    D   = (vehicle.fuselages['fuselage'].width +
            vehicle.fuselages['fuselage'].heights.maximum) / 2. * 1 / Units.ft
    Sf  = np.pi * (L / D - 1.7) * D ** 2  # Fuselage wetted area, ft**2
    WFURN = 0.0577 * NFLCR ** 0.1 * (vehicle.payload / Units.lbs) ** 0.393 * Sf ** 0.75 + 46 * vehicle.passengers
    WFURN += 75 * NFLCR
    WFURN += 2.5 * vehicle.passengers**1.33

    Vpr = D ** 2 * np.pi / 4 * L
    WAC = 62.36 * vehicle.passengers ** 0.25 * (Vpr / 1000) ** 0.604 * Wuav ** 0.1

    WAI = 0.002 * DG

    output                      = Data()
    output.wt_flight_control    = WSC * Units.lbs
    output.wt_apu               = WAPU * Units.lbs
    output.wt_hyd_pnu           = WHYD * Units.lbs
    output.wt_instruments       = WIN * Units.lbs
    output.wt_avionics          = WAVONC * Units.lbs
    output.wt_elec              = WELEC * Units.lbs
    output.wt_ac                = WAC * Units.lbs
    output.wt_furnish           = WFURN * Units.lbs
    output.wt_anti_ice          = WAI * Units.lbs
    output.wt_systems           = WSC + WAPU + WIN + WHYD + WELEC + WAVONC + WFURN + WAC + WAI
    return output
