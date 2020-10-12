## @ingroup Methods-Weights-Correlations-Raymer
# landing_gear.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units, Data

## @ingroup Methods-Weights-Correlations-Raymer
def landing_gear_Raymer(vehicle):
    """ Calculate the weight of the landing gear of a transport aircraft based on the Raymer method

        Assumptions:
            No fuselage mounted landing gear
            1 cargo door
            gear load factor = 3
            number of main gear shock struts = 2
            stall speed = 51 kts as defined by FAA
            Not a reciprocating engine

        Source:
            Aircraft Design: A Conceptual Approach (2nd edition)

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.design_range: design range of aircraft                        [m]
                -.mass_properties.max_takeoff: MTOW                             [kg]
                -.systems.accessories: type of aircraft (short-range, commuter
                                                        medium-range, long-range,
                                                        sst, cargo)
                -.landing_gear.main_strut_length: main strut length              [m]
                -.landing_gear.main_wheels: number of wheels on main landing gear
                -.landing_gear.nose_strut_length: nose strut length              [m]
                -.landing_gear.nose_wheels: number of wheels on nose landing gear

            fuse - data dictionary with specific fuselage properties            [dimensionless]

        Outputs:
            weight_fuse - weight of the fuselage                                [kilograms]

        Properties Used:
            N/A
    """
    Kmp         = 1  # assuming not a kneeling gear
    if vehicle.systems.accessories == "sst":
        RFACT   = 0.00009
    else:
        RFACT   = 0.00004
    DESRNG      = vehicle.design_range / Units.nmi  # Design range in nautical miles
    WLDG        = vehicle.mass_properties.max_takeoff / Units.lbs * (1 - RFACT * DESRNG)
    Ngear       = 3  # gear load factor, usually around 3
    Nl          = Ngear * 1.5  # ultimate landing load factor
    Lm          = vehicle.landing_gear.main_strut_length / Units.inch
    Nmss        = 2  # number of main gear shock struts assumed to be 2
    Nmw         = vehicle.landing_gear.main_wheels * Nmss
    Vstall      = 51 * Units.kts  # stall speed
    Knp         = 1  # assuming not a reciprocating engine
    Ln          = vehicle.landing_gear.nose_strut_length / Units.inch
    Nnw         = vehicle.landing_gear.nose_wheels
    wt_main_landing_gear = 0.0106 * Kmp * WLDG ** 0.888 * Nl ** 0.25 * Lm ** 0.4 * Nmw ** 0.321 * Nmss ** (-0.5) * Vstall ** 0.1
    wt_nose_landing_gear = 0.032 * Knp * WLDG ** 0.646 * Nl ** 0.2 * Ln ** 0.5 * Nnw ** 0.45

    output          = Data()
    output.main     = wt_main_landing_gear * Units.lbs
    output.nose     = wt_nose_landing_gear * Units.lbs
    return output
