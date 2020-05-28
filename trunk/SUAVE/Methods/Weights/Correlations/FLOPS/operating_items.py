## @ingroup Methods-Weights-Correlations-FLOPS
# operating_items.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units, Data
import numpy as np

def operating_system_FLOPS(vehicle):
    """ Calculate the weight of operating items, including:
        - crew
        - baggage
        - unusable fuel
        - engine oil
        - passenger service
        - ammunition and non-fixed weapons
        - cargo containers

        Assumptions:
            Every plane has 5 fuel tanks (includes main and auxiliary tanks)
            If the number of coach seats is not defined, then it assumed that 5% of
                of the seats are first class and an additional 10 % are business class.
            If the number of coach seats is defined, then the additional seats are 1/4 first class
                and 3/4 business class

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]

        Outputs:
            output - data dictionary with weights                               [kilograms]
                    - output.oper_items: unusable fuel, engine oil, passenger service weight and cargo containers
                    - output.flight_crew: flight crew weight
                    - output.flight_attendants: flight attendants weight
                    - output.total: total operating items weight

        Properties Used:
            N/A
    """
    propulsor_name = list(vehicle.propulsors.keys())[0]
    propulsors = vehicle.propulsors[propulsor_name]
    NENG = propulsors.number_of_engines
    THRUST = propulsors.sealevel_static_thrust * 1 / Units.lbf
    SW = vehicle.reference_area / Units.ft ** 2
    NTANK = 5  # Number of fuel tanks
    FMXTOT = vehicle.mass_properties.max_zero_fuel / Units.lbs
    WUF = 11.5 * NENG * THRUST ** 0.2 + 0.07 * SW + 1.6 * NTANK * FMXTOT ** 0.28  # unusable fuel weight
    WOIL = 0.082 * NENG * THRUST ** 0.65  # engine oil weight
    if hasattr(vehicle.fuselages['fuselage'], 'number_coach_seats'):
        NPT = vehicle.fuselages['fuselage'].number_coach_seats  # number of economy passengers
        NPF = (vehicle.passengers - NPT) / 4.  # number of first clss passengers
        NPB = vehicle.passengers - NPF - NPT  # number of bussines passengers
    else:
        NPF = vehicle.passengers / 20.
        NPB = vehicle.passengers / 10.
        NPT = vehicle.passengers - NPF - NPB
    vehicle.NPF = NPF
    vehicle.NPB = NPB
    vehicle.NPT = NPT
    DESRNG = vehicle.design_range / Units.nmi
    VMAX = vehicle.design_mach_number
    WSRV = (5.164 * NPF + 3.846 * NPB + 2.529 * NPT) * (DESRNG / VMAX) ** 0.255  # passenger service weight
    WCON = 175 * np.ceil(vehicle.mass_properties.cargo / Units.lbs * 1. / 950)  # cargo container weight

    if vehicle.passengers >= 150:
        NFLCR = 3  # number of flight crew
        NGALC = 1 + np.floor(vehicle.passengers / 250.)  # number of galley crew
    else:
        NFLCR = 2
        NGALC = 0
    if vehicle.passengers < 51:
        NSTU = 1  # number of flight attendants
    else:
        NSTU = 1 + np.floor(vehicle.passengers / 40.)

    WSTUAB = NSTU * 155 + NGALC * 200  # flight attendant weight
    WFLCRB = NFLCR * 225  # flight crew and baggage weight

    output = Data()
    output.oper_items = WUF * Units.lbs + WOIL * Units.lbs + WSRV * Units.lbs + WCON * Units.lbs
    output.flight_crew = WFLCRB * Units.lbs
    output.flight_attendants = WSTUAB * Units.lbs
    output.total = output.oper_items + output.flight_crew + output.flight_attendants
    return output
