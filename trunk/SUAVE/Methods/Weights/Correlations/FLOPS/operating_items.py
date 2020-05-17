from SUAVE.Core import Units, Data
import numpy as np


# Assumptions:
# 1) Every plane has 5 fuel tanks (includes main and auxiliary tanks)
# 2) If the number of coach seats is not defined, then it assumed that 5% of
#    of the seats are first class and an additional 10 % are business class.
#    If the number of coach seats is defined, then the additional seats are 1/4 first class
#    and 3/4 business class
def operating_system_FLOPS(vehicle):
    propulsor_name = list(vehicle.propulsors.keys())[0]
    propulsors = vehicle.propulsors[propulsor_name]
    NENG = propulsors.number_of_engines
    THRUST = propulsors.sealevel_static_thrust * 1 / Units.lbf
    SW = vehicle.reference_area / Units.ft ** 2
    NTANK = 5  # Number of fuel tanks
    FMXTOT = vehicle.mass_properties.max_zero_fuel / Units.lbs
    WUF = 11.5 * NENG * THRUST ** 0.2 + 0.07 * SW + 1.6 * NTANK * FMXTOT ** 0.28
    WOIL = 0.082 * NENG * THRUST ** 0.65
    if hasattr(vehicle.fuselages['fuselage'], 'number_coach_seats'):
        NPT = vehicle.fuselages['fuselage'].number_coach_seats
        NPF = (vehicle.passengers - NPT) / 4.
        NPB = vehicle.passengers - NPF - NPT
    else:
        NPF = vehicle.passengers / 20.
        NPB = vehicle.passengers / 10.
        NPT = vehicle.passengers - NPF - NPB
    vehicle.NPF = NPF
    vehicle.NPB = NPB
    vehicle.NPT = NPT
    DESRNG = vehicle.design_range / Units.nmi
    VMAX = vehicle.design_mach_number
    WSRV = (5.164 * NPF + 3.846 * NPB + 2.529 * NPT) * (DESRNG / VMAX) ** 0.255
    WCON = 175 * np.ceil(vehicle.mass_properties.cargo / Units.lbs * 1. / 950)

    if vehicle.passengers >= 150:
        NFLCR = 3
        NGALC = 1 + np.floor(vehicle.passengers / 250.)
    else:
        NFLCR = 2
        NGALC = 0
    if vehicle.passengers < 51:
        NSTU = 1
    else:
        NSTU = 1 + np.floor(vehicle.passengers / 40.)

    WSTUAB = NSTU * 155 + NGALC * 200
    WFLCRB = NFLCR * 225

    output = Data()
    output.oper_items = WUF * Units.lbs + WOIL * Units.lbs + WSRV * Units.lbs + WCON * Units.lbs
    output.flight_crew = WFLCRB * Units.lbs
    output.flight_attendants = WSTUAB * Units.lbs
    output.total = output.oper_items + output.flight_crew + output.flight_attendants

    return output
