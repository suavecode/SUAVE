from SUAVE.Core import Units, Data

# Assumptions
# 1) The weight of the average passenger is assumed to be 165 lbs

def payload_FLOPS(vehicle):
    WPPASS = 165 * Units.lbs
    WPASS = vehicle.passengers * WPPASS
    DESRNG = vehicle.design_range / Units.nmi
    if DESRNG <= 900:
        BPP = 35
    elif DESRNG <= 2900:
        BPP = 40
    else:
        BPP = 44
    WPBAG = BPP * vehicle.passengers
    WPAYLOAD = WPASS + WPBAG + vehicle.mass_properties.cargo / Units.lbs
    output = Data()
    output.total = WPAYLOAD * Units.lbs
    output.passengers = WPASS * Units.lbs
    output.baggage = WPBAG * Units.lbs
    output.cargo = vehicle.mass_properties.cargo
    return output
