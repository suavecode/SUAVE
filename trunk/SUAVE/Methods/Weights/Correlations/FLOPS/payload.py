## @ingroup Methods-Weights-Correlations-FLOPS
# payload.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units, Data

## @ingroup Methods-Weights-Correlations-FLOPS
def payload_FLOPS(vehicle, weight_per_passenger = 165. * Units.lb):
    """ Calculate the payload weight, including:
        - passenger and carry-on weight
        - baggage weight
        - cargo weight

        Assumptions:
            None

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.passengers: number of passengers in aircraft
                -.design_range: design range of aircraft                        [nmi]
                -.mass_properties.cargo: weight of cargo carried                [kilograms]

        Outputs:
            output - data dictionary with weights                               [kilograms]
                    - output.passengers: passenger weight
                    - output.baggage: baggage weight
                    - output.cargo: cargo weight
                    - output.total: total payload weight

        Properties Used:
            N/A
    """
    WPPASS  = weight_per_passenger
    WPASS   = vehicle.passengers * WPPASS
    DESRNG  = vehicle.design_range / Units.nmi
    if DESRNG <= 900:
        BPP = 35  # luggage weight per passenger depends on the design range
    elif DESRNG <= 2900:
        BPP = 40
    else:
        BPP = 44
    WPBAG       = BPP * vehicle.passengers  # baggage weight
    WPAYLOAD    = WPASS + WPBAG + vehicle.mass_properties.cargo / Units.lbs  # payload weight

    output              = Data()
    output.total        = WPAYLOAD * Units.lbs
    output.passengers   = WPASS * Units.lbs
    output.baggage      = WPBAG * Units.lbs
    output.cargo        = vehicle.mass_properties.cargo
    return output
