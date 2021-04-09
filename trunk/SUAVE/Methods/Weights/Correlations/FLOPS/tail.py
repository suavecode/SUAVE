## @ingroup Methods-Weights-Correlations-FLOPS
# tail.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units

## @ingroup Methods-Weights-Correlations-FLOPS
def tail_vertical_FLOPS(vehicle, wing):
    """ Calculate the vertical tail weight

        Assumptions:
           Conventional tail configuration

        Source:
            The Flight Optimization System Weight Estimation Method

       Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.mass_properties.max_takeoff: MTOW                             [kilograms]
            wing - data dictionary with vertical tail properties                [dimensionless]
                -.taper: taper of wing
                -.areas.reference: surface area of wing                         [m^2]

       Outputs:
            WVT - vertical tail weight                                          [kilograms]

        Properties Used:
            N/A
        """
    DG          = vehicle.mass_properties.max_takeoff / Units.lbs  # Design gross weight in lb
    TRVT        = wing.taper
    NVERT       = 1  # Number of vertical tails
    WVT         = 0.32 * DG ** 0.3 * (TRVT + 0.5) * NVERT ** 0.7 * (wing.areas.reference/Units.ft**2)**0.85
    return WVT * Units.lbs

def tail_horizontal_FLOPS(vehicle, wing):
    """ Calculate the horizontal tail weight

        Assumptions:
           Conventional tail configuration

        Source:
            The Flight Optimization System Weight Estimation Method

       Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.mass_properties.max_takeoff: MTOW                             [kilograms]
            wing - data dictionary with vertical tail properties                [dimensionless]
                -.taper: taper of wing
                -.areas.reference: surface area of wing                         [m^2]

       Outputs:
            WHT - vertical tail weight                                          [kilograms]

        Properties Used:
            N/A
        """
    SHT     = wing.areas.reference / Units.ft **2
    DG      = vehicle.mass_properties.max_takeoff / Units.lbs
    TRHT    = wing.taper
    WHT     = 0.53 * SHT * DG ** 0.2 * (TRHT + 0.5)
    return WHT * Units.lbs

