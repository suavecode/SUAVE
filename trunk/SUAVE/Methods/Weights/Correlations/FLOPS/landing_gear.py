## @ingroup Methods-Weights-Correlations-FLOPS
# landing_gear.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units, Data
import numpy as np

## @ingroup Methods-Weights-Correlations-FLOPS
def landing_gear_FLOPS(vehicle):
    """ Calculate the weight of the main and nose landing gear of a transport aircraft

        Assumptions:
            No fighter jet, change DFTE to 1 for a fighter jet
            Aircraft is not meant for carrier operations, change CARBAS to 1 for carrier-based aircraft

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.networks: data dictionary containing all propulsion properties
                -.design_range: design range of aircraft                        [nmi]
                -.systems.accessories: type of aircraft (short-range, commuter
                                                        medium-range, long-range,
                                                        sst, cargo)
                -.mass_properties.max_takeoff: MTOW                              [kilograms]
                -.wings['main_wing'].dihedral
                -.fuselages['fuselage'].width: fuselage width                    [meters]
                -.fuselages['fuselage'].lengths.total: fuselage total length     [meters]


        Outputs:
            output - data dictionary with main and nose landing gear weights    [kilograms]
                    output.main, output.nose

        Properties Used:
            N/A
    """
    DFTE    = 0
    CARBAS  = 0
    if vehicle.systems.accessories == "sst":
        RFACT = 0.00009
    else:
        RFACT = 0.00004
    DESRNG  = vehicle.design_range / Units.nmi  # Design range in nautical miles
    WLDG    = vehicle.mass_properties.max_takeoff / Units.lbs * (1 - RFACT * DESRNG)

    network_name  = list(vehicle.networks.keys())[0]  # obtain the key for the network for assignment purposes
    networks      = vehicle.networks[network_name]
    if sum(networks.wing_mounted) > 0:
        FNAC    = networks.nacelle_diameter / Units.ft
        DIH     = vehicle.wings['main_wing'].dihedral
        YEE     = np.max(np.abs(np.array(networks.origin)[:, 1])) / Units.ft
        WF      = vehicle.fuselages['fuselage'].width / Units.ft
        XMLG    = 12 * FNAC + (0.26 - np.tan(DIH)) * (YEE - 6 * WF)  # length of extended main landing gear
    else:
        XMLG    = 0.75 * vehicle.fuselages['fuselage'].lengths.total / Units.ft  # length of extended nose landing gear
    XNLG = 0.7 * XMLG
    WLGM = (0.0117 - 0.0012 * DFTE) * WLDG ** 0.95 * XMLG ** 0.43
    WLGN = (0.048 - 0.0080 * DFTE) * WLDG ** 0.67 * XNLG ** 0.43 * (1 + 0.8 * CARBAS)

    output      = Data()
    output.main = WLGM * Units.lbs
    output.nose = WLGN * Units.lbs
    return output
