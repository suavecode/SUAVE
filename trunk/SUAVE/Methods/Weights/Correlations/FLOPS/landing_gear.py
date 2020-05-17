from SUAVE.Core import Units, Data
import numpy as np


# The following assumptions are made
# 1) No fighter jet is sized, change DFTE to 1 for a fighter jet
# 2) Size aircraft is not meant for carrier operations, change CARBAS to 1 for carrier-based aircraft

def landing_gear_FLOPS(vehicle):
    DFTE = 0
    CARBAS = 0
    if vehicle.systems.accessories == "sst":
        RFACT = 0.00009
    else:
        RFACT = 0.00004
    DESRNG = vehicle.design_range / Units.nmi  # Design range in nautical miles
    WLDG = vehicle.mass_properties.max_takeoff / Units.lbs * (1 - RFACT * DESRNG)

    propulsor_name = list(vehicle.propulsors.keys())[0]  # obtain the key for the propulsor for assignment purposes
    propulsors = vehicle.propulsors[propulsor_name]
    if sum(propulsors.wing_mounted) > 0:
        FNAC = propulsors.nacelle_diameter / Units.ft
        DIH = vehicle.wings['main_wing'].dihedral
        YEE = np.max(np.abs(np.array(propulsors.origin)[:, 1])) / Units.ft
        WF = vehicle.fuselages['fuselage'].width / Units.ft
        XMLG = 12 * FNAC + (0.26 - np.tan(DIH)) * (YEE - 6 * WF)
    else:
        XMLG = 0.75 * vehicle.fuselages['fuselage'].lengths.total / Units.ft
    XNLG = 0.7 * XMLG

    WLGM = (0.0117 - 0.0012 * DFTE) * WLDG ** 0.95 * XMLG ** 0.43
    WLGN = (0.048 - 0.0080 * DFTE) * WLDG ** 0.67 * XNLG ** 0.43 * (1 + 0.8 * CARBAS)
    output = Data()
    output.main = WLGM * Units.lbs
    output.nose = WLGN * Units.lbs
    return output
