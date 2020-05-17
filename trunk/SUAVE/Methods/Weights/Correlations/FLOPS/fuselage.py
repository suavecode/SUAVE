from SUAVE.Core import Units, Data
import numpy as np
import SUAVE

# The following assumption are made:
# 1) NFUSE = 1, only one fuselage, please change the variable appropriately for multiple fuselages
# 2) delta_isa = 0, for pressure calculations

def fuselage_weight_FLOPS(vehicle):
    XL = vehicle.fuselages['fuselage'].lengths.total / Units.ft  # Fuselage length, ft
    DAV = (vehicle.fuselages['fuselage'].width +
           vehicle.fuselages['fuselage'].heights.maximum) / 2. * 1 / Units.ft  # Average fuselage diameter, ft
    if vehicle.systems.accessories == "short-range" or vehicle.systems.accessories == "commuter":
        SWFUS = np.pi * (XL / DAV - 1.7) * DAV ** 2  # Fuselage wetted area, ft**2
        ULF = vehicle.envelope.ultimate_load  # Ultimate load factor
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmo_data = atmosphere.compute_values(vehicle.design_cruise_alt, 0)
        atmo_data_floor = atmosphere.compute_values(0, 0)
        DELTA = atmo_data.pressure/atmo_data_floor.pressure
        QCRUS = 1481.35 * DELTA * vehicle.design_mach_number**2  # Cruise dynamic pressure, psf
        DG = vehicle.mass_properties.max_takeoff / Units.lbs  # Design gross weight in lb
        WFUSE = 0.052 * SWFUS ** 1.086 * (ULF * DG) ** 0.177 * QCRUS ** 0.241
    else:
        propulsor_name = list(vehicle.propulsors.keys())[0]
        propulsors = vehicle.propulsors[propulsor_name]
        FNEF = len(propulsors.wing_mounted) - sum(propulsors.wing_mounted)   # Number of fuselage mounted engines
        if vehicle.systems.accessories == 'cargo':
            CARGF = 1
        else:
            CARGF = 0  # Cargo aircraft floor factor [0 for passenger transport, 1 for cargo transport
        NFUSE = 1  # Number of fuselages
        WFUSE = 1.35 * (XL * DAV) ** 1.28 * (1 + 0.05 * FNEF) * (1 + 0.38 * CARGF) * NFUSE
    return WFUSE * Units.lbs
