from SUAVE.Core import Units
import numpy as np
import SUAVE

def tail_vertical_FLOPS(vehicle, wing):
    DG = vehicle.mass_properties.max_takeoff / Units.lbs  # Design gross weight in lb
    TRVT = wing.taper
    NVERT = 1  # Number of vertical tails
    WVT = 0.32 * DG ** 0.3 * (TRVT + 0.5) * NVERT ** 0.7 * (wing.areas.reference/Units.ft**2)**0.85
    return WVT * Units.lbs

def tail_horizontal_FLOPS(vehicle, wing):
    SHT = wing.areas.reference / Units.ft **2
    DG = vehicle.mass_properties.max_takeoff / Units.lbs
    TRHT = wing.taper
    WHT = 0.53 * SHT * DG ** 0.2 * (TRHT + 0.5)

    return WHT * Units.lbs

