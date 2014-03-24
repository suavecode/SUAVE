""" evaluate_PASS.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import SUAVE.Methods.Units
import copy

from SUAVE.Structure            import Data
from SUAVE.Attributes.Results   import Result, Segment
# from SUAVE.Methods.Utilities    import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def evaluate_PASS(vehicle,mission):
    """ Compute the Pass Performance Calculations using SU AA 241 course notes 
    """

    # unpack
    maxto = vehicle.Mass.mtow
    mzfw_ratio = vehicle.Mass.fmzfw
    sref = vehicle.Wing['Main Wing'].sref
    sfc_sfcref = vehicle.Turbo_Fan['TheTurboFan'].sfc_TF
    sls_thrust = vehicle.Turbo_Fan['TheTurboFan'].thrust_sls
    eng_type = vehicle.Turbo_Fan['TheTurboFan'].type_engine
    out = Data()

    # Calculate
    fuel_manuever = WeightManeuver(maxto)
    fuel_climb_added = WeightClimbAdded(vehicle,mission,maxto)
    reserve_fuel = FuelReserve(mission,maxto,mzfw_ratio,0)
    out.range,fuel_cruise = CruiseRange(vehicle,mission,maxto,fuel_burn_maneuever,fuel_climb_added,reserve_fuel,sfc_sfcref,sls_thrust,eng_type,mzfw_ratio)
    out.fuel_burn = (2 * fuel_manuever) + fuel_climb_added + fuel_cruise
    out.tofl = TOFL(vehicle,mission,maxto,sref,sfc_sfcref,sls_thrust,eng_type)
    out.climb_grad = ClimbGradient(vehicle,mission,maxto,sfc_sfcref,sls_thrust,eng_type)
    out.lfl = LFL(vehicle,mission,maxto,mzfw_ratio,fuel_burn_maneuever,reserve_fuel,sref,sfc_sfcref,sls_thrust,eng_type)

    return out
