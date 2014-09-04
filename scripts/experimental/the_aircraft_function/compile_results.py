# compile_results.py
# 
# Created:  Andrew Wendorff, Aug 2014
# Modified: 

""" Combining the aircraft outputs for the aircraft function
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Outputs
# ----------------------------------------------------------------------

def compile_results(vehicle,mission,results):
    
    # merge all segment conditions
    def stack_condition(a,b):
        if isinstance(a,np.ndarray):
            return np.vstack([a,b])
        else:
            return None

    conditions = None
    for segment in results.mission_profile.Segments:
        if conditions is None:
            conditions = segment.conditions
            continue
        conditions = conditions.do_recursive(stack_condition,segment.conditions)
      
    # pack
    results.output = Data()
    results.output.stability = Data()
    results.output.weight_empty = vehicle.Mass_Properties.operating_empty
    results.output.fuel_burn = max(conditions.weights.total_mass[:,0]) - min(conditions.weights.total_mass[:,0])
    #results.output.max_usable_fuel = vehicle.Mass_Properties.max_usable_fuel
    results.output.noise = results.noise    
    results.output.mission_time_min = max(conditions.frames.inertial.time[:,0] / Units.min)
    results.output.max_altitude_km = max(conditions.freestream.altitude[:,0] / Units.km)
    results.output.range_nmi = results.mission_profile.Segments[-1].conditions.frames.inertial.position_vector[-1,0] / Units.nmi
    results.output.field_length = results.field_length
    results.output.stability.cm_alpha = max(conditions.aerodynamics.cm_alpha[:,0])
    results.output.stability.cn_beta = max(conditions.aerodynamics.cn_beta[:,0])

    
    #TODO: revisit how this is calculated
    results.output.second_segment_climb_rate = results.mission_profile.Segments['Climb - 2'].climb_rate

    
    return results