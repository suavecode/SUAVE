# test_cnbeta.py
# Created:  Apr 2014 T. Momose
# Modified: Feb 2017, M. Vegh
# Reference: Aircraft Dynamics: from Modeling to Simulation, by M. R. Napolitano

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cnbeta import taw_cnbeta
from SUAVE.Core import (
    Data, Container,
)

import sys
sys.path.append('../Vehicles')


def main():
    #only do calculation for 747
    from Boeing_747 import vehicle_setup, configs_setup
    vehicle = vehicle_setup()
    configs = configs_setup(vehicle)
    
    Mach                          = np.array([0.198])
    
    segment                              = SUAVE.Analyses.Mission.Segments.Segment()
    segment.freestream                   = Data()
    segment.freestream.mach_number       = Mach
    segment.atmosphere                   = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    altitude                             = 0.0 * Units.feet
    conditions                           = segment.atmosphere.compute_values(altitude / Units.km)
    segment.a                            = conditions.speed_of_sound
    segment.freestream.density           = conditions.density
    segment.freestream.dynamic_viscosity = conditions.dynamic_viscosity
    segment.freestream.velocity          = segment.freestream.mach_number * segment.a
  
    
    #Method Test
    cn_b = taw_cnbeta(vehicle,segment,configs.base)
    expected = 0.09596976 # Should be 0.184
    error = Data()
    error.cn_b_747 = (cn_b-expected)/expected

  
  
    
    print(error)
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)

    return

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
