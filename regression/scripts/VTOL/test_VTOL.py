# test_VTOL.py
# 
# Created:  July 2018, M. Clarke
#
""" setup file for a mission with a QuadShot
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container,
)

import sys

sys.path.append('../Vehicles')
# the analysis functions

from QuadShot import vehicle_setup, configs_setup


sys.path.append('../VTOL')
# the analysis functions


from mission_QuadShot import vehicle_setup, configs_setup, analyses_setup, mission_setup, missions_setup
import copy

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main(): 

    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    ## ------------------------------------------------------------------
    ##  Weights
    #weights = SUAVE.Analyses.Weights.Weights()
    #weights.settings.empty_weight_method = \
        #SUAVE.Methods.Weights.Correlations.UAV.empty
    #weights.vehicle = vehicle
    #analyses.append(weights)
    
    ## ------------------------------------------------------------------
    ##  Aerodynamics Analysis
    #aerodynamics                                  = SUAVE.Analyses.Aerodynamics.AERODAS()
    #aerodynamics.geometry                         = copy.deepcopy(configs.base)   
    #aerodynamics.settings.drag_coefficient_increment = 0.0000
    #aerodynamics.settings.maximum_lift_coefficient   = 1.5    
    #configs_analyses.base.append(aerodynamics)     

    ## ------------------------------------------------------------------
    ##  Energy
    #energy = SUAVE.Analyses.Energy.Energy()
    #energy.network = vehicle.propulsors
    #analyses.append(energy)
    
    ## ------------------------------------------------------------------
    ##  Planet Analysis
    #planet = SUAVE.Analyses.Planets.Planet()
    #analyses.append(planet)
    
    ## ------------------------------------------------------------------
    ##  Atmosphere Analysis
    #atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    #atmosphere.features.planet = planet.features
    #analyses.append(atmosphere)       
    
    # ------------------------------------------------------------------        
    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    configs.finalize()
    analyses.finalize()

    # mission analysis
    mission = analyses.missions.base    
    results = mission.evaluate()

    # lift coefficient check
    lift_coefficient              = results.conditions.cruise.aerodynamics.lift_coefficient[0]
    lift_coefficient_true         = 0.59495841
    print lift_coefficient
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print 'CL difference'
    print diff_CL
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3

    # moment coefficient check
    moment_coefficient            = results.conditions.cruise.stability.static.CM[0][0]
    moment_coefficient_true       = -0.620326644
    print moment_coefficient
    diff_CM                       = np.abs(moment_coefficient - moment_coefficient_true)
    print 'CM difference'
    print diff_CM
    assert np.abs((moment_coefficient - moment_coefficient_true)/moment_coefficient_true) < 1e-3    

    return

if __name__ == '__main__': 
    main()    
