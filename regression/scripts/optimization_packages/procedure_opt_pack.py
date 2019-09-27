# Procedure.py
# 
# Created:  Sep. 2019, M. Clarke
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
import copy
from SUAVE.Analyses.Process import Process


# ----------------------------------------------------------------------        
#   Setup
# ----------------------------------------------------------------------   

def setup():
    
    # ------------------------------------------------------------------
    #   Analysis Procedure
    # ------------------------------------------------------------------ 
    
    # size the base config
    procedure = Process()

    # post process the results
    procedure.post_process = post_process
        
    # done!
    return procedure

# ----------------------------------------------------------------------        
#   Design Mission
# ----------------------------------------------------------------------    
def design_mission(nexus):
    
    mission = nexus.missions.base
    results = nexus.results
    results.base = mission.evaluate()
    
    return nexus

# ----------------------------------------------------------------------        
#   Analysis Setting
# ----------------------------------------------------------------------   

def post_process(nexus):
    
    x1 = nexus.vehicle_configurations.base.x1
    x2 = nexus.vehicle_configurations.base.x2
    
    obj = np.array([x2**2+x1**2]) 
    
    # for differential evolution, a penalty function is needed if constraint is violated
    if nexus.solver_name == 'differential_evolution':
         
        err1 = x1 + 10
        err2 = x2 - 1
        if err1 < 0.001:
            obj = obj + abs(err1)*np.exp(5)
        if err2 < 0.001:
            obj = obj + abs(err2)*np.exp(5) 
    nexus.obj = obj
    
    return nexus