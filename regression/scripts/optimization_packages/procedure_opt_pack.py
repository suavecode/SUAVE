# procedure_opt_pack.py
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
    
    obj = np.array([x2**2 + x1**2]) 
    nexus.obj = obj
    
    return nexus