# pyopt_setup.py
#
# Created:  Jul 2015, E. Botero
# Modified: Feb 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import numpy as np
from SUAVE.Optimization import helper_functions as help_fun
from SUAVE.Optimization.Package_Setups.TRM_2 import Trust_Region_Optimization as tro
from SUAVE.Optimization.Package_Setups.TRM_2.Trust_Region import Trust_Region

# ----------------------------------------------------------------------
#  Pyopt_Solve
# ----------------------------------------------------------------------

def TRM_Solve(problem):
   
    tr = Trust_Region()
    problem.trust_region = tr
    TRM_opt = tro.Trust_Region_Optimization()
    TRM_opt.optimize(problem)
    
    return
