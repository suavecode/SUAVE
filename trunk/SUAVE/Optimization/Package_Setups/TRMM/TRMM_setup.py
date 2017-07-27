# TRMM_setup.py
#
# Created:  Apr 2017, T. MacDonald
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import numpy as np
from SUAVE.Optimization import helper_functions as help_fun
from SUAVE.Optimization.Package_Setups.TRMM import Trust_Region_Optimization as tro
from SUAVE.Optimization.Package_Setups.TRMM.Trust_Region import Trust_Region

# ----------------------------------------------------------------------
#  Pyopt_Solve
# ----------------------------------------------------------------------

def TRMM_Solve(problem,tr=None,tr_opt=None,print_output=False):

    if tr == None:
        tr = Trust_Region()
    problem.trust_region = tr
    if tr_opt == None:
        TRM_opt = tro.Trust_Region_Optimization()
    else:
        TRM_opt = tr_opt
    TRM_opt.optimize(problem,print_output=print_output)

    return
