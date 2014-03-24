""" normal_shock.py: compute pressure and temperature ratios across an ideal normal shock """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def normal_shock(g,M,result="all"):

    """  normal_shock(g,M,result="all"): compute pressure and temperature ratios across an ideal normal shock
    
         Inputs:    g = ratio of specific heats                                     (required)                                  (float)    
                    area_ratio = A/A*                                               (required)                                  (float)
                    branch = "subsonic" or "supersonic", which solution to return   (required, default = "supersonic")          (string)

         Outputs:   T_ratio = absolute temperature ratio                                                                        (float)
                    p_ratio = absolute pressure ratio                                                                           (float)
                    Tt_ratio = stagnation temperature ratio                                                                     (float)
                    pt_ratio = stagnation pressure ratio                                                                        (float)

    """

    raise NotImplementedError