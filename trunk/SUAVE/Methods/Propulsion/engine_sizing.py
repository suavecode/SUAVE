""" Propulsion.py: Methods for Propulsion Analysis """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data
from SUAVE.Attributes import Constants
# import SUAVE

# ----------------------------------------------------------------------
#  Mission Methods
# ----------------------------------------------------------------------

def engine_sizing(Turbofan,State):

    #sls_thrust=Turbofan.sls_thrust
    
    ''' outputs = engine_dia,engine_length,nacelle_dia,inlet_length,eng_area,inlet_area
        inputs  engine related : sls_thrust - basic pass input

  '''
    
    if Turbofan.analysis_type == 'pass' :
        engine_sizing_pass(Turbofan,State)
    elif Turbofan.analysis_type == '1D' :
        engine_sizing_1d(Turbofan,State)          