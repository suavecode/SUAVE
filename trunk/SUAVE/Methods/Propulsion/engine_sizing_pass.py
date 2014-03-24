""" Propulsion.py: Methods for Propulsion Analysis """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data
from SUAVE.Attributes import Constants
# import SUAVE

# ----------------------------------------------------------------------
#  Mission Methods
# ----------------------------------------------------------------------

def engine_sizing_pass(Turbofan,State):

    #sls_thrust=Turbofan.sls_thrust
    
    ''' outputs = engine_dia,engine_length,nacelle_dia,inlet_length,eng_area,inlet_area
        inputs  engine related : sls_thrust - basic pass input

  '''    
    #unpack inputs
    
    sls_thrust=Turbofan.thrust_sls
    
    
    #calculate

    engine_dia=1.0827*sls_thrust**0.4134
    engine_length=2.4077*sls_thrust**0.3876
    nacelle_dia=1.1*engine_dia
    inlet_length=0.6*engine_dia
    eng_maxarea=3.14*0.25*engine_dia**2
    inlet_area=0.7*eng_maxarea

    
    
    
    #Pack results
    
    Turbofan.bare_engine_dia=engine_dia
    Turbofan.bare_engine_length=engine_length
    Turbofan.nacelle_dia=nacelle_dia
    Turbofan.inlet_length=inlet_length
    Turbofan.eng_maxarea=eng_maxarea
    Turbofan.inlet_area= inlet_area    
    
    #Vehicle.Turbofan.bare_engine_dia=engine_dia
    #Vehicle.Turbofan.bare_engine_length=engine_length
    #Vehicle.Turbofan.nacelle_dia=nacelle_dia
    #Vehicle.Turbofan.inlet_length=inlet_length
    #Vehicle.Turbofan.eng_maxarea=eng_maxarea
    #Vehicle.Turbofan.inlet_area= inlet.area



    #return (engine_dia,engine_length,nacelle_dia,inlet_length,eng_maxarea,inlet_area)



    #engine analysis based on TASOPT
    #constant Cp is not assumed 
    #pressure ratio prescribed
    #MAIN CODE