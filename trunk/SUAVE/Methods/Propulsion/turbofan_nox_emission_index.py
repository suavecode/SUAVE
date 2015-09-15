# test_gasturbine_network.py
# 
# Created:  Michael Vegh, September 2015
#correlation taken from Antione, Nicholas, Aircraft Optimization for Minimal Environmental Impact, pp. 31 (PhD Thesis)
#based on NASA's Engine Performance Program (NEPP)
# Modified: 
#        

""" create and evaluate a gas turbine network
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
import datetime
import time
from SUAVE.Core import Units, Data





def turbofan_nox_emission_index(turbofan, state):
    results=turbofan(state)
    p3=turbofan.combustor.inputs.stagnation_pressure
    T3=turbofan.combustor.inputs.stagnation_temperature
    T4=turbofan.combustor.outputs.stagnation_temperature
    p3=p3/Units.psi
    T3=T3/Units.degR #convert to Rankine
    T4=T4/Units.degR
    nox_emission_index=.004194*T4*((p3/439.)**.37)*np.exp((T3-1471.)/345.)
    nox_emission_index=nox_emission_index*(Units.g/Units.kg) #correlation in g Nox/kg fuel; convert to kg Nox/kg
    return nox_emission_index
    #now use their correlation
    
    
    
   