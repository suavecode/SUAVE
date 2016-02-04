# converge_root.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# Scipy
import scipy
import scipy.optimize

from SUAVE.Core.Arrays import array_type
from autograd.numpy import np
from autograd import grad
from autograd.numpy.numpy_extra import ArrayNode

# ----------------------------------------------------------------------
#  Converge Root
# ----------------------------------------------------------------------

def converge_root(segment,state):
    
    unknowns = state.unknowns.pack_array()
    
    try:
        root_finder = segment.settings.root_finder
    except AttributeError:
        root_finder = scipy.optimize.fsolve 
        
    prime = grad(iterate)
    
    unknowns = root_finder( iterate,
                            unknowns,
                            args = [segment,state],
                            xtol = state.numerics.tolerance_solution,
                            fprime = prime)
    
    return
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
    
def iterate(unknowns,(segment,state)):
    
    autograd_array = ArrayNode

    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    elif isinstance(unknowns,ArrayNode):
        state.unknowns.unpack_array(unknowns)        
    else:
        state.unknowns = unknowns
        
    segment.process.iterate(segment,state)
    
    residuals = state.residuals.pack_array()
        
    return residuals 

