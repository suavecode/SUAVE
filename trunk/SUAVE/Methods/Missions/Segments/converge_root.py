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
from autograd.convenience_wrappers import elementwise_grad, multigrad
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
        
    prime = make_into_jacobian(elementwise_grad(iterate))
    
    unknowns = root_finder( iterate,
                            unknowns,
                            args = [segment,state],
                            xtol = state.numerics.tolerance_solution,
                            fprime = prime)

    return
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

def make_into_jacobian(fun):
    
    def output(*args, **kwargs):
        return np.diagflat(fun(*args, **kwargs))
    
    return output

def iterate(unknowns,(segment,state)):
    
    autograd_array = ArrayNode

    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    elif isinstance(unknowns,ArrayNode):
        state.unknowns = unpack_autograd(state.unknowns, unknowns)   
    else:
        state.unknowns = unknowns
        
    segment.process.iterate(segment,state)
    
    residuals = state.residuals.pack_array()
        
    return residuals 


def unpack_autograd(s_unkowns,unknowns):
    
    # We need to take the grad object and slice it into the dictionary
    
    # Find the number of keys in unknowns and divide them up
    n_keys   = len(s_unkowns.keys()) - 1
    size_vec = unknowns.size/n_keys
    count    = 0
    
    for key in s_unkowns.keys():

        if key is not 'tag':
            s_unkowns[key] = np.reshape(unknowns[count*size_vec:(count+1)*size_vec],(size_vec,1))
            count = count + 1
        
    return s_unkowns