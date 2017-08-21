## @ingroup Methods-Missions-Segments
# converge_root.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import scipy.optimize
import autograd.numpy as np
import autograd

from SUAVE.Core.Arrays import array_type
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Converge Root
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments
def converge_root(segment,state):
    """Interfaces the mission to a numerical solver. The solver may be changed by using root_finder.
    Assumptions:
    N/A
    Source:
    N/A
    Inputs:
    state.unknowns                     [Data]
    segment                            [Data]
    state                              [Data]
    segment.settings.root_finder       [Data]
    state.numerics.tolerance_solution  [Unitless]
    Outputs:
    state.unknowns                     [Any]
    segment.state.numerics.converged   [Unitless]
    Properties Used:
    N/A
    """       
    
    unknowns = state.unknowns.pack_array()
    
    try:
        root_finder = segment.settings.root_finder
    except AttributeError:
        root_finder = scipy.optimize.fsolve 
    
    unknowns,infodict,ier,msg = root_finder( iterate,
                                         unknowns,
                                         args = [segment,state],
                                         xtol = state.numerics.tolerance_solution,
                                         full_output=1)

    if ier!=1:
        print "Segment did not converge. Segment Tag: " + segment.tag
        print "Error Message:\n" + msg
        segment.state.numerics.converged = False
    else:
        segment.state.numerics.converged = True
         
                            
    return
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments
def iterate(unknowns,(segment,state)):
    
    """Runs one iteration of of all analyses for the mission.
    Assumptions:
    N/A
    Source:
    N/A
    Inputs:
    state.unknowns                [Data]
    segment.process.iterate       [Data]
    Outputs:
    residuals                     [Unitless]
    Properties Used:
    N/A
    """       

    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    segment.process.iterate(segment,state)
    
    if type(state.residuals.forces) == autograd.numpy.numpy_extra.ArrayNode:
        grad = True
    else:
        grad = False
    
    if grad == False:
        residuals = state.residuals.pack_array()
    else:
        residuals = pack_autograd(state.residuals)
        
    return residuals 

def pack_autograd(s_residuals):
    
    # We are going to loop through the dictionary recursively and unpack
    
    dic = s_residuals
    pack_autograd.array = np.array([])
    
    def pack(dic):
        for key in dic.iterkeys():
            if isinstance(dic[key],Data):
                pack(dic[key]) # Regression
                continue
            #elif np.rank(dic[key])>2: continue
            elif isinstance(dic[key],str):continue
            
            pack_autograd.array = np.append(pack_autograd.array,dic[key])
            
            
    pack(dic)
    residuals = pack_autograd.array 
    
    return residuals