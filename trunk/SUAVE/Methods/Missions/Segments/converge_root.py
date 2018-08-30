## @ingroup Methods-Missions-Segments
# converge_root.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import scipy.optimize
import numpy as np

from SUAVE.Core.Arrays import array_type

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
    
    residuals = state.residuals.pack_array()
        
    return residuals 