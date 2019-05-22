## @ingroup Methods-Missions-Segments
# converge_root.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           May 2019, E. Botero

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
def converge_root(segment):
    """Interfaces the mission to a numerical solver. The solver may be changed by using root_finder.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment                            [Data]
    segment.settings.root_finder       [Data]
    state.numerics.tolerance_solution  [Unitless]

    Outputs:
    state.unknowns                     [Any]
    segment.state.numerics.converged   [Unitless]
    segment.state.normalization_factor [Array]

    Properties Used:
    N/A
    """       
    
    unknowns = segment.state.unknowns.pack_array()
    
    # Find the normalization factors
    if segment.settings.normalize == True:
        segment.state.unknowns_normalization_factor  = unknowns*1.
        segment.state.unknowns_normalization_factor[segment.state.unknowns_normalization_factor==0] = 1e-16 
    else:
        segment.state.unknowns_normalization_factor  = np.ones_like(unknowns)

    # Run one iteration to get the scaling
    if segment.settings.normalize == True:
        segment.process.iterate(segment)
        segment.state.residual_normalization_factor = 1*segment.state.residuals.pack_array() 
        segment.state.residual_normalization_factor[segment.state.residual_normalization_factor==0] = 1e-16
    else:
        segment.state.residual_normalization_factor = np.ones_like(segment.state.residuals.pack_array())
    
    # Normalize the unknowns
    unknowns = unknowns/segment.state.unknowns_normalization_factor
    
    
    try:
        root_finder = segment.settings.root_finder
    except AttributeError:
        root_finder = scipy.optimize.fsolve 
    
    unknowns,infodict,ier,msg = root_finder( iterate,
                                         unknowns,
                                         args = segment,
                                         xtol = segment.state.numerics.tolerance_solution,
                                         full_output=1)

    if ier!=1:
        print("Segment did not converge. Segment Tag: " + segment.tag)
        print("Error Message:\n" + msg)
        segment.state.numerics.converged = False
    else:
        segment.state.numerics.converged = True
         
                            
    return
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments
def iterate(unknowns, segment):
    
    """Runs one iteration of of all analyses for the mission.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.unknowns                     [Data]
    segment.process.iterate            [Data]
    segment.state.normalization_factor [array]

    Outputs:
    residuals                     [Unitless]

    Properties Used:
    N/A
    """       
    
    unknowns_normal = segment.state.unknowns_normalization_factor * unknowns
    
    if isinstance(unknowns_normal,array_type):
        segment.state.unknowns.unpack_array(unknowns_normal)
    else:
        segment.state.unknowns_normal = unknowns_normal
        
    segment.process.iterate(segment)
    
    residuals = segment.state.residuals.pack_array()
    
    residuals_normalized = residuals/segment.state.residual_normalization_factor
        
    return residuals_normalized 