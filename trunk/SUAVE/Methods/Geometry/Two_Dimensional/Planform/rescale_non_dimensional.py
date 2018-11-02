## @ingroup Planform
#rescale_non_dimensional.py

# Created : Jun 2016, M. Vegh
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np


# ----------------------------------------------------------------------
#  Set Origin Non-Dimensional
# ----------------------------------------------------------------------

def set_origin_non_dimensional(vehicle):
    """ Places the origin of all components 

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        vehicle    [SUAVE Vehicle]

        Outputs:
        vehicle    [SUAVE Vehicle]

        Properties Used:
        None
    """        

    for wing in vehicle.wings:
        origin  = wing.origin
        b       = wing.spans.projected
        non_dim = origin/b
        
        wing.non_dimensional_origin = non_dim
    
    for fuse in vehicle.fuselages:
        origin  = fuse.origin
        length  = fuse.lengths.total
        non_dim = origin/length
        
        fuse.non_dimensional_origin = non_dim
        
    
        
    return vehicle


# ----------------------------------------------------------------------
#  Scale to Non-Dimensional
# ----------------------------------------------------------------------

def set_origin_dimensional(vehicle):
    """ Places the origin of all components 

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        vehicle    [SUAVE Vehicle]

        Outputs:
        vehicle    [SUAVE Vehicle]

        Properties Used:
        None
    """    

    for wing in vehicle.wings:
        non_dim = wing.non_dimensional_origin
        b       = wing.spans.projected
        origin  = non_dim*b
        
        wing.origin = origin
    
    for fuse in vehicle.fuselages:
        non_dim = fuse.non_dimensional_origin
        length  = fuse.lengths.total
        origin  = non_dim*length
        
        fuse.origin = origin
        
    return vehicle