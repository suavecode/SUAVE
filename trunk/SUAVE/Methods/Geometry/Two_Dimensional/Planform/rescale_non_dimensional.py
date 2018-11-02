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
        non_dim = np.array(origin)/b
        
        wing.non_dimensional_origin = non_dim.tolist()
    
    for fuse in vehicle.fuselages:
        origin  = fuse.origin
        length  = fuse.lengths.total
        non_dim = np.array(origin)/length
        
        fuse.non_dimensional_origin = non_dim.tolist()
        
    
        
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
        origin  = np.array(non_dim)*b
        
        wing.origin = origin.tolist()
    
    for fuse in vehicle.fuselages:
        non_dim = fuse.non_dimensional_origin
        length  = fuse.lengths.total
        origin  = np.array(non_dim)*length
        
        fuse.origin = origin.tolist()
        
    return vehicle