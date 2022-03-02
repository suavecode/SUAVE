## @ingroup Planform
#rescale_non_dimensional.py

# Created : May 2020, E. Botero
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
    """ Places the origin of all major components in a 
    non-dimensional fashion. This is useful for optimization or
    generative design 

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        vehicle    [SUAVE Vehicle]
              .fuselages.*.origin
              .fuselages.fuselage.lengths.total
              .wings.*.origin
              .wings.main_wing.lengths.total
              .networks.*.origin

        Outputs:
        vehicle    [SUAVE Vehicle]
              .fuselages.*.non_dimensional_origin
              .wings.*.non_dimensional_origin
              .networks.*.non_dimensional_origin

        Properties Used:
        None
    """        
    
    
    try:
        length_scale = vehicle.fuselages.fuselage.lengths.total
    except:
        try:
            length_scale = vehicle.wings.main_wing.lengths.total
        except:
            length_scale = 1.

    for wing in vehicle.wings:
        origin  = wing.origin
        non_dim = np.array(origin)/length_scale
        
        wing.non_dimensional_origin = non_dim.tolist()
    
    for fuse in vehicle.fuselages:
        origin  = fuse.origin
        non_dim = np.array(origin)/length_scale
        
        fuse.non_dimensional_origin = non_dim.tolist()  

    for prop in vehicle.networks:
        origins  = prop.origin
        prop.non_dimensional_origin.clear()
        for eng in range(int(prop.number_of_engines)):
            origin = np.array(origins[eng])/length_scale
            prop.non_dimensional_origin.append(origin.tolist())       
    
        
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
    
    try:
        length_scale = vehicle.fuselages.fuselage.lengths.total
    except:
        try:
            length_scale = vehicle.wings.main_wing.lengths.total
        except:
            length_scale = 1.

    for wing in vehicle.wings:
        non_dim = wing.non_dimensional_origin
        origin  = np.array(non_dim)*length_scale
        
        wing.origin = origin.tolist()
    
    for fuse in vehicle.fuselages:
        non_dim = fuse.non_dimensional_origin
        origin  = np.array(non_dim)*length_scale
        
        fuse.origin = origin.tolist()
                
    for net in vehicle.networks:
        n = int(net.number_of_engines)
        non_dims  = net.non_dimensional_origin
        
        net.origin.clear()
        
        origin = np.zeros((n,3))
    
        for eng in range(0,n):
            origin[eng,:] = np.array(non_dims[0])*length_scale
            
            if eng % 2 != 0:
                origin[eng,1] = -origin[eng,1]
                
            elif (eng % 2 == 0) and (eng == n-1):
                
                origin[eng,1] = 0.

        net.origin = origin.tolist()  
        
    return vehicle