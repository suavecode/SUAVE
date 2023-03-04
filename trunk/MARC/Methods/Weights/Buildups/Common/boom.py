## @ingroup Methods-Weights-Buildups-Common

# boom.py
#
# Created:  Mar 2023, M. Clarke 

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import numpy as np


#-------------------------------------------------------------------------------
# Fuselage
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Common
def boom(boom,
             maximum_g_load = 3.8,
             safety_factor = 1.5):
    """ Calculates the structural mass of a boom for an eVTOL vehicle, 
        
        Assumptions: 
            Assumes cylindrical boom
        Sources: 

        Inputs:  

        Outputs: 
            weight:                 Estimated Boom Mass             [kg]
        
        Properties Used:
        Material Properties of Imported MARC Solids
    """

    #-------------------------------------------------------------------------------
    # Unpack Inputs
    #------------------------------------------------------------------------------- 
    bLength = boom.lengths.total
    bHeight = boom.heights.maximum 

    #-------------------------------------------------------------------------------
    # Unpack Material Properties
    #-------------------------------------------------------------------------------   
    density     = 1759   # a typical density of carbon fiber is 
    thickness   = 0.01  # thicness of boom is 1 cm

    # Calculate boom area assuming it is a hollow cylinder
    S_wet  = 2* np.pi* (bHeight/2) *bLength + 2*np.pi*(bHeight/2)**2
    weight = S_wet *thickness* density 
    
    return weight