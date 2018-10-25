## @ingroup Sizing
#size_from_PGM.py

# Created : Jun 2016, M. Vegh
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np

from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import fuselage_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion import compute_turbofan_geometry
from SUAVE.Methods.Propulsion import turbofan_sizing
from SUAVE.Methods.Propulsion import turbojet_sizing



# ----------------------------------------------------------------------
#  Size from PGM
# ----------------------------------------------------------------------

def size_from_PGM(vehicle):
        """Completes the sizing of a SUAVE vehicle to determine fill out all of the dimensions of the vehicle.
           This takes in the vehicle as it is provided from the PGM analysis
    
            Assumptions:
            Simple tapered wing (no sections)
            Simple fuselage  (no sections)
    
            Source:
            N/A
    
            Inputs:
            vehicle    [SUAVE Vehicle]
    
            Outputs:
            vehicle    [SUAVE Vehicle]
    
            Properties Used:
            None
        """        
        
        # The top level info
        vehicle.systems.control        = "fully powered" 
        vehicle.systems.accessories    = "medium range"        
        
        # Passengers
        vehicle.passengers  = vehicle.performance.vector[-1] *1.
        
        # Size the wings
        max_area = 0
        for wing in vehicle.wings:
                
                # Use existing scripts
                wing = wing_planform(wing)
                
                # Get the max area
                if wing.areas.reference>max_area:
                        max_area = wing.areas.reference
        
        # Size the fuselage
        for fuse in vehicle.fuselages:
                
                 # Use existing scripts
                fuse = fuselage_planform(fuse)
        
        # Size the propulsion system
        for prop in vehicle.propulsors:
                if prop.tag == 'Turbofan':
                        conditions = None
                        compute_turbofan_geometry(prop, conditions)
                        turbofan_sizing(prop,mach_number = 0.1, altitude = 0., delta_isa = 0)
                if prop.tag == 'Turbojet':
                        turbojet_sizing(prop,mach_number = 0.1, altitude = 0., delta_isa = 0)                        

        # Vehicle reference area
        try:
                area = vehicle.wings.main_wing.areas.reference
        except:
                area = max_area
        
        vehicle.reference_area = area    
    
    
        return vehicle