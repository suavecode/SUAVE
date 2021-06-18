## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Propulsion
# compute_propeller_thickness.py
# 
# Created:  Jun 2021, R. Erhard
# Modified: 
#           

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data , Units
import numpy as np
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry 


def compute_propeller_thickness(propeller):
    """This computes the thickness to chord ratio at each radial station of a propeller 
    given the airfoil geometries and airfoil locations.
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:
    propeller.
          a_geos                 <string>
          a_stations             <string>
           
    Outputs:
    propeller.
          thickness_to_chord                  [unitless]
          max_thickness_distribution          [m]      
    
    Properties Used:
    N/A
    """  
    
    # extract propeller parameters 
    a_geos       = propeller.airfoil_geometry
    a_stations   = propeller.airfoil_polar_stations
    c            = propeller.chord_distribution
    
    # import airfoil geometry
    airfoil_data = import_airfoil_geometry(a_geos)
    max_t_c      = airfoil_data.thickness_to_chord
    
    t_c_distribution = np.zeros(len(a_stations))
    for a in range(len(a_geos)):
        t_c_distribution[np.array(a_stations)==a] = max_t_c[a]
        
    
    propeller.thickness_to_chord = t_c_distribution
    propeller.max_thickness_distribution = t_c_distribution*c
    
    
    return 
   
   