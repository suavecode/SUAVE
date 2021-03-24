# airfoil_panel_method_test.py
# 
# Created:  
# Modified: Mar 2021, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------

import SUAVE 
import os
from SUAVE.Core import Units, Data 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis import airfoil_analysis
import matplotlib.pyplot as plt  
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars \
     import import_airfoil_polars
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series \
     import  compute_naca_4series

import numpy as np

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
    ospath    = os.path.abspath(__file__)
    separator = os.path.sep
    rel_path  = ospath.split('airfoil_import' + separator + 'airfoil_panel_method_test.py')[0] + 'Vehicles' + separator
    airfoil_geometry_names = [rel_path + 'NACA_4412.txt']     
    
    npanel = 100
    AoA    = 3*Units.degrees
    
    # NACA 4 series 
    airfoil_geometry_data  = compute_naca_4series(0.03,0.3,0.1,npoints=npanel )
    x = np.delete( airfoil_geometry_data.x_coordinates[0][::-1], int(npanel/2)) 
    y = np.delete( airfoil_geometry_data.y_coordinates[0][::-1], int(npanel/2))   
    airfoil_analysis(x,y,AoA,npanel)
   
    #airfoil_geometry_data  = import_airfoil_geometry(airfoil_geometry_names,npoints=npanel)  
    return  

if __name__ == '__main__': 
    main() 
    plt.show()