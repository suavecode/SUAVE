import SUAVE
from SUAVE.Core import Units, Data 
from SUAVE.Plots.Geometry_Plots import plot_airfoil
import matplotlib.pyplot as plt  
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars \
     import import_airfoil_polars

import numpy as np

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
    
    airfoil_polar_names    = ['airfoil_polar_1.txt','airfoil_polar_2.txt']    
    airfoil_polar_data =  import_airfoil_polars(airfoil_polar_names) 

    airfoil_geometry_names = ['airfoil_geometry_1.txt','airfoil_geometry_2.txt', 'airfoil_geometry_2-selig.txt']    
    airfoil_geometry_data = import_airfoil_geometry(airfoil_geometry_names)

    # Actual t/c values 

    airfoil_tc_actual = [0.11806510344827587, 0.11175207317073171, 0.11175207317073171]


    # Check t/c calculation against previously calculated values

    for i in range(0, len(airfoil_geometry_names)):
        assert(np.abs(airfoil_tc_actual[i]-airfoil_geometry_data.thickness_to_chord[i]) < 1E-8 )
    

    # Check that camber line comes back the same for the Lednicer and Selig formats

    for j in range(0, len(airfoil_geometry_data.camber_coordinates[1])):
         assert( np.abs(airfoil_geometry_data.camber_coordinates[1][j] - airfoil_geometry_data.camber_coordinates[2][j]) < 1E-8 )

    plot_airfoil(airfoil_geometry_names)
    return  

if __name__ == '__main__': 
    main() 
    plt.show()
