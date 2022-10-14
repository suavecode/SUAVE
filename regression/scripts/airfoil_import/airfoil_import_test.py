# airfoil_import_test.py
# 
# Created:  
# Modified: Sep 2020, M. Clarke 
#           May 2021, R. Erhard

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------

import SUAVE 
from SUAVE.Core import Units, Data 
from SUAVE.Plots.Geometry import plot_airfoil
import matplotlib.pyplot as plt  
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars \
     import import_airfoil_polars
from SUAVE.Plots.Performance.Airfoil_Plots import *
import os
import numpy as np

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():    
    # ----------------------------------------------------------------------------------------------------------------
    #  Define airfoil geometry and polar files 
    # ----------------------------------------------------------------------------------------------------------------    
    ospath    = os.path.abspath(__file__)
    separator = os.path.sep 
    rel_path  = ospath.split('airfoil_import' + separator + 'airfoil_import_test.py')[0] + 'Vehicles' + separator + 'Airfoils' + separator
    airfoil_geometry_with_selig =  [rel_path + 'NACA_4412.txt','airfoil_geometry_2.txt', 'airfoil_geometry_2-selig.txt']        
    airfoil_geometry_files      = [rel_path + 'NACA_4412.txt']
    airfoil_polar_files         =  [[rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_50000.txt',
                                     rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_100000.txt',
                                     rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_200000.txt',
                                     rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_500000.txt',
                                     rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_1000000.txt']]   
    
    

    # ----------------------------------------------------------------------------------------------------------------
    #  Plot polar files 
    # ----------------------------------------------------------------------------------------------------------------    
    # plot airfoil polar data with and without surrogate
    airfoil_geometry_1   = import_airfoil_geometry(airfoil_geometry_files)  
    airfoil_polar_data_1 = compute_airfoil_properties(airfoil_geometry_1,airfoil_polar_files) 
    plot_airfoil_polar_files(airfoil_polar_data_1)
    plot_airfoil_polar_files(airfoil_polar_data_1,use_surrogate=True , save_filename = "Airfoil_Polars_With_Surrogate")

    # ----------------------------------------------------------------------------------------------------------------
    #  
    # ----------------------------------------------------------------------------------------------------------------
    airfoil_geometry_2  = import_airfoil_geometry(airfoil_geometry_with_selig)

    # Actual t/c values  
    airfoil_tc_actual = [0.12031526401402462, 0.11177619218206997, 0.11177619218206997] 

    # Check t/c calculation against previously calculated values 
    for i in range(0, len(airfoil_geometry_with_selig)):
        assert(np.abs(airfoil_tc_actual[i]-airfoil_geometry_2.thickness_to_chord[i]) < 1E-8 ) 

    # Check that camber line comes back the same for the Lednicer and Selig formats 
    for j in range(0, len(airfoil_geometry_2.camber_coordinates[1])):
        assert( np.abs(airfoil_geometry_2.camber_coordinates[1][j] - airfoil_geometry_2.camber_coordinates[2][j]) < 1E-8 )

    plot_airfoil(airfoil_geometry_with_selig)
    
    
    return  

if __name__ == '__main__': 
    main() 
    plt.show()
