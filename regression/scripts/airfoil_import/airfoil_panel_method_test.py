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
    ospath    = os.path.abspath(__file__)
    separator = os.path.sep
    rel_path  = ospath.split('airfoil_import' + separator + 'airfoil_import_test.py')[0] + 'Vehicles' + separator
    airfoil_polar_names  =  [[rel_path + 'NACA_4412_polar_Re_50000.txt',
                              rel_path + 'NACA_4412_polar_Re_100000.txt',
                              rel_path + 'NACA_4412_polar_Re_200000.txt',
                              rel_path + 'NACA_4412_polar_Re_500000.txt',
                              rel_path + 'NACA_4412_polar_Re_1000000.txt']]   
    airfoil_polar_data     =  import_airfoil_polars(airfoil_polar_names) 

    airfoil_geometry_names = [rel_path + 'NACA_4412.txt','airfoil_geometry_2.txt', 'airfoil_geometry_2-selig.txt']    
    
    
    npanel = 100
    AoA      = 3*Units.degrees
    airfoil_geometry_data  = import_airfoil_geometry(airfoil_geometry_names,npoints=npanel) 
    x = airfoil_geometry_data .x_coordinates[0]  # x: x co-ordinates of the panels centers
    y = airfoil_geometry_data .y_coordinates[0]   # y: y co-ordinate of the panel centers     
    airfoil_analysis(x,y,AoA,npoints=npanel)
   
    return  