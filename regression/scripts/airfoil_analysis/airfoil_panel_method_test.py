# airfoil_panel_method_test.py
# 
# Created:  
# Modified: Mar 2021, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
from SUAVE.Core import Units
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis      import airfoil_analysis 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series      import  compute_naca_4series
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry   import import_airfoil_geometry
from SUAVE.Plots.Performance.Airfoil_Plots import *

# package imports 
import os 
import numpy as np
import matplotlib.pyplot as plt   

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():    
    # ----------------------------------------------------------------------
    # CASE 1: 2 airfoils with target alpha in inviscid mode 
    # ----------------------------------------------------------------------
    AoA                  = np.array([[2],[3]])*Units.degrees 
    Re                   = np.array([[1E5],[1E5]])  
    Ma                   = np.array([[0.2],[0.2]])  
    npoints              = 100

    ospath               = os.path.abspath(__file__)
    separator            = os.path.sep 
    rel_path             = ospath.split('airfoil_analysis' + separator + 'airfoil_panel_method_test.py')[0] + 'Vehicles' + separator + 'Airfoils' + separator 

    airfoils             = [rel_path + 'NACA_4412.txt',rel_path +'Clark_Y.txt']   
    airfoil_geometry     = import_airfoil_geometry(airfoils,npoints)    
    airfoil_results_1    = airfoil_analysis(airfoil_geometry,AoA,Re,Ma,airfoil_stations = [0,1],viscous_flag = False ) 
    
    plot_airfoil_cp(airfoil_results_1)
    plot_airfoil(airfoil_results_1) 
  
    # XFOIL Validation   :   
    xfoil_data_cl   = 0.771082 
    xfoil_data_cm   = -0.117095

    diff_cl           = np.abs(airfoil_results_1.cl[0,0] - xfoil_data_cl)   
    expected_cl_error = 0.005129150981253661
    print('\nCL difference')
    print(diff_cl)
    assert np.abs(((airfoil_results_1.cl[0,0]- expected_cl_error)  - xfoil_data_cl)/xfoil_data_cl)  < 1e-6  

    diff_cm           = np.abs(airfoil_results_1.cm[0,0] - xfoil_data_cm) 
    expected_cm_error =  0.008532607811312251
    print('\nCM difference')
    print(diff_cm)
    assert np.abs(((airfoil_results_1.cm[0,0]- expected_cm_error)  - xfoil_data_cm)/xfoil_data_cm)  < 1e-6  
    

    # ----------------------------------------------------------------------
    # CASE 2: 1 airfoil  with target alpha in viscous mode 
    # ----------------------------------------------------------------------
    AoA                  = np.array([[2]])*Units.degrees 
    Re                   = np.array([[1E5]])  
    Ma                   = np.array([[0.2]])  
    npoints              = 100   
    airfoils             = ['2412']
    airfoil_geometry     = compute_naca_4series(airfoils,npoints)     
    airfoil_results_2    = airfoil_analysis(airfoil_geometry,AoA,Re,Ma,airfoil_stations = [0],viscous_flag = True ) 
      
    plot_airfoil_panels(airfoil_results_2) 
    plot_airfoil_boundary_layers(airfoil_results_2)
    plot_airfoil_distributions(airfoil_results_2) 


    # XFOIL Validation 
    xfoil_data_cd     = 0.015692
    diff_cd           = np.abs(airfoil_results_2.cd[0,0] - xfoil_data_cd)   
    expected_cd_error = 0.00011515468663542833
    print('\nCL difference')
    print(diff_cd)
    assert np.abs(((airfoil_results_2.cd[0,0]- expected_cd_error)  - xfoil_data_cd)/xfoil_data_cd)  < 1e-6  
 
    return   
 

if __name__ == '__main__': 
    main() 
    plt.show()