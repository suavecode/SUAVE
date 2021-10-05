# airfoil_panel_method_test.py
# 
# Created:  
# Modified: Mar 2021, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
from SUAVE.Core import Units, Data 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis      import airfoil_analysis 
import matplotlib.pyplot as plt   
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series \
     import  compute_naca_4series
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
from SUAVE.Plots.Performance.Airfoil_Plots import * 
import os 
import numpy as np

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():    
    # Define Panelization 
    npanel = 200
    
    # -----------------------------------------------
    # Batch analysis of single airfoil - NACA 2410 
    # -----------------------------------------------
    Re_batch             = np.atleast_2d(np.array([1E5,2E5])).T
    AoA_batch            = np.atleast_2d(np.linspace(-5,10,4)*Units.degrees).T       
    airfoil_geometry     = compute_naca_4series(0.02,0.4,0.1,npoints=npanel)
    airfoil_properties_1 = airfoil_analysis(airfoil_geometry,AoA_batch,Re_batch, npanel, batch_analysis = True)  
    
     # Plots    
    plot_airfoil_analysis_polars(airfoil_properties_1,show_legend = True )  
    plot_airfoil_analysis_surface_forces(airfoil_properties_1,show_legend = True )   
    plot_airfoil_analysis_boundary_layer_properties(airfoil_properties_1,show_legend = True )   
    
    # XFOIL Validation - Source :   
    xfoil_data_Cl  = 0.7922
    xfoil_data_Cd  = 0.01588   
    xfoil_data_Cm  = -0.0504 
  
    diff_CL           = np.abs(airfoil_properties_1.Cl[2,0] - xfoil_data_Cl)   
    expected_Cl_error = -0.01801472136594917
    print('\nCL difference')
    print(diff_CL)
    assert np.abs(((airfoil_properties_1.Cl[2,0]- expected_Cl_error)  - xfoil_data_Cl)/xfoil_data_Cl)  < 1e-6 
    
    diff_CD           = np.abs(airfoil_properties_1.Cd[2,0] - xfoil_data_Cd) 
    expected_Cd_error = -0.00012125071270768784
    print('\nCD difference')
    print(diff_CD)
    assert np.abs(((airfoil_properties_1.Cd[2,0]- expected_Cd_error)  - xfoil_data_Cd)/xfoil_data_Cd)  < 1e-6
    
    diff_CM           = np.abs(airfoil_properties_1.Cm[2,0] - xfoil_data_Cm) 
    expected_Cm_error =  -0.002782281182096287
    print('\nCM difference')
    print(diff_CM)
    assert np.abs(((airfoil_properties_1.Cm[2,0]- expected_Cm_error)  - xfoil_data_Cm)/xfoil_data_Cm)  < 1e-6  

    # -----------------------------------------------
    # Single Condition Analysis of multiple airfoils  
    # ----------------------------------------------- 
    ospath               = os.path.abspath(__file__)
    separator            = os.path.sep 
    rel_path             = ospath.split('airfoil_analysis' + separator + 'airfoil_panel_method_test.py')[0] + 'Vehicles' + separator + 'Airfoils' + separator 
    Re_vals              = np.atleast_2d(np.array([1E5,1E5,1E5,1E5,1E5,1E5])).T
    AoA_vals             = np.atleast_2d(np.array([2,2,2,2,2,2])*Units.degrees).T       
    airfoil_stations     = [0,1,0,1,0,1] 
    airfoils             = [rel_path + 'NACA_4412.txt',rel_path +'Clark_y.txt']             
    airfoil_geometry     = import_airfoil_geometry(airfoils, npoints = (npanel + 2))    
    airfoil_properties_2 = airfoil_analysis(airfoil_geometry,AoA_vals,Re_vals, npanel, batch_analysis = False, airfoil_stations = airfoil_stations)    
       
    True_Cls  = np.array([[0.68788905],[0.60906619],[0.68788905],[0.60906619],[0.68788905],[0.60906619]])
    True_Cds  = np.array([[0.01015395],[0.00846174],[0.01015395],[0.00846174],[0.01015395],[0.00846174]])
    True_Cms  = np.array([[-0.10478782],[-0.08727453],[-0.10478782],[-0.08727453],[-0.10478782],[-0.08727453]])     
    
    print('\n\nSingle Point Validation')   
    print('\nCL difference')
    print(np.sum(np.abs((airfoil_properties_2.Cl  - True_Cls)/True_Cls)))
    assert np.sum(np.abs((airfoil_properties_2.Cl  - True_Cls)/True_Cls))  < 1e-5 
    
    print('\nCD difference') 
    print(np.sum(np.abs((airfoil_properties_2.Cd  - True_Cds)/True_Cds)))
    assert np.sum(np.abs((airfoil_properties_2.Cd  - True_Cds)/True_Cds))  < 1e-5
    
    print('\nCM difference') 
    print(np.sum(np.abs((airfoil_properties_2.Cm  - True_Cms)/True_Cms)))
    assert np.sum(np.abs((airfoil_properties_2.Cm  - True_Cms)/True_Cms))  < 1e-5  
    return   
    

if __name__ == '__main__': 
    main() 
    plt.show()