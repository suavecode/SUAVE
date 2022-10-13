# airfoil_panel_method_test.py
# 
# Created:  
# Modified: Mar 2021, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
from SUAVE.Core import Units
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
    npoints = 200
    
    # -----------------------------------------------
    # Batch analysis of single airfoil - NACA 2410 
    # -----------------------------------------------
    Re_vals              = np.atleast_2d(np.array([1E5,1E5,1E5,1E5]))
    AoA_vals             = np.atleast_2d(np.linspace(-5,10,4)*Units.degrees)      
    airfoil_geometry     = compute_naca_4series(['2412'],npoints)
    airfoil_properties_1 = airfoil_analysis(airfoil_geometry,AoA_vals,Re_vals,airfoil_stations = [0,0,0,0])  
    
     # Plots    
    plot_airfoil_analysis_surface_forces(airfoil_properties_1,show_legend = True )   
    plot_airfoil_analysis_boundary_layer_properties(airfoil_properties_1,show_legend = True )   
    
    # XFOIL Validation - Source   
    xfoil_data_cl   = 0.803793
    xfoil_data_cd   = 0.017329
    xfoil_data_cdpi = 0.005383
    xfoil_data_cm   = -0.053745
  
    diff_CL           = np.abs(airfoil_properties_1.cl[0,2] - xfoil_data_cl)   
    expected_cl_error = 0.005200678285353422
    print('\nCL difference')
    print(diff_CL)
    assert np.abs(((airfoil_properties_1.cl[0,2] - expected_cl_error)  - xfoil_data_cl)/xfoil_data_cl)  < 1e-6 
    

    diff_CD           = np.abs(airfoil_properties_1.cd[0,2] - xfoil_data_cd) 
    expected_cd_error = 0.0002122880489607154
    print('\nCDpi difference')
    print(diff_CD)
    assert np.abs(((airfoil_properties_1.cd[0,2]- expected_cd_error)  - xfoil_data_cd)/xfoil_data_cd)  < 1e-6  
     

    diff_CM           = np.abs(airfoil_properties_1.cm[0,2] - xfoil_data_cm) 
    expected_cm_error = -0.005670722786012765
    print('\nCM difference')
    print(diff_CM)
    assert np.abs(((airfoil_properties_1.cm[0,2]- expected_cm_error)  - xfoil_data_cm)/xfoil_data_cm)  < 1e-6 
    

    # -----------------------------------------------
    # Single Condition Analysis of multiple airfoils  
    # ----------------------------------------------- 
    ospath               = os.path.abspath(__file__)
    separator            = os.path.sep 
    rel_path             = ospath.split('airfoil_analysis' + separator + 'airfoil_panel_method_test.py')[0] + 'Vehicles' + separator + 'Airfoils' + separator 
    Re_vals              = np.atleast_2d(np.array([[1E5,1E5,1E5,1E5,1E5,1E5],[2E5,2E5,2E5,2E5,2E5,2E5]]))
    AoA_vals             = np.atleast_2d(np.array([[2,2,2,2,2,2],[4,4,4,4,4,4]])*Units.degrees)       
    airfoil_stations     = [0,1,0,1,0,1] 
    airfoils             = [rel_path + 'NACA_4412.txt',rel_path +'Clark_y.txt']             
    airfoil_geometry     = import_airfoil_geometry(airfoils, npoints)    
    airfoil_properties_2 = airfoil_analysis(airfoil_geometry,AoA_vals,Re_vals, airfoil_stations = airfoil_stations)    
       
    True_cls    = np.array([0.65581723, 0.57643382, 0.65581723, 0.57643382, 0.65581723,0.57643382])
    True_cd     = np.array([0.01224437, 0.0115724 , 0.01224437, 0.0115724 , 0.01224437,0.0115724 ])
    True_cms    = np.array([-0.09905913, -0.08084637, -0.09905913, -0.08084637, -0.09905913, -0.08084637])
    
    print('\n\nSingle Point Validation')   
    print('\nCL difference')
    print(np.sum(np.abs((airfoil_properties_2.cl[0]  - True_cls)/True_cls)))
    assert np.sum(np.abs((airfoil_properties_2.cl[0]   - True_cls)/True_cls))  < 1e-5 
    
    print('\nCD difference') 
    print(np.sum(np.abs((airfoil_properties_2.cd[0]  - True_cd)/True_cd)))
    assert np.sum(np.abs((airfoil_properties_2.cd[0]   - True_cd)/True_cd))  < 1e-5
    
    print('\nCM difference') 
    print(np.sum(np.abs((airfoil_properties_2.cm[0]  - True_cms)/True_cms)))
    assert np.sum(np.abs((airfoil_properties_2.cm[0]   - True_cms)/True_cms))  < 1e-5  
    return   
    

if __name__ == '__main__': 
    main() 
    plt.show()