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
from SUAVE.Plots.Airfoil_Plots  import plot_airfoil_batch_properties  
import os 
import numpy as np

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():    
    # Define Panelization 
    npanel = 100 
    
    # -----------------------------------------------
    # Batch analysis of single airfoil - NACA 2410 
    # -----------------------------------------------
    Re_batch     = np.atleast_2d(np.array([1E5,2E5])).T
    AoA_batch    = np.atleast_2d(np.linspace(-5,15,5)*Units.degrees).T       
    airfoil_geometry     = compute_naca_4series(0.02,0.4,0.1,npoints=npanel )
    airfoil_properties_1 = airfoil_analysis(airfoil_geometry,AoA_batch,Re_batch, npanel, n_computation = 200, batch_analysis = True)  
    plot_airfoil_batch_properties(airfoil_properties_1,arrow_color = 'r',plot_pressure_vectors = True)  
    
    # XFOIL Validation - Source :   
    xfoil_data_Cl             = 0.7922
    xfoil_data_Cd             = 0.01588   
    xfoil_data_Cm             = -0.0504 
 
    print('\n\nNACA 2410 Batch Analysis Validation at 5 deg, Re = 1E5') 
    diff_CL           = np.abs(airfoil_properties_1.Cl[2,0] - xfoil_data_Cl)  # 0.573685
    expected_Cl_error = -0.08267635005095875
    print('\nCL difference')
    print(diff_CL)
    assert np.abs(((airfoil_properties_1.Cl[2,0]- expected_Cl_error)  - xfoil_data_Cl)/xfoil_data_Cl)  < 1e-6 
    
    diff_CD           = np.abs(airfoil_properties_1.Cd[2,0] - xfoil_data_Cd) 
    expected_Cd_error = -0.006893203451858655
    print('\nCD difference')
    print(diff_CD)
    assert np.abs(((airfoil_properties_1.Cd[2,0]- expected_Cd_error)  - xfoil_data_Cd)/xfoil_data_Cd)  < 1e-6
    
    diff_CM           = np.abs(airfoil_properties_1.Cm[2,0] - xfoil_data_Cm) 
    expected_Cm_error =  0.012488216333628205
    print('\nCM difference')
    print(diff_CM)
    assert np.abs(((airfoil_properties_1.Cm[2,0]- expected_Cm_error)  - xfoil_data_Cm)/xfoil_data_Cm)  < 1e-6  
    return   

    # -----------------------------------------------
    # Single Condition Analysis of multiple airfoils  
    # ----------------------------------------------- 
    separator = os.path.sep 
    rel_path  = ospath.split('airfoil_analysis' + separator + 'airfoil_panel_method_test.py')[0] + 'Vehicles' + separator + 'Airfoils' + separator 
    Re_vals              = np.atleast_2d(np.array([1E5,1E5,1E5,1E5,1E5,1E5])).T
    AoA_vals             = np.atleast_2d(np.array([2,2,2,2,2,2])*Units.degrees).T       
    airfoil_stations     = [0,1,0,1,0,1] 
    airfoils             = [rel_path + 'NACA_4412.txt',rel_path +'Clark_y.txt']             
    airfoil_geometry     = import_airfoil_geometry(airfoils, npoints= npanel+2)    
    airfoil_properties_2 = airfoil_analysis(airfoil_geometry,AoA_vals,Re_vals, npanel, batch_analysis = False, airfoil_stations = airfoil_stations)    
    
    True_Cls  = np.array([0.57162636, 0.5362878 , 0.56544856, 0.5362878 , 0.56544856, 0.5362878 ])
    True_Cds  = np.array([0.01133623, 0.00716673, 0.00318314, 0.00716673, 0.00318314, 0.00716673])
    True_Cms  = np.array([-0.07574378, -0.05903947, -0.06817584, -0.05903947, -0.06817584,-0.05903947])    
    
    print('\n\nSingle Point Validation')  
    print('\nCL difference')
    print(diff_CL)
    assert np.sum(np.abs((airfoil_properties_2.Cl[2,0]  - True_Cls)/True_Cls))  < 1e-6 
     
    print('\nCD difference')
    print(diff_CD)
    assert np.sum(np.abs((airfoil_properties_2.Cd[2,0]  - True_Cds)/True_Cds))  < 1e-6
     
    print('\nCM difference')
    print(diff_CM)
    assert np.sum(np.abs((airfoil_properties_2.Cm[2,0]  - True_Cms)/True_Cms))  < 1e-6  
    return   
    

if __name__ == '__main__': 
    main() 
    plt.show()