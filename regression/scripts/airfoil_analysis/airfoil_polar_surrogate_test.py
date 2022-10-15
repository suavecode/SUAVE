# airfoil_polar_surrogate_test.py
#
# Created:   Aug 2022, R. Erhard
# Modified:  
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_properties import compute_airfoil_properties
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry    import import_airfoil_geometry
from SUAVE.Plots.Performance.Airfoil_Plots                                                   import plot_airfoil_polars
from SUAVE.Core import Units, Data 
import numpy as np
import os
import pylab as plt

def main():
    """
    This script generates airfoil surrogates for three airfoils, and plots the 
    surrogate data over a range of alpha and Reynolds number sweeps.
    """
    
    store_new_regression_results = False  # Make True if running new regression
    
    # -----------------------------------------------------------------------------------------
    # Prepare Airfoil Imports 
    # -----------------------------------------------------------------------------------------          
    airfoils_path               = os.path.dirname(os.path.abspath(__file__)) + "/../Vehicles/Airfoils/"
    polars_path                 = os.path.dirname(os.path.abspath(__file__)) + "/../Vehicles/Airfoils/Polars/" 
    airfoils                    = ["Clark_y.txt", "NACA_63_412.txt", "NACA_4412.txt"]  
    airfoil_data                = Data()
    airfoil_data.geometry_files = []    
    airfoil_data.polar_files    = []
    
    for a in airfoils: 
        airfoil_data.geometry_files.append(airfoils_path + a )
        aName          = a[:-4] 
        airfoil_polars = []
        Re_val         = []
        for f in os.listdir(polars_path):
            if aName in f and f.endswith('.txt'):
                airfoil_polars.append(polars_path + f)
                Re_val.append(float(f.split('_')[-1][:-4])) 
        sorted_ids = np.argsort(Re_val) # sort by Re number of polars
        airfoil_data.polar_files.append([airfoil_polars[i] for i in sorted_ids])
    
    # -----------------------------------------------------------------------------------------
    # Surrogate Tests  
    # -----------------------------------------------------------------------------------------       
    airfoil_data.geometry = import_airfoil_geometry(airfoil_data.geometry_files)
    airfoil_data.polars   = compute_airfoil_properties(airfoil_data.geometry,airfoil_data.polar_files, use_pre_stall_data=False)
    aoa_sweep             = np.linspace(-20,20,100) * Units.deg
    Re_sweep              = np.array([0.05, 0.1, 0.2, 0.5, 1., 3.5, 5., 7.5]) * 1e6  
    
    # plot airfoil polars 
    plot_airfoil_polars(airfoil_data) 
    
    # -----------------------------------------------------------------------------------------
    # Regression comparison
    # -----------------------------------------------------------------------------------------  
    xc         = airfoil_data.geometry.x_coordinates[0]
    cl_outputs = airfoil_data.polars.lift_coefficient_surrogates[airfoil_data.geometry_files[0]]((Re_sweep[0], aoa_sweep))
    cd_outputs = airfoil_data.polars.drag_coefficient_surrogates[airfoil_data.geometry_files[0]]((Re_sweep[0], aoa_sweep))
         
    if store_new_regression_results:
        np.save(airfoils[0][:-4]+"_xc", airfoil_data.geometry.x_coordinates[0])
        np.save(airfoils[0][:-4]+"_cl_sweep", cl_outputs)
        np.save(airfoils[0][:-4]+"_cd_sweep", cd_outputs)
    
    # regress coordinate values
    xc_true         = np.load(airfoils[0][:-4]+"_xc.npy")
    cl_outputs_true = np.load(airfoils[0][:-4]+"_cl_sweep.npy")
    cd_outputs_true = np.load(airfoils[0][:-4]+"_cd_sweep.npy") 
    assert(np.max(abs(xc-xc_true)) < 1e-6)
    assert(np.max(abs(cl_outputs-cl_outputs_true)) < 1e-6)
    assert(np.max(abs(cd_outputs-cd_outputs_true)) < 1e-6)

    plt.show()
    return
 
if __name__ == "__main__":
    main()