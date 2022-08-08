# airfoil_polar_surrogate_test.py
#
# Created:   Aug 2022, R. Erhard
# Modified:  
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars import (
    compute_airfoil_polars,
)

from SUAVE.Plots.Performance.Airfoil_Plots import plot_airfoil_polars, plot_raw_data_airfoil_polars
from SUAVE.Core import Units
import numpy as np
import os
import pylab as plt

def main():
    """
    This script generates airfoil surrogates for three airfoils, and plots the 
    surrogate data over a range of alpha and Reynolds number sweeps.
    """
    airfoils_path = os.path.dirname(os.path.abspath(__file__)) + "/../Vehicles/Airfoils/"
    polars_path = os.path.dirname(os.path.abspath(__file__)) + "/../Vehicles/Airfoils/Polars/"
    
    airfoils = ["Clark_y.txt", "NACA_63_412.txt", "NACA_4412.txt"]

    airfoil_geometry_files = []    
    airfoil_polar_files = []
    
    for a in airfoils:
        
        airfoil_geometry_files.append(airfoils_path + a )
        aName = a[:-4]
     
        airfoil_polars = []
        Re_val = []
        for f in os.listdir(polars_path):
            if aName in f and f.endswith('.txt'):
                airfoil_polars.append(polars_path + f)
                Re_val.append(float(f.split('_')[-1][:-4]))
        
        # sort by Re number of polars
        sorted_ids = np.argsort(Re_val)
        airfoil_polar_files.append([airfoil_polars[i] for i in sorted_ids])
    
    airfoil_data = compute_airfoil_polars(airfoil_geometry_files, airfoil_polar_files, use_pre_stall_data=True)
    aoa_sweep = np.linspace(-20,20,100) * Units.deg
    #Re_sweep = np.array([0.1, 0.2, 0.5, 1., 3.5, 5., 7.5]) * 1e6
    Re_sweep = np.array([0.05, 0.1, 0.2, 0.5, 1.]) * 1e6
    plot_airfoil_polars(airfoil_data, aoa_sweep, Re_sweep)   
    plot_raw_data_airfoil_polars(airfoil_data.airfoil_names, airfoil_polar_files)

    plt.show()
    
    # -----------------------------------------------------------------------------------------
    # Regression comparison
    # -----------------------------------------------------------------------------------------  
    xc = airfoil_data.x_coordinates[0]
    cl_outputs = airfoil_data.lift_coefficient_surrogates[0]((Re_sweep[0], aoa_sweep))
    cd_outputs = airfoil_data.drag_coefficient_surrogates[0]((Re_sweep[0], aoa_sweep))
        
    ## store new regression data (LEAVE COMMENTED OUT)
    #np.save(airfoil_data.airfoil_names[0]+"_xc", airfoil_data.x_coordinates[0])
    #np.save(airfoil_data.airfoil_names[0]+"_cl_sweep", cl_outputs)
    #np.save(airfoil_data.airfoil_names[0]+"_cd_sweep", cd_outputs)
    
    # regress coordinate values
    xc_true = np.load(airfoil_data.airfoil_names[0]+"_xc.npy")
    cl_outputs_true = np.load(airfoil_data.airfoil_names[0]+"_cl_sweep.npy")
    cd_outputs_true = np.load(airfoil_data.airfoil_names[0]+"_cd_sweep.npy")
    assert(np.max(abs(xc-xc_true)) < 1e-6)
    assert(np.max(abs(cl_outputs-cl_outputs_true)) < 1e-6)
    assert(np.max(abs(cd_outputs-cd_outputs_true)) < 1e-6)

    return

if __name__ == "__main__":
    main()