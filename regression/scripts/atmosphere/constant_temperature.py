# constant_temperature.py
# 
# Created:  April 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

import SUAVE
import numpy as np
import matplotlib.pyplot as plt
from SUAVE.Core import Units


# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  
def main():
    
    # ------------------------------------------------------------------
    #   The Tests
    # ------------------------------------------------------------------    

    # initialize atmospheric models
    atm = SUAVE.Analyses.Atmospheric.Constant_Temperature()
    
    # test elevations -3 km <= z <= 90 km
    z = np.linspace(-3,90,10) * Units.km

    # compute values from each model
    conditions = atm.compute_values(z)
    p = conditions.pressure
    T = conditions.temperature
    rho = conditions.density
    a = conditions.speed_of_sound
    
    # get the comparison values
    p_truth, rho_truth = get_truth()
    
    # difference
    p_err   = np.max( p_truth   - p   )
    rho_err = np.max( rho_truth - rho )
   
    print('Max Pressure Difference       = %.4e' % p_err)
    print('Max Density Difference        = %.4e' % rho_err)   
    
    # ------------------------------------------------------------------
    #   Plotting
    # ------------------------------------------------------------------    

    # plot data

    title = "Constant Temperature Atmosphere"
    plt.subplot(121)
    plt.plot(p/101325,z/Units.km)
    plt.xlabel('Pressure (atm)'); plt.xscale('log')
    plt.ylabel('Altitude (km)')
    plt.title(title)
    plt.grid(True)

    plt.subplot(122)
    plt.plot(rho,z/Units.km)
    plt.xlabel('Density (kg/m^3)'); plt.xscale('log')
    plt.ylabel('Altitude (km)')
    plt.title(title)
    plt.grid(True)

    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------    

    assert( p_err   < 1e-1 )
    assert( rho_err < 1e-5 )   
 
    return

# ----------------------------------------------------------------------        
#   Helper Function
# ---------------------------------------------------------------------- 

def get_truth():
    p_truth = np.array([[  1.27774000e+05],[  4.25163568e+04],[  1.03269587e+04],
                        [  2.15160449e+03],[  4.20949197e+02],[  9.50899534e+01],
                        [  2.76459966e+01],[  8.32002187e+00],[  1.59116434e+00],
                        [  7.65702963e-01]])
   
    rho_truth = np.array([[  1.54476339e+00],[  5.14014677e-01],[  1.24850969e-01],
                          [  2.60124895e-02],[  5.08919582e-03],[  1.14961947e-03],
                          [  3.34234847e-04],[  1.00587484e-04],[  1.92368746e-05],
                          [  9.25720336e-06]])
    
    return p_truth, rho_truth
    

# ----------------------------------------------------------------------        
#   Call Main
# ---------------------------------------------------------------------- 

if __name__ == '__main__':
    main()
    plt.show(block=True)
