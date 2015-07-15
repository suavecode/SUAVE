# wing_fuel_volume.py
#
# Created:  Tarik, Apr 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Core  import Data

# package imports
import numpy
from math import pi, sqrt

# ----------------------------------------------------------------------
#  Correlation-based methods for wing fuel capacity estimation
# ----------------------------------------------------------------------

def wing_fuel_volume(wing):
    """ SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_fuel_volume(wing):
        Estimates wing fuel capacity based in correlation methods.

        Inputs:
            wing.sref    - Wing reference area [m2]
            wing.ar     - Wing Aspect Ratio (span?/wing area) [-]
            wing.t_C    - Wing thickness ration [-]

        Outputs:
            wing.fuel_volume - Wing fuel capacity [m?]


        Assumptions:
      		Wing fuel capacity calculated according to Torenbeek, E., "Advanced
    Aircraft Design", 2013 (equation 10.30)

    """

    # unpack
    sref  = wing.areas.reference
    ar    = wing.aspect_ratio
    tc    = wing.thickness_to_chord

    # calculate
    volume = 0.90* tc * sref** 1.5 * ar**-0.5 * 0.55

    # pack
    wing.fuel_volume = volume

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
if __name__ == '__main__':

        # imports
    import SUAVE
    import scipy as sp
    import pylab as plt

    #define arrays of wing area, AR and t/c
    tc_vec   = sp.linspace(0.10,0.12,2)
    sw_array = sp.linspace(50,200,11)
    AR_vec   = sp.linspace(6,16,11)

    wing     = SUAVE.Components.Wings.Wing()

    wing_fuel = sp.zeros((len(tc_vec),len(sw_array),len(AR_vec)))

    for i in range(len(tc_vec)):
        wing.t_c = tc_vec[i]
        for j in range(len(sw_array)):
            wing.sref = sw_array[j]
            for k in range(len(AR_vec)):
                wing.ar = AR_vec[k]
                wing_fuel_volume(wing)
                wing_fuel[i,j,k] = wing.fuel_volume

    # ------------------------------------------------------------------
    #   Plotting
    # ------------------------------------------------------------------
    title = "Wing Fuel Capacity"

    for tc in range(len(tc_vec)):
        plt.figure(tc); plt.hold
        for AR in range(len(AR_vec)):
            plt.plot(sw_array , wing_fuel[tc,:,AR] ,'bo-', \
                     label = 't/c: ' +  str(tc_vec[tc]) + ' AR = ' + str(AR_vec[AR]))
        plt.xlabel('Wing Area (deg)'); plt.ylabel('Fuel Volume (m3)')
        plt.title(title); plt.grid(True)
        legend = plt.legend(loc='upper right', shadow = 'true')
    plt.show(block=True)