## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# compute_slat_lift.py
#
# Created:  Dec 2013, A. Variyar
# Modified: Feb 2014, T. Orra
#           Jun 2014, T. Orra 
#           Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#  compute_slat_lift
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def compute_slat_lift(slat_angle,sweep_angle):
    """Computes the increase in lift due to slats

    Assumptions:
    None

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    slat_angle   [radians]
    sweep_angle  [radians]

    Outputs:
    dcl_slat     [Unitless]

    Properties Used:
    N/A
    """     

    # unpack
    sa = slat_angle  / Units.deg
    sw = sweep_angle

    # AA241 Method from adg.stanford.edu
    dcl_slat = (sa/23.)*(np.cos(sw))**1.4 * np.cos(sa * Units.deg)**2

    #returning dcl_slat
    return dcl_slat

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':

    #imports
    import pylab as plt
    import matplotlib
    matplotlib.interactive(True)
    import scipy as sp
    import SUAVE
    from SUAVE.Core import Units

    #define array of sweep and deflection
    sweep_vec = sp.linspace(-10,30,20) * Units.deg
    deflection_vec = sp.linspace(0,50,6)* Units.deg

    dcl_slat = sp.zeros((len(sweep_vec),len(deflection_vec)))
    legend = ''

    for i in range(len(sweep_vec)):
        for j in range(len(deflection_vec)):
            sweep = sweep_vec[i]
            deflection = deflection_vec[j]
            dcl_slat[i,j] = compute_slat_lift(deflection,sweep)

    # ------------------------------------------------------------------
    #   Plotting Delta CL due to Slat vs Sweep angle
    # ------------------------------------------------------------------
    title = "Delta dCL_slat vs Wing sweep"
    plt.figure(1); 
    for deflection in range(len(deflection_vec)):
        plt.plot(sweep_vec/Units.deg , dcl_slat[:,deflection] ,'bo-', \
                    label = 'Deflection: ' +  str(deflection_vec[deflection]/Units.deg) + ' deg')
    plt.xlabel('Sweep angle (deg)'); plt.ylabel('delta CL due to Slat')
    plt.title(title); plt.grid(True)
    legend = plt.legend(loc='upper right', shadow = 'true')
    plt.show(block=True)

