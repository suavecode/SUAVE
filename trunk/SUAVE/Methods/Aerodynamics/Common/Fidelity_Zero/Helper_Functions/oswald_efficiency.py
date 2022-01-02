## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
# Oswald_efficiency.py
# 
# Created:  Nov 2021, S. Karpuk
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
from SUAVE.Core import Units
import numpy as np

# ------------------------------------------------------------------------------------
#  Compute Oswald efficiency using the method of Scholz for the constraint analysis
# ------------------------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
def oswald_efficiency(vehicle,cdmin):
    """Calculate an average Oswald efficiencies based on the method of Scholz for the constraint analysis

        Assumptions:
        None

        Source:
           M. Nita, D.Scholtz, 'Estimating the Oswald factor from basic aircraft geometrical parameters',
            Deutscher Luft- und Raumfahrtkongress 20121DocumentID: 281424

        Inputs:
            constraint_analysis.geometry.taper                 [Unitless]
                                aspect_ratio                   [Unitless]
                                sweep_quarter_chord            [radians]
                                aerodynamics.fuselage_factor   [Unitless]
                                viscous_factor                 [Unitless]

        Outputs:
            e          [Unitless]

        Properties Used:

    """  

    # Unpack inputs
    taper = vehicle.wings['main_wing'].taper
    AR    = vehicle.wings['main_wing'].aspect_ratio
    sweep = vehicle.wings['main_wing'].sweeps.quarter_chord / Units.degrees
    kf    = vehicle.constraints.aerodynamics.fuselage_factor 
    K     = vehicle.constraints.aerodynamics.viscous_factor 

    dtaper    = -0.357+0.45*np.exp(-0.0375*np.abs(sweep))
    eff_taper = taper - dtaper
    f_taper   = 0.0524*eff_taper**4-0.15*eff_taper**3+0.1659*eff_taper**2-0.0706*eff_taper+0.0119
    u         = 1 / (1+f_taper*AR)
    P         = K*cdmin
    Q         = 1/(kf*u)
      
    e = 1/(Q+P*np.pi*AR)


    return e
