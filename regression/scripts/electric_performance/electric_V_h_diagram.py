# electric_V_h_diagram.py
#
# Created: Jan 2021, J. Smart
# Modified:

#-------------------------------------------------------------------------------
# Imports
#_______________________________________________________________________________

import SUAVE

from SUAVE.Core import Units, Data
from SUAVE.Methods.Performance.electric_V_h_diagram import electric_V_h_diagram

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../Vehicles')

from X57_Maxwell_Mod2 import vehicle_setup

#-------------------------------------------------------------------------------
# Test Function
#-------------------------------------------------------------------------------

def main():

    vehicle = vehicle_setup()

    analyses = SUAVE.Analyses.Vehicle()

    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    aerodynamics = SUAVE.Analyses.Aerodynamics.AERODAS()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)

    energy = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    analyses.finalize()

    climb_rate = electric_V_h_diagram(vehicle,
                                      analyses,
                                      CL_max          = 1.4,
                                      delta_isa       = 0.,
                                      grid_points     = 5,
                                      altitude_ceiling= 1e4 * Units.ft,
                                      max_speed       = 150 * Units.knots,
                                      test_omega      = 2400 * Units.rpm,
                                      display_plot=True)


    climb_rate_r = [[  0.        ,   0.        ,   0.        ,   0.        ,   0.        ],
                    [  0.        ,   0.        ,   0.        ,   0.        ,   0.        ],
                    [365.28164616,   0.        ,   0.        ,   0.        ,   0.        ],
                    [671.06501573, 565.12568368, 462.33970468, 362.03685772, 263.49032629],
                    [677.20427984, 577.26538105, 483.80816586, 395.7082193 , 311.81340271]]


    assert (np.all(np.nan_to_num(np.abs(climb_rate-climb_rate_r)/climb_rate_r) < 1e-6)), "Electric V_h Diagram Regression Failed"

    return

if __name__ == '__main__':
    main()
    plt.show()

    print('Electric V_h Diagram Regression Passed.')