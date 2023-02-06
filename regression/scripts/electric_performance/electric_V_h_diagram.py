# electric_V_h_diagram.py
#
# Created: Jan 2021, J. Smart
# Modified:

#-------------------------------------------------------------------------------
# Imports
#_______________________________________________________________________________

import MARC

from MARC.Core import Units, Data
from MARC.Methods.Performance.electric_V_h_diagram import electric_V_h_diagram

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

    analyses = MARC.Analyses.Vehicle()

    sizing = MARC.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    weights = MARC.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    aerodynamics = MARC.Analyses.Aerodynamics.AERODAS()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    stability = MARC.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)

    energy = MARC.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    planet = MARC.Analyses.Planets.Planet()
    analyses.append(planet)

    atmosphere = MARC.Analyses.Atmospheric.US_Standard_1976()
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
                                      test_omega      = 1900 * Units.rpm,
                                      display_plot=True)


    climb_rate_r = [[  0.        ,   0.        ,   0.        ,   0.        ,              0.        ],
                    [  0.        ,   0.        ,   0.        ,   0.        ,              0.        ],
                    [487.63179542,   0.        ,   0.        ,   0.        ,              0.        ],
                    [772.92643016, 666.52055528, 562.06865597, 459.0544259 ,            356.95140061],
                    [447.06298061, 375.79491086, 305.65495126, 236.2785227 ,            167.26496564]]


    assert (np.all(np.nan_to_num(np.abs(climb_rate-climb_rate_r)/climb_rate_r) < 1e-6)), "Electric V_h Diagram Regression Failed"

    return

if __name__ == '__main__':
    main()
    plt.show()

    print('Electric V_h Diagram Regression Passed.')