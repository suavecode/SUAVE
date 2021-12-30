# propeller_single_point.py
#
# Created:   Jan 2021, J. Smart
# Modified:  Sep 2021, R. Erhard

#-------------------------------------------------------------------------------
# Imports
#_______________________________________________________________________________

import SUAVE

from SUAVE.Core import Units, Data
from SUAVE.Methods.Performance.propeller_single_point import propeller_single_point
import matplotlib.pyplot as plt

import numpy as np

import sys
sys.path.append('../Vehicles')

from X57_Maxwell_Mod2 import vehicle_setup

#-------------------------------------------------------------------------------
# Test Function
#-------------------------------------------------------------------------------

def main():

    test_1()
    test_2()

    return


def test_1():
    """
    This tests the propeller_single_point function using the BEMT model.
    """
    HFW = False
    vehicle = vehicle_setup()

    analyses = SUAVE.Analyses.Vehicle()
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(atmosphere)


    results = propeller_single_point(vehicle.networks.battery_propeller,
                                     analyses,
                                     pitch=0.,
                                     omega=2200. * Units.rpm,
                                     altitude= 5000. * Units.ft,
                                     delta_isa=0.,
                                     speed=10 * Units['m/s'],
                                     plots=True,
                                     HFW=HFW,
                                     print_results=True
                                     )

    thrust  = results.thrust
    torque  = results.torque
    power   = results.power
    Cp      = results.power_coefficient
    etap    = results.efficiency

    thrust_r    = 715.0296827081207
    torque_r    = 89.64307447098327
    power_r     = 20652.281774904543
    Cp_r        = 0.0008238886294125409
    etap_r      = 0.34622309074679747

    assert (np.abs(thrust - thrust_r) / thrust_r < 1e-6), "Propeller Single Point Regression Failed at Thrust Test"
    assert (np.abs(torque - torque_r) / torque_r < 1e-6), "Propeller Single Point Regression Failed at Torque Test"
    assert (np.abs(power - power_r) / power_r < 1e-6), "Propeller Single Point Regression Failed at Power Test"
    assert (np.abs(Cp - Cp_r) / Cp_r < 1e-6), "Propeller Single Point Regression Failed at Power Coefficient Test"
    assert (np.abs(etap - etap_r) / etap_r < 1e-6), "Propeller Single Point Regression Failed at Efficiency Test"

    return

def test_2():
    """
    This tests the propeller_single_point function using the helical fixed wake (HFW) + BET model.
    """    
    HFW = True
    vehicle = vehicle_setup()

    analyses = SUAVE.Analyses.Vehicle()
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(atmosphere)


    results = propeller_single_point(vehicle.networks.battery_propeller,
                                     analyses,
                                     pitch=0.,
                                     omega=2200. * Units.rpm,
                                     altitude= 5000. * Units.ft,
                                     delta_isa=0.,
                                     speed=10 * Units['m/s'],
                                     plots=True,
                                     HFW=HFW,
                                     print_results=True
                                     )

    thrust  = results.thrust
    torque  = results.torque
    power   = results.power
    Cp      = results.power_coefficient
    etap    = results.efficiency

    thrust_r    = 699.3344413021324
    torque_r    = 84.0161513197105
    power_r     = 19355.931743052
    Cp_r        = 0.0007721728886230788
    etap_r      = 0.3613023907015819

    assert (np.abs(thrust - thrust_r) / thrust_r < 1e-6), "Propeller Single Point Regression Failed at Thrust Test"
    assert (np.abs(torque - torque_r) / torque_r < 1e-6), "Propeller Single Point Regression Failed at Torque Test"
    assert (np.abs(power - power_r) / power_r < 1e-6), "Propeller Single Point Regression Failed at Power Test"
    assert (np.abs(Cp - Cp_r) / Cp_r < 1e-6), "Propeller Single Point Regression Failed at Power Coefficient Test"
    assert (np.abs(etap - etap_r) / etap_r < 1e-6), "Propeller Single Point Regression Failed at Efficiency Test"

    return

if __name__ == '__main__':
    main()
    plt.show()

    print('Propeller Single Point Regression Passed.')
