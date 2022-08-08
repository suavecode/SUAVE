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

from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_One import Rotor_Wake_Fidelity_One

#-------------------------------------------------------------------------------
# Test Function
#-------------------------------------------------------------------------------

def main():

    test_1()
    test_2()

    return


def test_1():
    """
    This tests the propeller_single_point function using the Fidelity Zero rotor wake inflow model.
    """
    vehicle = vehicle_setup()
    prop_key = list(vehicle.networks.battery_propeller.propellers.keys())[0]
    prop = vehicle.networks.battery_propeller.propellers[prop_key]

    _, results = propeller_single_point(prop,
                                     pitch=0.,
                                     omega=1500. * Units.rpm,
                                     altitude= 5000. * Units.ft,
                                     delta_isa=0.,
                                     speed=10 * Units['m/s'],
                                     plots=True,
                                     print_results=True
                                     )
        
    thrust  = results.thrust
    torque  = results.torque
    power   = results.power
    Cp      = results.power_coefficient
    etap    = results.efficiency

    thrust_r    = 283.34728002003214
    torque_r    = 65.5433160782699
    power_r     = 10295.520014170324
    Cp_r        = 0.04146608161360752
    etap_r      = 0.2752141510385534


    assert (np.abs(thrust - thrust_r) / thrust_r < 1e-6), "Propeller Single Point Regression Failed at Thrust Test"
    assert (np.abs(torque - torque_r) / torque_r < 1e-6), "Propeller Single Point Regression Failed at Torque Test"
    assert (np.abs(power - power_r) / power_r < 1e-6), "Propeller Single Point Regression Failed at Power Test"
    assert (np.abs(Cp - Cp_r) / Cp_r < 1e-6), "Propeller Single Point Regression Failed at Power Coefficient Test"
    assert (np.abs(etap - etap_r) / etap_r < 1e-6), "Propeller Single Point Regression Failed at Efficiency Test"

    return

def test_2():
    """
    This tests the propeller_single_point function using the Fidelity One rotor inflow model.
    """    
    vehicle = vehicle_setup()
    prop_key = list(vehicle.networks.battery_propeller.propellers.keys())[0]
    prop = vehicle.networks.battery_propeller.propellers[prop_key]
    
    # update the wake method used for each prop
    prop.Wake = Rotor_Wake_Fidelity_One()

    _, results = propeller_single_point(prop,
                                     pitch=0.,
                                     omega=1500. * Units.rpm,
                                     altitude= 5000. * Units.ft,
                                     delta_isa=0.,
                                     speed=10 * Units['m/s'],
                                     plots=True,
                                     print_results=True
                                     )

    thrust  = results.thrust
    torque  = results.torque
    power   = results.power
    Cp      = results.power_coefficient
    etap    = results.efficiency

    thrust_r    = 284.79236891828293
    torque_r    = 65.2604080020779
    power_r     = 10251.080917480023
    Cp_r        = 0.041287099375929504
    etap_r      = 0.27781691629480587


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
