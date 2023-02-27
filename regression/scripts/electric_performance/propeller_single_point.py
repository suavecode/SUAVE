# propeller_single_point.py
#
# Created:   Jan 2021, J. Smart
# Modified:  Sep 2021, R. Erhard

#-------------------------------------------------------------------------------
# Imports
#_______________________________________________________________________________

import MARC

from MARC.Core import Units, Data
from MARC.Methods.Performance.propeller_single_point import propeller_single_point
import matplotlib.pyplot as plt

import numpy as np

import sys
sys.path.append('../Vehicles')

from X57_Maxwell_Mod2 import vehicle_setup

from MARC.Analyses.Propulsion.Rotor_Wake_Fidelity_One import Rotor_Wake_Fidelity_One

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
    prop_key = list(vehicle.networks.battery_electric_rotor.rotors.keys())[0]
    prop = vehicle.networks.battery_electric_rotor.rotors[prop_key]

    _, results = propeller_single_point(prop,
                                     pitch=0.,
                                     omega=2200. * Units.rpm,
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

    thrust_r    = 1560.955978730636
    torque_r    = 145.48910609381227
    power_r     = 33518.283837988885
    Cp_r        = 0.042789008180351654
    etap_r      = 0.46570283439198124


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
    prop_key = list(vehicle.networks.battery_electric_rotor.rotors.keys())[0]
    prop = vehicle.networks.battery_electric_rotor.rotors[prop_key]
    
    # update the wake method used for each prop
    prop.Wake = Rotor_Wake_Fidelity_One()

    _, results = propeller_single_point(prop,
                                     pitch=0.,
                                     omega=2200. * Units.rpm,
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

    thrust_r    = 1530.7123997734416
    torque_r    = 153.06201097359252
    power_r     = 35262.955876010026
    Cp_r        = 0.045016233967559396
    etap_r      = 0.4340851076567891

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
