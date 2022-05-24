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

    #test_1()
    test_2()

    return


def test_1():
    """
    This tests the propeller_single_point function using the Fidelity Zero rotor wake inflow model.
    """
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
                                     print_results=True
                                     )
        
    thrust  = results.thrust
    torque  = results.torque
    power   = results.power
    Cp      = results.power_coefficient
    etap    = results.efficiency

    thrust_r    = 642.8368993870412
    torque_r    = 127.98612952140998
    power_r     = 29485.887512829242
    Cp_r        = 0.03764130311950602
    etap_r      = 0.21801510946799357


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

    # update the wake method used for each prop
    for p in vehicle.networks.battery_propeller.propellers:
        p.Wake = Rotor_Wake_Fidelity_One()
        
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
                                     print_results=True
                                     )

    thrust  = results.thrust
    torque  = results.torque
    power   = results.power
    Cp      = results.power_coefficient
    etap    = results.efficiency

    thrust_r    = 645.8643152096909
    torque_r    = 127.10023264665021
    power_r     = 29281.791524499793
    Cp_r        = 0.03738075681718287
    etap_r      = 0.22056857916951647


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
