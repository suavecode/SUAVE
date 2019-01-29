# dual_battery_ducted_fan.py
# 
# Created:  Jan 2019, W. Maier
# Modified:       

""" create and evaluate a dual_battery_ducted_fan network
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container,
)

from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Energy.Networks.Dual_Battery_Ducted_Fan import Dual_Battery_Ducted_Fan
from SUAVE.Methods.Propulsion.ducted_fan_sizing import ducted_fan_sizing

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():   
    # call the network function
    energy_network()
    
    return


def energy_network():
    # ------------------------------------------------------------------
    #   Evaluation Conditions
    # ------------------------------------------------------------------    
    
    # SEE BATTERY DUCTED FAN
    return 

if __name__ == '__main__':
    
    main()