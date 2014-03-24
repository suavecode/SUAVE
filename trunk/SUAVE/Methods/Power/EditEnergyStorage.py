# Power.py
#
# Created By:       M. Colonno
# Last updated:     M. Vegh 8/6/13

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from scipy import trapz
from SUAVE.Structure  import Data
#from SUAVE.Attributes import Constants
from SUAVE.Attributes.Atmospheres.Earth import USStandard1976
from SUAVE.Attributes.Gases import Air
atm = USStandard1976()
air=Air()
# from SUAVE.Attributes.Missions import Mission

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def EditEnergyStorage(index,inputs):

    for parameter in inputs:
        AV.Energy.Storage[index][parameter] = inputs[parameter]

    #tank = Data()
    #tank.Type = "Liquid Propellant"         # Type
    #tank.Name = "LH2 1"                     # Name
    #tank.MassDensity = 0.0                  # kg/m^3
    #tank.StorageTemperature = 300           # K
    #tank.StoragePressure = 1.0              # atmospheres
    #tank.HeatOfReaction = 0.0               # MJ/kg
    #tank.Viscosity = 0.0                    # kg/m-s

    # get current list length 
    #AV.Energy.Storage.append(tank)

    return 
