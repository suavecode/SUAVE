# propeller_single_point.py
#
# Created: Jan 2021, J. Smart
# Modified:

#-------------------------------------------------------------------------------
# Imports
#_______________________________________________________________________________

import SUAVE

from SUAVE.Core import Units, Data
from SUAVE.Methods.Performance.propeller_single_point import propeller_single_point

import sys
sys.path.append('../Vehicles')

from X57_Maxwell import vehicle_setup

#-------------------------------------------------------------------------------
# Test Function
#-------------------------------------------------------------------------------

def main():

    vehicle = vehicle_setup()

    analyses = SUAVE.Analyses.Analysis.Container()
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(atmosphere)


    results = propeller_single_point(vehicle.propulsors.battery_propeller,
                                     analyses.atmosphere,
                                     )


    pass

if __name__ == '__main__':
    main()