# Analysis.py

# Created:  Mar, 2014, SUAVE Team
# Modified: Jan, 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Attributes.Atmospheres import Earth
from Runway import Runway

# ----------------------------------------------------------------------
#  Airport Data Class
# ----------------------------------------------------------------------

class Airport(Data):
    """ SUAVE.Attributes.Airports.Airport
    """
    def __defaults__(self):
        self.tag = 'Airport'
        self.altitude = 0.0        # m
        self.atmosphere = Earth.US_Standard_1976()
        self.delta_isa = 0.0    

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------
Airport.Runway = Runway
