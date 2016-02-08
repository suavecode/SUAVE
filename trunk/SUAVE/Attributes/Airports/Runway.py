#Runway.py
# 
# Created:  Mar, 2014, SUAVE Team
# Modified: Jan, 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Runway Data Class
# ----------------------------------------------------------------------
    
class Runway(Data):
    """ SUAVE.Attributes.Airport.Runway
        Data object used to hold runway data
    """
    def __defaults__(self):
        self.tag = 'Runway'
