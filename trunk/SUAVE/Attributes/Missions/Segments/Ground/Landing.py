#Landing.py
#Tim Momose, May 2014


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from Ground_Segment import Ground_Segment

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr
deg = Units.deg

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Landing(Ground_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):

	self.tag = "Landing Segment"
	

    def initialize_conditions(self,conditions,numerics,initials=None):
	
	raise NotImplementedError