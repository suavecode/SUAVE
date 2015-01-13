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

        self.velocity_start       = 150 * Units.knots
        self.velocity_end         = 0.0
        self.friction_coefficient = 0.4
        self.throttle             = 0.0
        self.battery_energy = 0.0
        self.latitude       = 0.0
        self.longitude      = 0.0        


    def initialize_conditions(self,conditions,numerics,initials=None):

        conditions = Ground_Segment.initialize_conditions(self,conditions,numerics,initials)

        m_initial = self.analyses.weights.mass_properties.landing
        conditions.weights.total_mass[:,0] = m_initial

        throttle = self.throttle	
        conditions.propulsion.throttle[:,0] = throttle

        return conditions