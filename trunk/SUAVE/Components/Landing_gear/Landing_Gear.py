# Main_Landing_Gear.py
#
# Created:  Carlos, Aug 2015
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave imports
import numpy as np

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body, Mass_Properties

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

class Landing_Gear(Physical_Component):
    """ SUAVE.Components.Landing_Gear.Landing_Gear()

        Attributes:

        Methods:

        Assumptions:
            if needed

    """

    def __defaults__(self):
       
        self.tag = 'landing_gear'

        self.position  = [0.0,0.0,0.0]        

    vehicle.landing_gear = Data()
    vehicle.landing_gear.main_tire_diameter = 3.5000
    vehicle.landing_gear.nose_tire_diameter = 2.2000
    vehicle.landing_gear.main_strut_length = 5.66
    vehicle.landing_gear.nose_strut_length = 4.5
    vehicle.landing_gear.main_units = 2     #number of main landing gear units
    vehicle.landing_gear.nose_units = 1     #number of nose landing gear
    vehicle.landing_gear.main_wheels = 2    #number of wheels on the main landing gear
    vehicle.landing_gear.nose_wheels = 2    #number of wheels on the nose landing gear    
# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError , 'test failed, not implemented'