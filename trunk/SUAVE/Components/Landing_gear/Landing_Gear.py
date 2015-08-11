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

class Main_Landing_Gear(Landing_Gear):
    """ SUAVE.Components.Landing_Gear.Landing_Gear()

        Attributes:

        Methods:

        Assumptions:
            if needed

    """

    def __defaults__(self):
       
        self.tag = 'landing_gear'

        self.tires = Data()
        self.tires.tire_diameter = 0.0
        self.tires.units=0.0
        
        
        self.areas.side_projected = 0.0
        self.areas.wetted = 0.0
        
        self.effective_diameter = 0.0
        self.width = 0.0
        
        self.heights = Data()
        self.heights.maximum = 0.0
        self.heights.at_quarter_length = 0.0
        self.heights.at_three_quarters_length = 0.0
        self.heights.at_vertical_root_quarter_chord = 0.0
        
        self.lengths = Data()
        self.lengths.nose = 0.0
        self.lengths.tail = 0.0
        self.lengths.total = 0.0
        self.lengths.cabin = 0.0
        self.lengths.fore_space = 0.0
        self.lengths.aft_space = 0.0
            
        self.fineness = Data()
        pass

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