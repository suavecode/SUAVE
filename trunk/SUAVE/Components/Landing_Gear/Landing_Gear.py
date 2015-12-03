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
#  A ttribute
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

        #self.position  = [0.0,0.0,0.0]        
        self.main_tire_diameter = 0.
        self.nose_tire_diameter = 0.
        self.main_strut_length  = 0.
        self.nose_strut_length  = 0.
        self.main_units         = 0.   #number of main landing gear units
        self.nose_units         = 0.   #number of nose landing gear
        self.main_wheels        = 0.  #number of wheels on the main landing gear
        self.nose_wheels        = 0.  #number of wheels on the nose landing gear
        
        
        
        
        
        
        
        
        
   
# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError , 'test failed, not implemented'