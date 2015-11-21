# Main_Wing.py
#
# Created:  Carlos, Aug 2015
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave imports
from Wing import Wing

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

class Main_Landing_Gear(Landing_Gear):
    """ SUAVE.Components.Landing_Gear.Nose_Landing_Gear()

        Attributes:

        Methods:

        Assumptions:
            if needed

    """

    def __defaults__(self):
        self.main_units = 2     #number of main landing gear units
        
        self.main_tire_diameter = 3.5000 #main gear tire diameter
        self.main_strut_length = 5.66 #main landing gear strut length
        self.main_wheels = 2    #number of wheels on the main landing gear



# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError , 'test failed, not implemented'