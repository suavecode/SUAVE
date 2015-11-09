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

class Nose_Landing_Gear(Landing_Gear):
    """ SUAVE.Components.Landing_Gear.Nose_Landing_Gear()

        Attributes:

        Methods:

        Assumptions:
            if needed

    """

    def __defaults__(self):
        
        self.nose_tire_diameter = 2.2000    #nose gear tire diameter
        self.nose_strut_length = 4.5        #nose landing gear strut length
        self.nose_units = 1                 #number of nose landing gear
        self.nose_wheels = 2                #number of wheels on the nose landing gear           


# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError , 'test failed, not implemented'