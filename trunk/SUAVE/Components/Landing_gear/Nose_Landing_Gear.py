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
          
        self.tire_diameter = 0.    
        self.strut_length  = 0.    
        self.units         = 0.   #number of nose landing gear    
        self.wheels        = 0.  #number of wheels on the nose landing gear

# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError , 'test failed, not implemented'