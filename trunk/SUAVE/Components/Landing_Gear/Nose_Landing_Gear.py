# Nose_Landing_Gear.py
# 
# Created:  Aug 2015, C. R. I. da Silva
# Modified: Feb 2016, T. MacDonald 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Landing_Gear import Landing_Gear

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

class Nose_Landing_Gear(Landing_Gear):
    """ SUAVE.Components.Landing_Gear.Nose_Landing_Gear()

        Attributes:

        Methods:

        Assumptions:


    """

    def __defaults__(self):        
          
        self.tire_diameter = 0. #diameter of the tire   
        self.strut_length  = 0. #landing gear strut lenght   
        self.units         = 0. # number of nose landing gear    
        self.wheels        = 0. # number of wheels on the nose landing gear

# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError , 'test failed, not implemented'