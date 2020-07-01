## @ingroup Components-Landing_Gear
# Nose_Landing_Gear.py
# 
# Created:  Aug 2015, C. R. I. da Silva
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Landing_Gear import Landing_Gear

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------
## @ingroup Components-Landing_Gear
class Nose_Landing_Gear(Landing_Gear):
    """ SUAVE.Components.Landing_Gear.Nose_Landing_Gear()
        
        The NLG Landing Gear Component Class
            
            Assumptions:
            None
            
            Source:
            N/A        
    """

    def __defaults__(self): 
        
        """ This sets the default values for the component attributes.
        
                Assumptions:
                None
                
                Source:
                N/A
                
                Inputs:
                None
                
                Output:
                None
                
                Properties Used:
                N/A
        """
        self.tag           = 'nose_gear'
        self.tire_diameter = 0.    
        self.strut_length  = 0.    
        self.units         = 0. # number of nose landing gear    
        self.wheels        = 0. # number of wheels on the nose landing gear

# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError('test failed, not implemented')