## @ingroup Components-Landing_Gear
# Main_Landing_Gear.py
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
class Main_Landing_Gear(Landing_Gear):
    """ SUAVE.Components.Landing_Gear.Main_Landing_Gear()
        
        The MLG Landing Gear Component Class
        
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
        self.tag           = 'main_gear'
        self.units         = 0. # number of main landing gear units        
        self.strut_length  = 0.
        self.tire_diameter = 0.
        self.units         = 0. # number of main landing gear units
        self.wheels        = 0. # number of wheels on the main landing gear


# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError('test failed, not implemented')