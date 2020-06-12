## @ingroup Components-Landing_Gear
# Landing_Gear.py
# 
# Created:  Aug 2015, C. R. I. da Silva
# Modified: Feb 2016, T. MacDonald
#           May 2020, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Physical_Component

# ----------------------------------------------------------------------
#  A ttribute
# ----------------------------------------------------------------------
## @ingroup Components-Landing_Gear
class Landing_Gear(Physical_Component):
    """ SUAVE.Components.Landing_Gear.Landing_Gear()
        
        The Top Landing Gear Component Class
        
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
                
                Outputs:
                None
                
                Properties Used:
                N/A
        """
       
        self.tag = 'landing_gear'
    
# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

Landing_Gear.Container = Physical_Component.Container

# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError('test failed, not implemented')