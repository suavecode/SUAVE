## @ingroup Components-Wings
# Vertical_Tail_All_Moving.py
#
# Created:  Jul 2021, A. Blaufox
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from .Vertical_Tail      import Vertical_Tail
from .All_Moving_Surface import All_Moving_Surface

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

## @ingroup Components-Wings
class Vertical_Tail_All_Moving(Vertical_Tail, All_Moving_Surface):
    """ This class is used to define all-moving vertical tails in SUAVE. Note that it 
    inherits from both Horizontal_Tail and All_Moving_Surface
    
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

    def __defaults__(self):
        """This sets the default for all moving-vertical tails in SUAVE.
        
        See All_Moving_Surface().__defaults__ and Wing().__defaults__ for an explanation 
        of attributes
    
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
        self.tag = 'vertical_tail_all_moving'
        self.sign_duplicate        = -1.0   
    
    def make_x_z_reflection(self):
        """This returns a Vertical_Tail class or subclass object that is the reflection
        of this object over the x-z plane. This is useful since if Vertical_Tail's symmetric 
        attribute is True, the symmetric wing gets reflected over the x-y plane.
        
        This function uses deepcopy to achieve its purpose. If this copies too many unwanted 
        attributes, it is recommended that the user should write their own code, taking 
        after the form of this function.
        
        It is also recommended that the user call this function after they set control surface
        or all moving surface deflections. This way the deflection is also properly reflected 
        to the other side
    
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
        wing                  = super().make_x_z_reflection()
        wing.deflection      *= -1*self.sign_duplicate
        wing.hinge_vector[1] *= -1
        return wing
