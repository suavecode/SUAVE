## @ingroup Components-Wings
# All_Moving_Surface.py
#
# Created:  Jul 2021, A. Blaufox
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components.Lofted_Body import Lofted_Body

import numpy as np

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

## @ingroup Components-Wings
class All_Moving_Surface(Lofted_Body):
    """ This class is used to allow every all-moving control surface class
    (e.g. Stabilator) to inherit from both a type of Wing (Horizontal_Tail
    in the case of a Stabilator) and this class. This, way All_Moving_Surface
    subclasses can inherit necessary functionality without code bloat or 
    lengthy type checking if-statements.
    
    In general, this class should not be used directly, and should only exist
    as one of the parents of another class that also inherits from Wing  
    
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
        """This sets the default for All_Moving_Surface objects in SUAVE.
        
        Attributes also found in Control_Surface:
            see Control_Surface().__defaults__ for an explanation of attributes. Any
            attributes used by this class that are shared with Control_Surface should 
            always adhere to the convention established in Control_Surface.py
    
        Attributes unique to All_Moving_Surface:
        - use_constant_hinge_fraction: false by default. If this is true, the hinge vector 
            will follow a constant chord_fraction allong the wing, regardless of what is set
            for hinge_vector. Note that constant hinge fractions are how hinges are handled for 
            Control_Surfaces. If this attribute is false, the hinge vector is described by
            the hinge_vector attribute
        - hinge_vector: The vector in body-frame that the hingeline points along. By default, 
            it is [0,0,0], and this is taken to mean that the hinge line is normal to the root
            chord, in-plane with the wing. This attribute does nothing if use_constant_hinge_fraction
            is set to True.
        
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
        self.tag = 'All_Moving_Surface_Data_Object'
        
        #describe control surface-like behavior
        self.sign_duplicate        = 1.0
        self.hinge_fraction        = 0.25
        self.deflection            = 0.0   
        
        #attributes unique to All_Moving_Surface
        self.use_constant_hinge_fraction = False
        self.hinge_vector                = np.array([0.,0.,0.])
