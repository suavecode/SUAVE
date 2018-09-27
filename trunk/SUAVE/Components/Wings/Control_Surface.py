## @ingroup Components-Wings
# Control_Surface.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           Jun 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


from SUAVE.Core import Data
from SUAVE.Components import Component
from SUAVE.Components import Lofted_Body

# ------------------------------------------------------------
#  Control Surfaces
# ------------------------------------------------------------

## @ingroup Components-Wings
class Control_Surface(Component):
    def __defaults__(self):
        """This sets the default values of control surfaces defined in SUAVE. 
        sign_duplicate: 1.0 or -1.0 - the sign of the duplicate control on the mirror wing.
        Use 1.0 for a mirrored control surface, like an elevator. Use -1.0 for an aileron.
        The span fraction is given by the array shown below:  
        [abs. % span location at beginning of crtl surf, abs. % span location at end  of crtl surf]

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

        self.tag                   = 'control_surface'
        self.span                  = 0.0
        self.span_fraction         = [0.0,0.0] 
        self.chord_fraction        = 0.0  
        self.deflection_gain       = 0.0  
        self.origin                = [0.0,0.0,0.0]
        self.transformation_matrix = [[1,0,0],[0,1,0],[0,0,1]]	
        self.deflection_symmetry   = 1.0    
        self.sections = Data()  
        self.prev = None
        self.next = None # for connectivity

    def append_section(self,section):	
        """Adds a section	

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

        # assert database type	
        if not isinstance(section,Data):	
            raise Exception('input control surface section must be of type Data()')	

        # store data	
        self.sections.append(section)	

        return	


## @ingroup Components-Wings	
class Control_Surface_Section(Lofted_Body.Section):	
    def __defaults__(self):	
        """This sets the default values control surface sections defined in SUAVE.	

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

        self.tag            = 'control_section'	
        self.chord          = 0.0	
        self.chord_fraction = 0.0	
        self.twist          = 0.0 # Offset / deflection in neutral position	

        self.origins = Data()	
        self.origins.dimensional    = [0.0,0.0,0.0]	
        self.origins.span_fraction  = 0.0	
        self.origins.chord_fraction = 0.0	

        self.points = Data()	
        self.points.leading_edge  = 0.0	
        self.points.trailing_edge = 0.0	
