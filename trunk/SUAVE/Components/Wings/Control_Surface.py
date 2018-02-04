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
from SUAVE.Components import Lofted_Body

# ------------------------------------------------------------
#  Control Surfaces
# ------------------------------------------------------------

## @ingroup Components-Wings
class Control_Surface(Lofted_Body.Control_Surface):
    def __defaults__(self):
        """This sets the default values of control surfaces defined in SUAVE.

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
	self.span_fraction         = [0.0,0.0] # [absolute percent span location at start of control surface, absolute percent span location at end of control surface]
	self.chord_fraction        = 0.0  
	self.deflection            = 0.0  # Units in degrees
	self.origin                = [0.0,0.0,0.0]
	self.transformation_matrix = [[1,0,0],[0,1,0],[0,0,1]]	
	self.deflection_symmetry   = 1.0    # sign_duplicate: 1.0 or -1.0 - the sign of
					    # the duplicate control on the mirror wing.
					    # Use 1.0 for a mirrored control surface,
					    # like an elevator. Use -1.0 for an aileron.

        self.sections = Data()
        

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
            raise Exception, 'input control surface section must be of type Data()'

        # store data
        self.sections.append(section)

        return

## @ingroup Components-Wings
class Control_Surface_Container(Lofted_Body.Control_Surface.Container):
    """ Container for wing segment

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

    pass