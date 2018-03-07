## @ingroup Methods-Aerodynamics-AVL-Data
# Cases.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero
#           Jun 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Core import DataOrdered 

# ------------------------------------------------------------
#  AVL Case
# ------------------------------------------------------------

## @ingroup Methods-Aerodynamics-AVL-Data
class Run_Case(Data):
    """ A data class defining the parameters for the analysis cases 
    including angle of attack and mach number 

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        None

    Outputs:
        None

    Properties Used:
        N/A
    """    
    
    def __defaults__(self):
        """Defines the data structure and defaults of aerodynamics coefficients, 
        body derivatives and stability derivatives   

        Assumptions:
            None
    
        Source:
            None
    
        Inputs:
            None
    
        Outputs:
            None
    
        Properties Used:
            N/A
        """ 

        self.index                      = 0		# Will be overwritten when passed to an AVL_Callable object
        self.tag                        = 'case'
        self.mass                       = 0.0

        self.conditions                 = Data()
        self.stability_and_control      = Data()
        free                            = Data()
        aero                            = Data()

        free.mach                       = 0.0
        free.velocity                   = 0.0
        free.density                    = 1.225
        free.gravitational_acceleration = 9.81

        aero.parasite_drag              = 0.0
        aero.angle_of_attack            = 0.0
        aero.side_slip_angle            = 0.0

        self.stability_and_control.control_deflections  = None
        self.stability_and_control.number_control_surfaces = 0
        self.conditions.freestream      = free
        self.conditions.aerodynamics    = aero

        self.result_filename            = None
        self.eigen_result_filename      = None
 

    def append_control_deflection(self,control_tag,deflection):
        """ Adds a control deflection case 

	Assumptions:
	    None
    
	Source:
	    None
    
	Inputs:
	    None
    
	Outputs:
	    None
    
	Properties Used:
	    N/A
	"""         
        control_deflection              = Data()
        control_deflection.tag          = control_tag
        control_deflection.deflection   = deflection
        if self.stability_and_control.control_deflections is None:
            self.stability_and_control.control_deflections = Data()
        self.stability_and_control.control_deflections.append(control_deflection)

        return

class Container(DataOrdered):
    """ A data class for the addition of a cases to the set of run cases

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        None

    Outputs:
        None

    Properties Used:
        N/A
    """    
    def append_case(self,case):
        """ Adds a case to the set of run cases "
        
	Assumptions:
	    None
    
	Source:
	    None
    
	Inputs:
	    None
    
	Outputs:
	    None
    
	Properties Used:
	    N/A
	"""         
        case.index = len(self)+1
        self.append(case)

        return
    
    
# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Run_Case.Container = Container
