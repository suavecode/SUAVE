# Tim Momose, October 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Core import Container as Container_Base

# ------------------------------------------------------------
#   Configuration
# ------------------------------------------------------------

#class Cases(Data):

    #def __defaults__(self):

        #self.num_cases = 0
        #self.cases = Data()


    #def append_case(self,case):
        #""" adds a case to the set of run cases """

        ## assert database type
        #if not isinstance(case,Data):
            #raise Component_Exception, 'input component must be of type Data()'

        ## store data with the appropriate case index
        ## AVL uses indices starting from 1, not 0!
        #self.num_cases += 1
        #case.index = self.num_cases
        #self.cases.append(case)

        #return

# ------------------------------------------------------------
#  AVL Case
# ------------------------------------------------------------

class Run_Case(Data):
    def __defaults__(self):
        """
        OUTPUTS:
        	- 'aerodynamic' (CL, CD, CM)
        	- 'body derivatives' (CMa,CNb,Clb,
        	- 'stability derivatives' (

        """

        self.index = 0		# Will be overwritten when passed to an AVL_Callable object
        self.tag   = 'case'
        self.mass  = 0.0

        self.conditions = Data()
        self.stability_and_control = Data()
        free = Data()
        aero = Data()

        free.mach     = 0.0
        free.velocity = 0.0
        free.density  = 1.225
        free.gravitational_acceleration = 9.81

        aero.parasite_drag    = 0.0
        aero.angle_of_attack  = 0.0
        aero.side_slip_angle  = 0.0

        self.stability_and_control.control_deflections = None
        self.conditions.freestream = free
        self.conditions.aerodynamics = aero

        self.result_filename = None


    def append_control_deflection(self,control_tag,deflection):
        """ adds a control deflection case """
        control_deflection = Data()
        control_deflection.tag        = control_tag
        control_deflection.deflection = deflection
        if self.stability_and_control.control_deflections is None:
            self.stability_and_control.control_deflections = Data()
        self.stability_and_control.control_deflections.append(control_deflection)

        return

class Container(Container_Base):

    def append_case(self,case):
        """ adds a case to the set of run cases """
        case.index = len(self)+1
        self.append(case)
        #case = self.check_new_val(case)
        
        ## store data with the appropriate case index
        ## AVL uses indices starting from 1, not 0!
        ##num_cases = len(self)
        #Data.append(self,case)

        return
    
    
# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Run_Case.Container = Container