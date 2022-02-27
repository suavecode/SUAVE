## @ingroup Analyses-Propulsion
# Rotor_Wake_Fidelity_Zero.py
#
# Created:  Jan 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component


from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_Zero.fidelity_zero_wake_convergence import fidelity_zero_wake_convergence
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_Zero.compute_bevw_induced_velocity import compute_bevw_induced_velocity


# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Propulsion
class Rotor_Wake_Fidelity_Zero(Energy_Component):
    """This is a general rotor wake component. 

    Assumptions:
    None

    Source:
    None
    """
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        self.tag            = 'rotor_wake'
        self.wake_method    = 'Fidelity_Zero'

    
    def evaluate(self,rotor,wake_inputs,conditions):
        """
        
        Wake evaluation is performed using a simplified vortex wake method for Fidelity Zero, 
        following Helmholtz vortex theory.
        
        Assumptions:
        None

        Source:
        Drela, M. "Qprop Formulation", MIT AeroAstro, June 2006
        http://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf

        Inputs:
           self         - rotor wake
           rotor        - SUAVE rotor
           wake_inputs.
              Ua        - Axial velocity
              Ut        - Tangential velocity
              r         - radius distribution
           conditions   - conditions
           
           
        Outputs:
           va  - axially-induced velocity from rotor wake
           vt  - tangentially-induced velocity from rotor wake
        
        Properties Used:
        None
        
        
        """
        
        va, vt = fidelity_zero_wake_convergence(self, rotor, wake_inputs)
            
        return va, vt
    
    def evaluate_wake_velocities(self,rotor,network,geometry,conditions,VD,num_ctrl_pts):
        """
        Links the rotor wake to compute the wake-induced velocities at the vortex distribution
        control points.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
           self         - rotor wake
           rotor        - rotor
           network      - propulsion network
           geometry     - vehicle geometry
           conditions   - conditions
           VD           - vortex distribution
           num_ctrl_pts - number of analysis control points
           
        Outputs:
           prop_V_wake_ind  - induced velocity from rotor wake at (VD.XC, VD.YC, VD.ZC)
        
        Properties Used:
        None
        """  
        
        identical_flag = network.identical_propellers
        
        if network.number_of_propeller_engines == None:
            pass
        else:   
            rots = Data()
            rots.append(rotor)
            rot_V_wake_ind = compute_bevw_induced_velocity(rots,geometry,num_ctrl_pts,conditions,identical_flag)  
        
        return rot_V_wake_ind
    