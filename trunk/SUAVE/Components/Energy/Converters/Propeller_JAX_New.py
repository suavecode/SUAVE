## @ingroup Component-Energy-Converters
# Propeller_JAX.py
#
# Created: Feb 2022, J. Smart
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Rotor_JAX_New import Rotor_JAX

# ----------------------------------------------------------------------
#  Differentiable Propeller Class
# ----------------------------------------------------------------------

##@ingroup Components-Energy-Converters
class Propeller_JAX(Rotor_JAX):
    """This is a propeller component, and is a sub-class of rotor.

    Modified for use with JAX-based autodifferentiation and GPU acceleration

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

        self.tag = 'propeller'
        self.orientation_euler_angles = [0., 0., 0.]  # This is X-direction thrust in vehicle frame
        self.use_2d_analysis = False
        self.variable_pitch = False