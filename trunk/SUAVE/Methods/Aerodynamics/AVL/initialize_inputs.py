## @ingroup Methods-Aerodynamics-AVL
#initialize_inputs.py
# 
# Created:  Oct 2014, T. Momose
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
from SUAVE.Core import Data
# SUAVE-AVL Imports
from .create_avl_datastructure import create_avl_datastructure
from .write_geometry           import write_geometry
from .write_run_cases          import write_run_cases
from .write_input_deck         import write_input_deck

## @ingroup Methods-Aerodynamics-AVL
def initialize_inputs(geometry,configuration,conditions):
        """ This intializes the functions used in the AVL class

        Assumptions:
                None

        Source:
                None

        Inputs:
                avl_inputs - passed into the write_geometry, write_run_cases and write_input_deck functions

        Outputs:
                avl_imputs

        Properties Used:
                N/A
        """    	

        avl_inputs = create_avl_datastructure(geometry,configuration,conditions)
        avl_inputs = write_geometry(avl_inputs)
        avl_inputs = write_run_cases(avl_inputs)
        avl_inputs = write_input_deck(avl_inputs)

        return avl_inputs
