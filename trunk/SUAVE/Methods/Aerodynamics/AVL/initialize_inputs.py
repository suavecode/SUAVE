# Tim Momose, October 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
# SUAVE Imports
from SUAVE.Structure import Data, Data_Exception, Data_Warning
# SUAVE-AVL Imports
from create_avl_datastructure import create_avl_datastructure
from write_geometry           import write_geometry
from write_run_cases          import write_run_cases
from write_input_deck         import write_input_deck


def initialize_inputs(geometry,configuration,conditions):
	
	avl_inputs = create_avl_datastructure(geometry,configuration,conditions)
	avl_inputs = write_geometry(avl_inputs)
	avl_inputs = write_run_cases(avl_inputs)
	avl_inputs = write_input_deck(avl_inputs)
	
	return avl_inputs
