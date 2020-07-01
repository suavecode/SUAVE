## @defgroup Methods-Aerodynamics-AVL AVL
# Functions to AVL calculations
# @ingroup Methods-Aerodynamics

""" SUAVE AVL Interface Package Setup
"""

from .create_avl_datastructure import translate_avl_wing, translate_avl_body , populate_wing_sections, populate_body_sections
from .purge_files              import purge_files
from .read_results             import read_results
from .run_analysis             import run_analysis
from .translate_data           import translate_conditions_to_cases, translate_results_to_conditions
from .write_geometry           import write_geometry
from .write_mass_file          import write_mass_file
from .write_input_deck         import write_input_deck
from .write_run_cases          import write_run_cases
from .write_avl_airfoil_file   import write_avl_airfoil_file

from . import Data
