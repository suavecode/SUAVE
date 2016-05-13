# SUAVE/__init__.py
#

""" SUAVE Package Setup
"""

# ----------------------------------------------------------------------
#  IMPORT!!
# ----------------------------------------------------------------------

# packages
import Plugins
import Core
import Methods
import Attributes
import Components
import Analyses
import Optimization
import Surrogate
import Input_Output

# the vehicle class
from Vehicle import Vehicle

from warnings import simplefilter
simplefilter('ignore')
