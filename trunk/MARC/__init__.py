# MARC/__init__.py
#

""" MARC Package Setup
"""

# ----------------------------------------------------------------------
#  IMPORT!!
# ----------------------------------------------------------------------

# packages
from . import Plugins
from . import Core
from . import Components
from . import Analyses
from . import Methods
from . import Attributes
from . import Optimization
from . import Input_Output
from . import Visualization

# the vehicle class
from .Vehicle import Vehicle

from warnings import simplefilter
simplefilter('ignore')

from MARC.version import version as __version__
