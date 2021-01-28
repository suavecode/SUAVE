## @defgroup Methods-Aerodynamics-AVL-Data Data
# @ingroup Methods-Aerodynamics-AVL

""" SUAVE AVL Data Package Setup
"""

from .Aircraft      import Aircraft
from .Body          import Body
from .Wing          import Wing,Section,Control_Surface, Control_Surface_Results , Control_Surface_Data
from .Cases         import Run_Case
from .Configuration import Configuration
from .Settings      import Settings
from .Inputs        import Inputs