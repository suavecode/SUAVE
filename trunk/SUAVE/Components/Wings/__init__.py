## @defgroup Components-Wings Wings
# @ingroup Components
#
# __init__.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           Aug 2019, M. Clarke
# classes
from .Wing            import Wing
from .Control_Surface import Control_Surface
from .Control_Surface import append_ctrl_surf_to_wing_segments 
from .Main_Wing       import Main_Wing 
from .Segment         import Segment, SegmentContainer

# packages
from . import Airfoils

