## @defgroup Components-Wings Wings
# @ingroup Components
#
# __init__.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           Jan 2020, M. Clarke

# classes
from .Wing import Wing
from .Main_Wing import Main_Wing
from .Vertical_Tail import Vertical_Tail
from .Horizontal_Tail import Horizontal_Tail
from .Segment import Segment, Segment_Container 

# packages
from . import Airfoils
from . import Control_Surfaces
