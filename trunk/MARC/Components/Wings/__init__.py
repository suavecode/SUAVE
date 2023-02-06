## @defgroup Components-Wings Wings
# @ingroup Components
#
# __init__.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           Jan 2020, M. Clarke

# classes
from .Wing                      import Wing
from .Main_Wing                 import Main_Wing
from .Vertical_Tail             import Vertical_Tail
from .Horizontal_Tail           import Horizontal_Tail
from .Segment                   import Segment, Segment_Container 
from .All_Moving_Surface        import All_Moving_Surface 
from .Stabilator                import Stabilator
from .Vertical_Tail_All_Moving  import Vertical_Tail_All_Moving

# packages
from . import Control_Surfaces
