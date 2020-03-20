## @defgroup Analyses-Mission-Segments Segment
# Segment analyses to setup each part of a mission to fly
# @ingroup Analyses-Mission


from .Segment     import Segment
from .Simple      import Simple
from .Aerodynamic import Aerodynamic

from . import Climb
from . import Conditions
from . import Cruise
from . import Descent
from . import Ground
from . import Hover
from . import Single_Point
from . import Transition