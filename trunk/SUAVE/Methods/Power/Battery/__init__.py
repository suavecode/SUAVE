## @defgroup Methods-Power-Battery Battery
# Functions pertaining to battery discharge and sizing
# @ingroup Methods-Power 

from . import Ragone
from . import Sizing
from . import Variable_Mass
from . import Cell_Cycle_Models

# utility funtions 
from .append_initial_battery_conditions     import append_initial_battery_conditions
from .compute_net_generated_battery_heat    import compute_net_generated_battery_heat
from .pack_battery_conditions               import pack_battery_conditions