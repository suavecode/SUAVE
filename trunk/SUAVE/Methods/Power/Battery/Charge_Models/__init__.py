## @defgroup Methods-Power-Battery-Charge Charge
# Functions to evaluate battery charge losses and voltage requirements
# @ingroup Methods-Power-Battery

from .datta_charge    import datta_charge
from .LiNCA_thevenin_charge import LiNCA_thevenin_charge
from .LiNiMnCo_charge import LiNiMnCo_charge