## @defgroup Methods-Power-Battery-Discharge Discharge
# Functions to evaluate battery discharge losses and voltage requirements
# @ingroup Methods-Power-Battery

from .datta_discharge    import datta_discharge
from .LiNCA_thevenin_discharge import LiNCA_thevenin_discharge
from .LiNiMnCo_discharge import LiNiMnCo_discharge