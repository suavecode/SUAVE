## @defgroup Methods-Power-Fuel_Cell-Discharge Discharge
# Functions to evaluate fuel cell discharge losses and voltage requirements
# @ingroup Methods-Power-Fuel_Cell

from .zero_fidelity import zero_fidelity
from .larminie import larminie

from .setup_larminie import setup_larminie
from .find_voltage_larminie import find_voltage_larminie
from .find_power_larminie import find_power_larminie
from .find_power_diff_larminie import find_power_diff_larminie