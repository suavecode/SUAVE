## @defgroup Methods-Propulsion Propulsion
# Description
# @ingroup Methods

from .ducted_fan_sizing import ducted_fan_sizing
from .propeller_design import propeller_design
from .turbofan_emission_index import turbofan_emission_index
from .electric_motor_sizing import size_from_kv, size_from_mass
from .turbofan_sizing import turbofan_sizing
from .turbojet_sizing import turbojet_sizing
from .ramjet_sizing import ramjet_sizing
from .scramjet_sizing import scramjet_sizing
from .fm_id import fm_id
from .fm_solver import fm_solver
from .rayleigh import rayleigh
from .nozzle_calculations import exit_Mach_shock, mach_area, normal_shock, pressure_ratio_isentropic, pressure_ratio_shock_in_nozzle
from . import electric_motor_sizing
from .liquid_rocket_sizing import liquid_rocket_sizing