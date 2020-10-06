## @defgroup Methods-Weights-Correlations-Raymer
# Provides structural weight correlations for aircraft components based on the Raymer method
# @ingroup Methods-Weights-Correlations

from .wing_main_raymer import wing_main_raymer
from .tail import tail_horizontal_Raymer, tail_vertical_Raymer
from .fuselage import fuselage_weight_Raymer
from .landing_gear import landing_gear_Raymer
from .systems import systems_Raymer
from .prop_system import total_prop_Raymer
