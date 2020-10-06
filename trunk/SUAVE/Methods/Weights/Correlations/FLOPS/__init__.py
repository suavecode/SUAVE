## @defgroup Methods-Weights-Correlations-FLOPS
# Provides structural weight correlations for aircraft components based on the FLOPS method
# @ingroup Methods-Weights-Correlations

from .fuselage import fuselage_weight_FLOPS
from .landing_gear import landing_gear_FLOPS
from .operating_items import operating_items_FLOPS
from .payload import payload_FLOPS
from .prop_system import fuel_system_FLOPS, nacelle_FLOPS, thrust_reverser_FLOPS, misc_engine_FLOPS, engine_FLOPS
from .systems import systems_FLOPS
from .tail import tail_vertical_FLOPS, tail_horizontal_FLOPS
from .wing import wing_weight_FLOPS, wing_weight_constants_FLOPS

