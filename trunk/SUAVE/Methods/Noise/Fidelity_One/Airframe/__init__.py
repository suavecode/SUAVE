## @defgroup Methods-Noise-Fidelity_One-Airframe Airframe
# Fidelity One level noise calculations for the airframe components
# @ingroup Methods-Noise-Fidelity_One

from .noise_airframe_Fink import noise_airframe_Fink
from .noise_clean_wing import noise_clean_wing
from . import noise_landing_gear
from . import noise_leading_edge_slat
from . import noise_trailing_edge_flap
