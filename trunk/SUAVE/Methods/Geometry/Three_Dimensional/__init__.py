## @defgroup Methods-Geometry-Three_Dimensional Three Dimensional
# Geometry functions for three dimensions.
# @ingroup Methods-Geometry

from .angles_to_dcms                         import angles_to_dcms
from .orientation_product                    import orientation_product
from .orientation_transpose                  import orientation_transpose
from .estimate_naca_4_series_internal_volume import estimate_naca_4_series_internal_volume
from .compute_span_location_from_chord_length import compute_span_location_from_chord_length
from .compute_chord_length_from_span_location import compute_chord_length_from_span_location