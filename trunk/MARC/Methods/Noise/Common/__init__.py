## @defgroup Methods-Noise-Common Common
# Various functions that are used to calculate noise using the fidelity one level
# @ingroup Methods-Noise-Fidelity_One
 
from .atmospheric_attenuation                            import atmospheric_attenuation
from .background_noise                                   import background_noise 
from .noise_tone_correction                              import noise_tone_correction  
from .decibel_arithmetic                                 import pressure_ratio_to_SPL_arithmetic
from .decibel_arithmetic                                 import SPL_arithmetic
from .convert_to_third_octave_band                       import convert_to_third_octave_band 
from .compute_noise_source_coordinates                   import compute_rotor_point_source_coordinates
from .generate_microphone_points                         import generate_zero_elevation_microphone_points
from .generate_microphone_points                         import generate_terrain_elevated_microphone_points
from .compute_relative_noise_evaluation_locations        import compute_relative_noise_evaluation_locations 