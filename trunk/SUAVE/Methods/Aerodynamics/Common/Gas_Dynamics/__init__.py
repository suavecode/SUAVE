## @defgroup Methods-Aerodynamics-Common-Gas_Dynamics Gas_Dymamics
# Gas Dynamics methods that are directly specified by analyses.
# @ingroup Methods-Aerodynamics-Common-Gas_Dynamics

from .Isentropic import isentropic_relations, get_m
from .Oblique_Shock import oblique_shock_relations, theta_beta_mach