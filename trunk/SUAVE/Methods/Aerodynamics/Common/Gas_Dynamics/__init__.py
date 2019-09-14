## @defgroup Methods-Aerodynamics-Common-Gas_Dynamics Gas_Dymamics
# Gas Dynamics methods that are directly specified by analyses.
# @ingroup Methods-Aerodynamics-Common-Gas_Dynamics

from .Conical_Shock import get_Ms, get_Cp, get_beta, get_invisc_press_recov
from .Isentropic import isentropic_relations, get_m
from .Oblique_Shock import oblique_shock_relations, theta_beta_mach