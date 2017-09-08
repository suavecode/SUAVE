## @defgroup Components-Energy-Networks Networks
# Components used in energy networks.
# These scripts are the blue prints the connect the component of your energy system. The mission will call these
# at each iteration to calculate thrust and a mass flow rate.
# @ingroup Components-Energy

from Solar import Solar
from Ducted_Fan import Ducted_Fan
from Battery_Ducted_Fan import Battery_Ducted_Fan 
from Turbofan import Turbofan
from Turbojet_Super import Turbojet_Super
from Solar_Low_Fidelity import Solar_Low_Fidelity
from Dual_Battery_Ducted_Fan import Dual_Battery_Ducted_Fan
from Ramjet import Ramjet
from Propulsor_Surrogate import Propulsor_Surrogate