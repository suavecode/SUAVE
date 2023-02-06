## @defgroup Components
# Components are classes that represent objects that are put together to form a vehicle.
# They contain default variables and may contain functions that operate on these variables.

# classes
from .Component import Component

from .Mass_Properties import Mass_Properties
from .Physical_Component import Physical_Component

from .Lofted_Body import Lofted_Body
from .Envelope import Envelope

# packages
from . import Wings
from . import Fuselages
from . import Payloads
from . import Energy
from . import Systems
from . import Nacelles
from . import Configs
from . import Landing_Gear
from . import Costs