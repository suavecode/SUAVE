## @defgroup Analyses
# Analyses are classes that are comprised of several methods. 
# Methods operate on attributes. This process and structure is described 
# <a href="http://suave.stanford.edu">here</a>.

from Analysis  import Analysis
from Sizing    import Sizing
from Surrogate import Surrogate
from Process   import Process
from Settings  import Settings
from Vehicle   import Vehicle

import Aerodynamics
import Stability
import Energy
import Weights
import Geometry
import Loads
import Mission
import Structures
import Atmospheric
import Planets
import Sizing
import Noise
import Costs