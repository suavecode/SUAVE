## @defgroup Methods-Missions-Segments Segments
# Mission Segment folders containing the functions for setting up and solving a mission.
# @ingroup Methods-Missions

from converge_root import converge_root
from expand_state  import expand_state
from optimize      import converge_opt

import Common
import Cruise
import Climb
import Descent
import Ground
import Hover
import Single_Point