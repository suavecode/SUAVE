# Drag Polar
#
# Created:  7/13/14     Tim MacDonald
# Modified: 

import SUAVE
import numpy as np

from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)
from SUAVE.Methods.Aerodynamics.Drag.Correlations import \
     wave_drag_lift, wave_drag_volume

from wave_drag_lift import wave_drag_lift
from wave_drag_volume import wave_drag_volume

# python imports
import os, sys, shutil
import copy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

def drag_polar(conditions,configuration,geometry):
    CL = SUAVE.Methods.Aerodynamics.Lift.compute_aircraft_lift_supersonic(conditions,configuration,geometry)
    CD = compute_aircraft_drag_supersonic(conditions,configuration,geometry)
        