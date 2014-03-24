""" Fan.py: Fan Propulsor Segment """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from Segment import Segment

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Fan(Segment):   

    def __defaults__(self):

        self.tag = 'Fan'

    def __call__(self):

        raise NotImplementedError