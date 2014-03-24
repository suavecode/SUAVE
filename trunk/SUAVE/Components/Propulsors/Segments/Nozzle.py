""" Nozzle.py: Nozzle Propulsor Segment """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from Segment import Segment

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Nozzle(Segment):   

    def __defaults__(self):

        self.tag = 'Nozzle'
        self.shaft = False

    def __call__(self):

        raise NotImplementedError