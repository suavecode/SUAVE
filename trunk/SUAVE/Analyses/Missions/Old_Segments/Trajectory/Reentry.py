""" Reentry.py: general flight with thrust control vector """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Analyses.Missions.Segments import Segment
from SUAVE.Methods.Flight_Dynamics import equations_of_motion

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Reentry(Segment):

    """ Reentry Segment: general flight with thrust control vector """

    def __defaults__(self):
        self.tag = 'Reentry Segment'

    def initialize(self):
        raise NotImplementedError

    def unpack(self,x):
        raise NotImplementedError

    def dynamics(self,x_state,x_control,D,I):
        raise NotImplementedError

    def constraints(self,x_state,x_control,D,I):
        raise NotImplementedError

    def solution(self,x):
        raise NotImplementedError