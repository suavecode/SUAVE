""" Result.py: container class for simulation results """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data, Container, Data_Exception, Data_Warning
from SUAVE.Core import Container as ContainerBase
from Segment import Segment

# ----------------------------------------------------------------------
#  Results Data Class
# ----------------------------------------------------------------------

class Result(Data):
    """ SUAVE.Results()
        Results Data
    """
    def __defaults__(self):
        self.tag = 'Results'
        
        # self.regulation = Data()        # FAA constraints... PASS / FAIL
        # self.fuel_burn = 0.0
        # self.total_power = 0.0

        # Pointer to config
        # Pointer to segment 
        # CL margins 
        # Takeoff field margins
        # 2nd stage climb contraints 
        
        pass

    def add_segment(self,new_seg):
        """ Add a Results Segment  """
        tag = new_seg['tag']
        new_seg = Segment(new_seg)
        self.segments[tag] = new_seg
        return

class Container(ContainerBase):
    pass

Result.Container = Container
