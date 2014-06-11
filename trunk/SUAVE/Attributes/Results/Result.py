""" Result.py: container class for simulation results """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data, Container, Data_Exception, Data_Warning
from SUAVE.Structure import Container as ContainerBase
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
        self.Segments[tag] = new_seg
        return

class Container(ContainerBase):
    pass

Result.Container = Container
