""" Mission.py: Top-level mission class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception
from SUAVE.Structure import Container as ContainerBase
from Segments import Base_Segment

# ----------------------------------------------------------------------
#   Class
# ----------------------------------------------------------------------

class Mission(Data):
    """ Mission.py: Top-level mission class """

    def __defaults__(self):
        self.tag = 'Mission'
        self.segments = Base_Segment.Container()

    def append_segment(self,segment):
        """ Add a Mission Segment  """
        self.segments.append(segment)
        return

class Container(ContainerBase):
    pass

Mission.Container = Container
