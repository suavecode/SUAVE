
from SUAVE.Structure import Data, Data_Exception
from SUAVE.Structure import Container as ContainerBase

# ----------------------------------------------------------------------
#   Segments
# ----------------------------------------------------------------------

class Segment(Data):
    """ One Result Segment """
    def __defaults__(self):
        self.tag = 'Result Segment'
    
class Container(ContainerBase):
    pass

Segment.Container = Container


