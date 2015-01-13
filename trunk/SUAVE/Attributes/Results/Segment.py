
from SUAVE.Core import Data, Data_Exception
from SUAVE.Core import Container as ContainerBase

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


