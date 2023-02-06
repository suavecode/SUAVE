## @defgroup Core
# Core is all the under the hood magic that makes MARC work.

from .Arrays import *

from .Data             import Data
from .DataOrdered      import DataOrdered
from .Diffed_Data      import Diffed_Data, diff
from .Container        import Container
from .ContainerOrdered import ContainerOrdered
from .Utilities        import *
from .Units            import Units