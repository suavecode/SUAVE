## @defgroup Core
# Core is all the under the hood magic that makes SUAVE work.

from .Arrays import *

from .Data             import Data
from .DataOrdered      import DataOrdered
from .Diffed_Data      import Diffed_Data, diff
from .Container        import Container
from .ContainerOrdered import ContainerOrdered
from .JAX              import to_jnumpy, to_numpy

from .Units import Units