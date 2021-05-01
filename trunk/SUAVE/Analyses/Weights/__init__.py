## @defgroup Analyses-Weights Weights
# Classes that represent the types of aircraft configuration weight computations
# @ingroup Analyses

# Attributes
from .Weights                            import Weights
from .Weights_BWB                        import Weights_BWB
from .Weights_Tube_Wing                  import Weights_Tube_Wing
from .Weights_UAV                        import Weights_UAV
from .Weights_eVTOL                      import Weights_eVTOL 

# to be removed 
from .Weights_Electric_Lift_Cruise       import Weights_Electric_Lift_Cruise 
from .Weights_Electric_Multicopter       import Weights_Electric_Multicopter
from .Weights_Electric_Vectored_Thrust   import Weights_Electric_Vectored_Thrust 