## @defgroup Analyses-Mission-Segments-Climb Climb
# Segments for climbing flight
# @ingroup Analyses-Mission-Segments

from .Constant_Dynamic_Pressure_Constant_Angle import Constant_Dynamic_Pressure_Constant_Angle
from .Constant_Dynamic_Pressure_Constant_Rate  import Constant_Dynamic_Pressure_Constant_Rate
from .Constant_Mach_Constant_Angle             import Constant_Mach_Constant_Angle
from .Constant_Mach_Constant_Rate              import Constant_Mach_Constant_Rate
from .Constant_Mach_Linear_Altitude            import Constant_Mach_Linear_Altitude
from .Constant_Speed_Constant_Angle            import Constant_Speed_Constant_Angle
from .Constant_Speed_Constant_Angle_Noise      import Constant_Speed_Constant_Angle_Noise
from .Constant_Speed_Constant_Rate             import Constant_Speed_Constant_Rate
from .Constant_Speed_Linear_Altitude           import Constant_Speed_Linear_Altitude
from .Constant_Throttle_Constant_Speed         import Constant_Throttle_Constant_Speed
from .Linear_Mach_Constant_Rate                import Linear_Mach_Constant_Rate
from .Linear_Speed_Constant_Rate               import Linear_Speed_Constant_Rate 
from .Constant_EAS_Constant_Rate               import Constant_EAS_Constant_Rate
from .Constant_CAS_Constant_Rate               import Constant_CAS_Constant_Rate
from .Optimized                                import Optimized
from .Unknown_Throttle                         import Unknown_Throttle