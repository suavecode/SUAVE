## @ingroup Attributes-Atmospheres-Earth
# Constant_Temperature.py: 

# Created:  Mar 2014, SUAVE Team
# Modified: Jan 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Atmospheres import Atmosphere
from SUAVE.Attributes.Planets import Earth
from SUAVE.Core import Data
from SUAVE.Core import Units
from SUAVE.Attributes.Atmospheres import Atmosphere

# ----------------------------------------------------------------------
#  Constant_Temperature Atmosphere
# ----------------------------------------------------------------------
## @ingroup Attributes-Atmospheres-Earth
class Constant_Temperature(Atmosphere):
    """Contains US Standard 1976 values with temperature modified to be constant.
    
    Assumptions:
    Constant temperature
    
    Source:
    U.S. Standard Atmosphere (1976 version)
    """
    def __defaults__(self):
        """This sets the default values at breaks in the atmosphere.

        Assumptions:
        Constant temperature

        Source:
        U.S. Standard Atmosphere (1976 version)

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """          
        self.fluid_properties = Air()
        self.planet = Earth()
        self.breaks = Data()
        self.breaks.altitude    = np.array( [-2.00    , 0.00,     11.00,      20.00,      32.00,      47.00,      51.00,      71.00,      84.852]) * Units.km     # m, geopotential altitude
        self.breaks.temperature = np.array( [301.15   , 301.15,    301.15,    301.15,     301.15,     301.15,     301.15,     301.15,     301.15])      # K
        self.breaks.pressure    = np.array( [127774.0 , 101325.0, 22632.1,    5474.89,    868.019,    110.906,    66.9389,    3.95642,    0.3734])      # Pa
        self.breaks.density     = np.array( [1.545586 , 1.2256523,.273764,	 .0662256,	0.0105000 ,	1.3415E-03,	8.0971E-04,	4.78579E-05, 4.51674E-06]) #kg/m^3
    


    
    pass
