#US_Standard_1976.py

# Created:  Mar, 2014, SUAVE Team
# Modified: Feb, 2015, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Atmospheres import Atmosphere
from SUAVE.Attributes.Planets import Earth
from SUAVE.Core import Data
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  US_Standard_1976 Atmosphere Class
# ----------------------------------------------------------------------

class US_Standard_1976(Atmosphere):
    
    """ Implements the U.S. Standard Atmosphere (1976 version)
    """
    
    def __defaults__(self):
        self.tag = ' U.S. Standard Atmosphere (1976)'

        # break point data: 
        self.fluid_properties = Air()
        self.planet = Earth()
        self.breaks = Data()
        self.breaks.altitude    = np.array( [-2.00    , 0.00,     11.00,      20.00,      32.00,      47.00,      51.00,      71.00,      84.852]) * Units.km     # m, geopotential altitude
        self.breaks.temperature = np.array( [301.15   , 288.15,   216.65,     216.65,     228.65,     270.65,     270.65,     214.65,     186.95])      # K
        self.breaks.pressure    = np.array( [127774.0 , 101325.0, 22632.1,    5474.89,    868.019,    110.906,    66.9389,    3.95642,    0.3734])      # Pa
        self.breaks.density     = np.array( [1.47808e0, 1.2250e0, 3.63918e-1, 8.80349e-2, 1.32250e-2, 1.42753e-3, 8.61606e-4, 6.42099e-5, 6.95792e-6])  # kg/m^3
    

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    import pylab as plt
    
    h = np.linspace(-1.,60.,200) * Units.km
    
    atmosphere = US_Standard_1976()
    
    p, T, rho, a, mew = atmosphere.compute_values(h)
    
    plt.figure(1)
    plt.plot(p,h)
    plt.xlabel('Pressure (Pa)')
    plt.ylabel('Altitude (km)')
    
    plt.figure(2)
    plt.plot(T,h)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (km)')    
    
    plt.figure(3)
    plt.plot(rho,h)
    plt.xlabel('Density (kg/m^3)')
    plt.ylabel('Altitude (km)')       
    
    plt.figure(4)
    plt.plot(a,h)
    plt.xlabel('Speed of Sound (m/s)')
    plt.ylabel('Altitude (km)') 
    
    plt.figure(6)
    plt.plot(mew,h)
    plt.xlabel('Viscosity (kg/m-s)')
    plt.ylabel('Altitude (km)')   

    plt.show()