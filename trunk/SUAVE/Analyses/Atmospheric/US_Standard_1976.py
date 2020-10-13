## @ingroup Analyses-Atmospheric
# US_Standard_1976.py
#
# Created: 
# Modified: Feb 2016, Andrew Wendorff
#           Jan 2018, W. Maier

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from warnings import warn

import SUAVE

from SUAVE.Analyses.Atmospheric import Atmospheric

from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Planets import Earth

from SUAVE.Analyses.Mission.Segments.Conditions import Conditions

from SUAVE.Core import Units
from SUAVE.Core.Arrays import atleast_2d_col


# ----------------------------------------------------------------------
#  Classes
# ----------------------------------------------------------------------

## @ingroup Analyses-Atmospheric
class US_Standard_1976(Atmospheric):

    """ Implements the U.S. Standard Atmosphere (1976 version)
        
    Assumptions:
    None
    
    Source:
    U.S. Standard Atmosphere, 1976, U.S. Government Printing Office, Washington, D.C., 1976
    """
    
    def __defaults__(self):
        """This sets the default values for the analysis to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Output:
        None

        Properties Used:
        None
        """     
        
        atmo_data = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
        self.update(atmo_data)        
    
    def compute_values(self,altitude,temperature_deviation=0.0,var_gamma=False):

        """Computes atmospheric values.

        Assumptions:
        US 1976 Standard Atmosphere

        Source:
        U.S. Standard Atmosphere, 1976, U.S. Government Printing Office, Washington, D.C., 1976

        Inputs:
        altitude                                 [m]
        temperature_deviation                    [K]

        Output:
        atmo_data.
          pressure                               [Pa]
          temperature                            [K]
          speed_of_sound                         [m/s]
          dynamic_viscosity                      [kg/(m*s)]

        Properties Used:
        self.
          fluid_properties.gas_specific_constant [J/(kg*K)]
          planet.sea_level_gravity               [m/s^2]
          planet.mean_radius                     [m]
          breaks.
            altitude                             [m]
            temperature                          [K]
            pressure                             [Pa]
        """

        # unpack
        zs        = altitude
        gas       = self.fluid_properties
        planet    = self.planet
        grav      = self.planet.sea_level_gravity        
        Rad       = self.planet.mean_radius
        R         = gas.gas_specific_constant
        delta_isa = temperature_deviation
        
        # check properties
        if not gas == Air():
            warn('US Standard Atmosphere not using Air fluid properties')
        if not planet == Earth():
            warn('US Standard Atmosphere not using Earth planet properties')          
        
        # convert input if necessary
        zs = atleast_2d_col(zs)

        # get model altitude bounds
        zmin = self.breaks.altitude[0]
        zmax = self.breaks.altitude[-1]   
        
        # convert geometric to geopotential altitude
        zs = zs/(1 + zs/Rad)
        
        # check ranges
        if np.amin(zs) < zmin:
            print("Warning: altitude requested below minimum for this atmospheric model; returning values for h = -2.0 km")
            zs[zs < zmin] = zmin
        if np.amax(zs) > zmax:
            print("Warning: altitude requested above maximum for this atmospheric model; returning values for h = 86.0 km")   
            zs[zs > zmax] = zmax        

        # initialize return data
        zeros = np.zeros_like(zs)
        p     = zeros * 0.0
        T     = zeros * 0.0
        rho   = zeros * 0.0
        a     = zeros * 0.0
        mu    = zeros * 0.0
        z0    = zeros * 0.0
        T0    = zeros * 0.0
        p0    = zeros * 0.0
        alpha = zeros * 0.0
        
        # populate the altitude breaks
        # this uses >= and <= to capture both edges and because values should be the same at the edges
        for i in range( len(self.breaks.altitude)-1 ): 
            i_inside = (zs >= self.breaks.altitude[i]) & (zs <= self.breaks.altitude[i+1])
            z0[ i_inside ]    = self.breaks.altitude[i]
            T0[ i_inside ]    = self.breaks.temperature[i]
            p0[ i_inside ]    = self.breaks.pressure[i]
            alpha[ i_inside ] = -(self.breaks.temperature[i+1] - self.breaks.temperature[i])/ \
                                 (self.breaks.altitude[i+1]    - self.breaks.altitude[i])
        
        # interpolate the breaks
        dz = zs-z0
        i_isoth = (alpha == 0.)
        i_adiab = (alpha != 0.)
        p[i_isoth] = p0[i_isoth] * np.exp(-1.*dz[i_isoth]*grav/(R*T0[i_isoth]))
        p[i_adiab] = p0[i_adiab] * ( (1.-alpha[i_adiab]*dz[i_adiab]/T0[i_adiab]) **(1.*grav/(alpha[i_adiab]*R)) )
        
        T   = T0 - dz*alpha + delta_isa
        rho = gas.compute_density(T,p)
        a   = gas.compute_speed_of_sound(T,p,var_gamma)
        mu  = gas.compute_absolute_viscosity(T)
        gamma = gas.compute_gamma(T,p)
                
        atmo_data = Conditions()
        atmo_data.expand_rows(zs.shape[0])
        atmo_data.pressure                = p
        atmo_data.temperature             = T
        atmo_data.density                 = rho
        atmo_data.speed_of_sound          = a
        atmo_data.dynamic_viscosity       = mu
        atmo_data.ratio_of_specific_heats = gamma
        
        return atmo_data


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    import pylab as plt
    
    h = np.linspace(-1.,60.,200) * Units.km
    delta_isa = 0.
    h = 5000.
    atmosphere = US_Standard_1976()
    
    data = atmosphere.compute_values(h,delta_isa)
    p   = data.pressure
    T   = data.temperature
    rho = data.density
    a   = data.speed_of_sound
    mu = data.dynamic_viscosity
    
    print(data)
    