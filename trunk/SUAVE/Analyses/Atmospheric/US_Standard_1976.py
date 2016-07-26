""" US_Standard_1976.py: U.S. Standard Atmosphere (1976) """
#
#
# Modified by Tim MacDonald 2/16/15  
# Modified: Feb 2016, Andrew Wendorff

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

class US_Standard_1976(Atmospheric):

    """ Implements the U.S. Standard Atmosphere (1976 version)
    """
    
    def __defaults__(self):
        
        atmo_data = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
        self.update(atmo_data)        
    
    def compute_values(self,altitude,temperature_deviation=0.0):

        """ Computes values from the International Standard Atmosphere

        Inputs:
            altitude     : geometric altitude (elevation) (m)
                           can be a float, list or 1D array of floats
            temperature_deviation :  delta_isa
         
        Outputs:
            list of conditions -
                pressure       : static pressure (Pa)
                temperature    : static temperature (K)
                density        : density (kg/m^3)
                speed_of_sound : speed of sound (m/s)
                dynamic_viscosity      : dynamic_viscosity (kg/m-s)
            
        Example:
            atmosphere = SUAVE.Attributes.Atmospheres.Earth.USStandard1976()
            atmosphere.ComputeValues(1000).pressure
          
        """

        # unpack
        zs        = altitude
        gas       = self.fluid_properties
        planet    = self.planet
        grav      = self.planet.sea_level_gravity        
        Rad       = self.planet.mean_radius
        gamma     = gas.gas_specific_constant
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
            print "Warning: altitude requested below minimum for this atmospheric model; returning values for h = -2.0 km"
            zs[zs < zmin] = zmin
        if np.amax(zs) > zmax:
            print "Warning: altitude requested above maximum for this atmospheric model; returning values for h = 86.0 km"   
            zs[zs > zmax] = zmax        

        # initialize return data
        zeros = np.zeros_like(zs)
        p     = zeros * 0.0
        T     = zeros * 0.0
        rho   = zeros * 0.0
        a     = zeros * 0.0
        mew   = zeros * 0.0
        z0    = zeros * 0.0
        T0    = zeros * 0.0
        p0    = zeros * 0.0
        alpha = zeros * 0.0
        
        # populate the altitude breaks
        # this uses >= and <= to capture both edges and because values should be the same at the edges
        for i in xrange( len(self.breaks.altitude)-1 ): 
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
        p[i_isoth] = p0[i_isoth] * np.exp(-1.*dz[i_isoth]*grav/(gamma*T0[i_isoth]))
        p[i_adiab] = p0[i_adiab] * ( (1.-alpha[i_adiab]*dz[i_adiab]/T0[i_adiab]) **(1.*grav/(alpha[i_adiab]*gamma)) )
        
        T   = T0 - dz*alpha + delta_isa
        rho = gas.compute_density(T,p)
        a   = gas.compute_speed_of_sound(T)
        mew = gas.compute_absolute_viscosity(T)
                
        atmo_data = Conditions()
        atmo_data.expand_rows(zs.shape[0])
        atmo_data.pressure          = p
        atmo_data.temperature       = T
        atmo_data.density           = rho
        atmo_data.speed_of_sound    = a
        atmo_data.dynamic_viscosity = mew
        
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
    mew = data.dynamic_viscosity
    
    print data
    