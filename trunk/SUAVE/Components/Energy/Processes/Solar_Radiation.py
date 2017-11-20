## @ingroup Components-Energy-Processes
# Solar_Radiation.py
# 
# Created:  Jun 2014, E. Botero
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Solar Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Processes
class Solar_Radiation(Energy_Component):
    """A class that handle solar radiation computation.
    
    Assumptions:
    None
    
    Source:
    N/A
    """     
    def solar_radiation(self,conditions):  
        """Computes the adjusted solar flux

        Assumptions:
        Solar intensity =1305 W/m^2
        Includes a diffuse component of 0% of the direct component
        Altitudes are not excessive 

        Source:
        N/A

        Inputs:
        conditions.frames.
          planet.start_time        [s]
          planet.latitude          [degrees]
          planet.longitude         [degrees]
          body.inertial_rotations  [radians]
          inertial.time            [s]
        conditions.freestream.
          altitude                 [m]

        Outputs:
        self.outputs.flux          [W/m^2]
        flux                       [W/m^2]

        Properties Used:
        N/A
        """            
        
        # Unpack
        timedate  = conditions.frames.planet.start_time
        latitude  = conditions.frames.planet.latitude
        longitude = conditions.frames.planet.longitude
        phip      = conditions.frames.body.inertial_rotations[:,0,None]
        thetap    = conditions.frames.body.inertial_rotations[:,1,None]
        psip      = conditions.frames.body.inertial_rotations[:,2,None]
        altitude  = conditions.freestream.altitude
        times     = conditions.frames.inertial.time
        
        # Figure out the date and time
        day       = timedate.tm_yday + np.floor_divide(times, 24.*60.*60.)
        TUTC      = timedate.tm_sec + 60.*timedate.tm_min+ 60.*60.*timedate.tm_hour + np.mod(times,24.*60.*60.)
        
        # Gamma is defined to be due south, so
        gamma = psip - np.pi
        
        # Solar intensity external to the Earths atmosphere
        Io = 1367.0
        
        # Indirect component adjustment
        Ind = 1.0
        
        # B
        B = (360./365.0)*(day-81.)*np.pi/180.0
        
        # Equation of Time
        EoT = 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
        
        # Time Correction factor
        TC = 4*longitude+EoT
        
        # Local Solar Time
        LST = TUTC/3600.0+TC/60.0
        
        # Hour Angle   
        HRA = (15.0*(LST-12.0))*np.pi/180.0
        
        # Declination angle (rad)
        delta = -23.44*np.cos((360./365.)*(day+10.)*np.pi/180.)*np.pi/180.
        
        # Zenith angle (rad)
        psi = np.arccos(np.sin(delta)*np.sin(latitude*np.pi/180.0)+np.cos(delta)*np.cos(latitude*np.pi/180.0)*np.cos(HRA))
        
        # Solar Azimuth angle, Duffie/Beckman 1.6.6
        gammas = np.sign(HRA)*np.abs((np.cos(psi)*np.sin(latitude*np.pi/180.)-np.sin(delta))/(np.sin(psi)*np.cos(latitude*np.pi/180)))
        
        # Slope of the solar panel, Bower AIAA 2011-7072 EQN 15
        beta = np.reshape(np.arccos(np.cos(thetap)*np.cos(phip)),np.shape(gammas))
        
        # Angle of incidence, Duffie/Beckman 1.6.3
        theta = np.arccos(np.cos(psi)*np.cos(beta)+np.sin(psi)*np.sin(beta)*np.cos(gammas-gamma))
        
        flux = np.zeros_like(psi)
        
        earth = SUAVE.Attributes.Planets.Earth()
        Re = earth.mean_radius
        
        # Atmospheric properties
        Yatm = 9. # The atmospheres thickness in km
        r    = Re/Yatm
        c    = altitude/9000.
        
        # Air mass
        AM = np.zeros_like(psi)
        AM[altitude<9000.] = (((r+c[altitude<9000.])*(r+c[altitude<9000.]))*(np.cos(psi[altitude<9000.])*np.cos(psi[altitude<9000.]))+2.*r*(1.-c[altitude<9000.])-c[altitude<9000.]*c[altitude<9000.] +1.)**(0.5)-(r+c[altitude<9000.])*np.cos(psi[altitude<9000.])
        
        # Direct component 
        Id = Ind*Io*(0.7**(AM**0.678))
        
        # Horizontal flux
        Ih = Id*(np.cos(latitude*np.pi/180.)*np.cos(delta)*np.cos(HRA)+np.sin(latitude*np.pi/180.)*np.sin(delta))              
        
        # Flux on the inclined panel, if the altitude is less than 9000 meters
        I = Ih*np.cos(theta)/np.cos(psi)
        
        # Now update if the plane is outside the majority of the atmosphere, (9km)
        Id = Ind*Io
        Ih = Id*(np.cos(latitude*np.pi/180.)*np.cos(delta)*np.cos(HRA)+np.sin(latitude*np.pi/180.)*np.sin(delta))           
        I[altitude>9000.] = Ih[altitude>9000.]*np.cos(theta[altitude>9000.])/np.cos(psi[altitude>9000.])
        
        # Now if the sun is on the other side of the Earth...
        I[((psi<-np.pi/2.)|(psi>96.70995*np.pi/180.))] = 0
        
        flux = np.maximum(0.0,I)        
        
        # Store to outputs
        self.outputs.flux = flux      
        
        # Return result for fun/convenience
        return flux