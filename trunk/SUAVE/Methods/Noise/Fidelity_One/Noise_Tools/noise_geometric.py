## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# noise_geometric.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data , Units
import numpy as np

# ----------------------------------------------------------------------
#   Noise Geometric
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def noise_geometric(noise_segment,analyses,config,mic_loc = 0):
    """ This computes the geometric parameters for the noise tools: distance and emission angles for 
    both polar and azimuthal angles.
     
    Assumptions:
        For sideline condition we assume the maximum noise at takeoff occurs at 1000ft from the ground.     
        
    Inputs:
        noise_segment	 - SUAVE type vehicle
        analyses
        config
        mic_loc          - ground microphone index
    
    Outputs:
        dist  - Distance vector from the aircraft position in relation to the microphone coordinates,    [meters]
        theta - Polar angle emission vector relatively to the aircraft to the microphone coordinates,     [rad]
        phi   - Azimuthal angle emission vector relatively to the aircraft to the microphone coordinates, [rad]

    Properties Used:
        None     
    """
    
    # unpack
    sideline = analyses.noise.settings.sideline
    flyover  = analyses.noise.settings.flyover
    approach = analyses.noise.settings.approach
    x0       = analyses.noise.settings.mic_x_position # only sideline
    
    position_vector = noise_segment.conditions.frames.inertial.position_vector 
    altitude        = -noise_segment.conditions.frames.inertial.position_vector[:,2]
    
    s       = position_vector[:,0]
    s[s==0] = 1E-8
    n_steps = len(altitude)  # number of time steps (space discretization)
       
    if approach == True:
        
        #--------------------------------------------------------
        #-------------------APPROACH CALCULATION-----------------
        #--------------------------------------------------------
        
        # Azimuthal angle is zero for approach condition
        phi   = np.zeros(n_steps)
        theta = np.zeros(n_steps)  
        
        # Microphone position from the approach threshold
        x0 = 2000.
       
        # Calculation of the distance vector and emission angle
        dist  = np.sqrt(altitude**2+(s-x0)**2)

        for i in range(0, n_steps):
            if (s[i]-x0)< 0.:
                theta[i] = np.arctan(np.abs(altitude[i]/(s[i]-x0)))
            else:
                theta[i] = np.pi - np.arctan(np.abs(altitude[i]/(s[i]-x0)))
        
    elif flyover == True:
        
        #--------------------------------------------------------
        #------------------ FLYOVER CALCULATION -----------------
        #--------------------------------------------------------
        
        # Azimuthal angle is zero for flyover condition
        phi   = np.zeros(n_steps)    
        theta = np.zeros(n_steps)  
        
        # Lift-off position from the brake release    
        estimate_tofl = SUAVE.Methods.Performance.estimate_take_off_field_length
    
        # Defining required data for tofl evaluation S0
        takeoff_airport      = SUAVE.Attributes.Airports.Airport()        
        atmo                 = Data()
        atmo.base            = Data()
        atmo.base.atmosphere = analyses.atmosphere
        
        S_0 = estimate_tofl(config,analyses,takeoff_airport)           

        # Microphone position from the brake release point
        x0 = np.float(6500. - S_0)
        
        # Calculation of the distance vector and emission angle
        dist  = np.sqrt(altitude**2+(s-x0)**2)

        for i in range(0, n_steps):
            if (s[i]-x0)< 0.:
                theta[i] = np.arctan(np.abs(altitude[i]/(s[i]-x0)))
            else:
                theta[i] = np.pi - np.arctan(np.abs(altitude[i]/(s[i]-x0)))        
        
    elif sideline == True:
        
        #--------------------------------------------------------
        #-------------------SIDELINE CALCULATION-----------------
        #--------------------------------------------------------        
        
        theta = np.zeros(n_steps) 

        y0 = 450.  # position on the y-direction of the sideline microphone (lateral coordinate)
        

        estimate_tofl = SUAVE.Methods.Performance.estimate_take_off_field_length
        
        # defining required data for tofl evaluation
        takeoff_airport      = SUAVE.Attributes.Airports.Airport()    
        S_0 = estimate_tofl(config,analyses,takeoff_airport)   
 
        
        # looking for X coordinate for 1000ft altitude
        if not x0:
            if position_vector[-1,2] > -304.8 or position_vector[0,2] < -304.8:
                degree = 3
                coefs = np.polyfit(-position_vector[:,2],position_vector[:,0],degree)
                x0 = 0.
                for idx,coef in enumerate(coefs):
                    x0 += coef * 304.8 ** (degree-idx)
            else:
                x0 = np.interp(304.8,np.abs(position_vector[:,2]),position_vector[:,0])
        
        # Calculation of the distance vector and emission angles phi and theta
        phi   = np.arctan(y0/altitude)
        dist  = np.sqrt((y0/np.sin(phi))**2+(s-x0)**2)

        for i in range(0, n_steps):
            if (s[i]-x0)< 0.:
                theta[i] = np.arccos(np.abs((x0-s[i])/dist[i]))
            else:
                theta[i] = np.pi - np.arccos(np.abs((x0-s[i])/dist[i]))  
    
    else: 
        
        #--------------------------------------------------------
        #-------------------ARBITRARY LOCATION -----------------
        #--------------------------------------------------------        

        theta = np.zeros(n_steps) 

        y0 = noise_segment.conditions.noise.microphone_locations[:,mic_loc,1] # position on the y-direction of the sideline microphone (lateral coordinate)
        x0 = noise_segment.conditions.noise.microphone_locations[:,mic_loc,0] # position on the x-direction of the sideline microphone (lateral coordinate)
        y0[y0==0] = 1E-8
        x0[x0==0] = 1E-8

        estimate_tofl = SUAVE.Methods.Performance.estimate_take_off_field_length
        
        # Calculation of the distance vector and emission angles phi and theta
        phi   = np.arctan(y0/altitude)
        dist  = np.sqrt((y0/np.sin(phi))**2+(s-x0)**2)

        for i in range(0, n_steps):
            if (s[i]-x0[i]) < 0.:
                theta[i] = np.arccos(np.abs((x0[i]-s[i])/dist[i]))
            else:
                theta[i] = np.pi - np.arccos(np.abs((x0[i]-s[i])/dist[i]))  

    # Pack the results in Noise Segments    
    noise_segment.dist  = dist
    noise_segment.theta = theta
    noise_segment.phi   = phi

    return noise_segment