## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# print_propeller_output.py
# 
# Created:  Oct 2020, M. Clarke 
# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import numpy as np
from SUAVE.Core            import Units  

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def print_propeller_output(speed,nsteps,time,altitude, RPM,theta ,dist ,PNL,PNL_dBA): 
    """This prints the noise of a propeller aircraft using SAE noise analysis methods

    Assumptions:
       N/A

    Inputs:
    speed     aircraft speed                     [m/s]
    nsteps    numer of timesteps                 [unitless]
    time      time                               [s]
    altitude  aircraft altitude                  [m]
    RPM       rpm of propeller                   [unitless]
    theta     emission angle                     [rad]
    dist      emission distance                  [m]
    PNL       perceived noise level              [dB]
    PNL_dBA   A -weighted perceived noise level  [dBa]

    Outputs:  
        N/A
        
    Properties Used:
        None 
    """ 

    fid = open('prop_test.dat','w')   # Open output file    
    
    fid.write('Reference speed =  ')
    fid.write(str('%2.2f' % (speed[-1]/Units.kts))+'  kts')
    fid.write('\n')
    fid.write('PNLT history')
    fid.write('\n')
    fid.write('time       altitude      Mach    RPM     Polar_angle    distance        PNL  	   dBA')
    fid.write('\n')
    
    for id in range (0,nsteps):
        fid.write(str('%2.2f' % time[id])+'        ')
        fid.write(str('%2.2f' % altitude [id])+'        ')
        fid.write(str('%2.2f' % speed[id])+'        ')
        fid.write(str('%2.2f' % (RPM[id]))+'        ')
        fid.write(str('%2.2f' % (theta[id]*180/np.pi))+'        ')
        fid.write(str('%2.2f' % dist[id])+'        ')
        fid.write(str('%2.2f' % PNL[id])+'        ')
        fid.write(str('%2.2f' % PNL_dBA[id])+'        ')
        fid.write('\n')
    fid.write('\n')
    fid.write('PNLT max =  ')
    fid.write(str('%2.2f' % (np.max(PNL)))+'  dB')
    fid.write('\n')
    fid.write('dBA max =  ')
    fid.write(str('%2.2f' % (np.max(PNL_dBA)))+'  dBA')             
    fid.close      
    return