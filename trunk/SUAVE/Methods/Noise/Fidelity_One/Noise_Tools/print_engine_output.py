## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# print_engine_output.py
# 
# Created:  Oct 2020, M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import numpy as np
from SUAVE.Core            import Units  

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def print_engine_output(filename, tag,EPNL_total,PNLT_total,Velocity_aircraft, time,Altitude,Mach_aircraft,
                            Velocity_primary,Velocity_secondary,angles,phi, distance_microphone,PNLT_primary,PNLT_secondary,
                            PNLT_mixed,SPLt_dBA_max, EPNL_primary,EPNL_secondary,EPNL_mixed,SENEL_total, nsteps,
                            frequency, SPL_primary_history,SPL_secondary_history,SPL_mixed_history , SPL_total_history ): 
    
    """This method prints

    Assumptions: 

    Inputs: 


    Outputs:  

    """ 
    # unpack   
    if not filename:
        filename = ('SAE_Noise_' + str( tag) + '.dat')

    fid      = open(filename,'w')
     
    # print EPNL_total

    # Printing the output solution for the engine noise calculation

    fid.write('Engine noise module - SAE Model for Turbofan' + '\n')
    fid.write('Certification point = FLYOVER' + '\n')
    fid.write('EPNL = ' + str('%3.2f' % EPNL_total) + '\n')
    fid.write('PNLTM = ' + str('%3.2f' % np.max(PNLT_total)) + '\n')


    fid.write('Reference speed =  ')
    fid.write(str('%2.2f' % (Velocity_aircraft/Units.kts))+'  kts')
    fid.write('\n')
    fid.write('PNLT history')
    fid.write('\n')
    fid.write('time     	altitude     Mach     Core Velocity   Fan Velocity  Polar angle    Azim angle    distance    Primary	  Secondary 	 Mixed        Total')
    fid.write('\n')
    for id in range (0,nsteps):
        fid.write(str('%2.2f' % time[id])+'        ')
        fid.write(str('%2.2f' % Altitude[id])+'        ')
        fid.write(str('%2.2f' % Mach_aircraft[id])+'        ')
        fid.write(str('%3.3f' % Velocity_primary[id])+'        ')
        fid.write(str('%3.3f' % Velocity_secondary[id])+'        ')
        fid.write(str('%2.2f' % (angles[id]*180/np.pi))+'        ')
        fid.write(str('%2.2f' % (phi[id]*180/np.pi))+'        ')
        fid.write(str('%2.2f' % distance_microphone[id])+'        ')
        fid.write(str('%2.2f' % PNLT_primary[id])+'        ')
        fid.write(str('%2.2f' % PNLT_secondary[id])+'        ')
        fid.write(str('%2.2f' % PNLT_mixed[id])+'        ')
        fid.write(str('%2.2f' % PNLT_total[id])+'        ')
        fid.write(str('%2.2f' % SPLt_dBA_max[id])+'        ')
        fid.write('\n')
    fid.write('\n')
    fid.write('PNLT max =  ')
    fid.write(str('%2.2f' % (np.max(PNLT_total)))+'  dB')
    fid.write('\n')
    fid.write('dBA max =  ')
    fid.write(str('%2.2f' % (np.max(SPLt_dBA_max)))+'  dBA') 
    fid.write('\n')
    fid.write('EPNdB')
    fid.write('\n')
    fid.write('Primary    Secondary  	 Mixed       Total')
    fid.write('\n')
    fid.write(str('%2.2f' % EPNL_primary)+'        ')
    fid.write(str('%2.2f' % EPNL_secondary)+'        ')
    fid.write(str('%2.2f' % EPNL_mixed)+'        ')
    fid.write(str('%2.2f' % EPNL_total)+'        ')
    fid.write('\n')
    fid.write('\n')
    fid.write('SENEL = ')
    fid.write(str('%2.2f' % SENEL_total)+'        ')        

    for id in range (0,nsteps):
        fid.write('\n')
        fid.write('\n')
        fid.write('Emission angle = ' + str(angles[id]*180/np.pi) + '\n')
        fid.write('Altitude = ' + str(Altitude[id]) + '\n')
        fid.write('Distance = ' + str(distance_microphone[id]) + '\n')
        fid.write('Time = ' + str(time[id]) + '\n')
        fid.write('f		Primary  Secondary  	Mixed  		Total' + '\n')


        for ijd in range(0,24):
            fid.write(str((frequency[ijd])) + '       ')
            fid.write(str('%3.2f' % SPL_primary_history[id][ijd]) + '       ')
            fid.write(str('%3.2f' % SPL_secondary_history[id][ijd]) + '       ')
            fid.write(str('%3.2f' % SPL_mixed_history[id][ijd]) + '       ')
            fid.write(str('%3.2f' % SPL_total_history[id][ijd]) + '       ')
            fid.write('\n')

    fid.close 
    
    return  







