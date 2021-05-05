## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# print_airframe_output.py
# 
# Created:  Oct 2020, M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import numpy as np
from SUAVE.Core            import Units  

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def print_airframe_output(SAE_Airframe_Noise_Outputs): 
    """This prints the airframe noise of a turbofan aircraft

    Assumptions:
       N/A

    Inputs: 
       SAE_Airframe_Noise_Outputs  - Airframe Noise Data Structure  

    Outputs:  
        N/A
        
    Properties Used:
        None 
    """  
    tag                      = SAE_Airframe_Noise_Outputs.tag
    filename                 = SAE_Airframe_Noise_Outputs.filename
    velocity                 = SAE_Airframe_Noise_Outputs.velocity
    nsteps                   = SAE_Airframe_Noise_Outputs.nsteps
    time                     = SAE_Airframe_Noise_Outputs.time
    altitude                 = SAE_Airframe_Noise_Outputs.altitude
    M                        = SAE_Airframe_Noise_Outputs.M
    angle                    = SAE_Airframe_Noise_Outputs.angle
    distance_vector          = SAE_Airframe_Noise_Outputs.distance_vector
    PNLT_wing                = SAE_Airframe_Noise_Outputs.PNLT_wing
    phi                      = SAE_Airframe_Noise_Outputs.phi
    PNLT_ht                  = SAE_Airframe_Noise_Outputs.PNLT_ht
    PNLT_vt                  = SAE_Airframe_Noise_Outputs.PNLT_vt
    PNLT_flap                = SAE_Airframe_Noise_Outputs.PNLT_flap
    PNLT_slat                = SAE_Airframe_Noise_Outputs.PNLT_slat
    PNLT_nose_landing_gear   = SAE_Airframe_Noise_Outputs.PNLT_nose_landing_gear
    PNLT_main_landing_gear   = SAE_Airframe_Noise_Outputs.PNLT_main_landing_gear
    PNLT_total               = SAE_Airframe_Noise_Outputs.PNLT_total
    SPLt_dBA_max             = SAE_Airframe_Noise_Outputs.SPLt_dBA_max
    nrange                   = SAE_Airframe_Noise_Outputs.nrange
    frequency                = SAE_Airframe_Noise_Outputs.frequency
    EPNL_wing                = SAE_Airframe_Noise_Outputs.EPNL_wing
    EPNL_ht                  = SAE_Airframe_Noise_Outputs.EPNL_ht
    EPNL_vt                  = SAE_Airframe_Noise_Outputs.EPNL_vt
    EPNL_flap                = SAE_Airframe_Noise_Outputs.EPNL_flap
    EPNL_slat                = SAE_Airframe_Noise_Outputs.EPNL_slat
    EPNL_nose_landing_gear   = SAE_Airframe_Noise_Outputs.EPNL_nose_landing_gear
    EPNL_main_landing_gear   = SAE_Airframe_Noise_Outputs.EPNL_main_landing_gear
    EPNL_total               = SAE_Airframe_Noise_Outputs.EPNL_total
    SENEL_total              = SAE_Airframe_Noise_Outputs.SENEL_total
    SPL_total_history        = SAE_Airframe_Noise_Outputs.SPL_total_history
    SPLt_dBA_history         = SAE_Airframe_Noise_Outputs.SPLt_dBA_history  
    
    # write header of file
    if not filename:            
        filename = ('Noise_' + str(tag) + '.dat')
        
    fid = open(filename,'w')   # Open output file    
    
    fid.write('Reference speed =  ')
    fid.write(str('%2.2f' % (velocity/Units.kts))+'  kts')
    fid.write('\n')
    fid.write('PNLT history')
    fid.write('\n')
    fid.write('time       altitude      Mach    Polar_angle    Azim_angle   distance        wing  	   ht 	        vt 	   flap   	 slat         nose        main         total         dBA')
    fid.write('\n')
    
    for id in range (0,nsteps):
        fid.write(str('%2.2f' % time[id])+'        ')
        fid.write(str('%2.2f' % altitude[id])+'        ')
        fid.write(str('%2.2f' % M[id])+'        ')
        fid.write(str('%2.2f' % (angle[id]*180/np.pi))+'        ')
        fid.write(str('%2.2f' % (phi[id]*180/np.pi))+'        ')
        fid.write(str('%2.2f' % distance_vector[id])+'        ')
        fid.write(str('%2.2f' % PNLT_wing[id])+'        ')
        fid.write(str('%2.2f' % PNLT_ht[id])+'        ')
        fid.write(str('%2.2f' % PNLT_vt[id])+'        ')
        fid.write(str('%2.2f' % PNLT_flap[id])+'        ')
        fid.write(str('%2.2f' % PNLT_slat[id])+'        ')
        fid.write(str('%2.2f' % PNLT_nose_landing_gear[id])+'        ')
        fid.write(str('%2.2f' % PNLT_main_landing_gear[id])+'        ')
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
    fid.write('\n')
    fid.write('EPNdB')
    fid.write('\n')
    fid.write('wing	       ht          vt         flap         slat    	nose        main	total')
    fid.write('\n')
    fid.write(str('%2.2f' % EPNL_wing)+'        ')
    fid.write(str('%2.2f' % EPNL_ht)+'        ')
    fid.write(str('%2.2f' % EPNL_vt)+'        ')
    fid.write(str('%2.2f' % EPNL_flap)+'        ')
    fid.write(str('%2.2f' % EPNL_slat)+'        ')
    fid.write(str('%2.2f' % EPNL_nose_landing_gear)+'        ')
    fid.write(str('%2.2f' % EPNL_main_landing_gear)+'        ')
    fid.write(str('%2.2f' % EPNL_total)+'        ')
    fid.write('\n')
    fid.write('SENEL = ')
    fid.write(str('%2.2f' % SENEL_total)+'        ')       
    fid.close 
    
    

    filename1 = ('History_Noise_' + str(tag) + '.dat')
    fid = open(filename1,'w')   # Open output file
    fid.write('Reference speed =  ')
    fid.write(str('%2.2f' % (velocity/Units.kts))+'  kts')
    fid.write('\n')
    fid.write('Sound Pressure Level for the Total Aircraft Noise')
    fid.write('\n')
    
    for nid in range (0,nrange):
        fid.write('Polar angle = ' + str('%2.2f' % (angle[nid]*(180/np.pi))) + '  degrees' + '\n')
        fid.write('f		total SPL(dB)    total SPL(dBA)' + '\n')
        for id in range(0,24):
            fid.write(str((frequency[id])) + '           ')
            fid.write(str('%3.2f' % SPL_total_history[nid][id]) + '          ')
            fid.write(str('%3.2f' % SPLt_dBA_history[nid][id]))
            fid.write('\n')
        fid.write('SPLmax (dB) =  ')
        fid.write(str('%3.2f' % (np.max(SPL_total_history[nid][:])))+'  dB' + '\n')
        fid.write('SPLmax (dBA) =  ')
        fid.write(str('%3.2f' % (np.max(SPLt_dBA_history[nid][:])))+'  dB')
        fid.write('\n')

    fid.close    
    return