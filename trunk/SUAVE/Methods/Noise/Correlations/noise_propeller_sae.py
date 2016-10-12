# noise_propeller_sae.py
#
# Created:  Oct 2016, C. Ilario
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np

def noise_propeller_sae(noise_data, ioprint = 0):
    """ SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.noise_propeller_sae(noise_data):
                Computes the Far-field noise for propeller noise following SAE AIR1407 procedure.

                Inputs:
                    noise_data	 - SUAVE type vehicle

                Outputs:
                    OASPL                         - Overall Sound Pressure Level, [dB]
                    PNL                           - Perceived Noise Level, [dB]
                    PNL_dBA                       - Perceived Noise Level A-weighted level, [dBA]
                    EPNdB_takeoff                 - Takeoff Effective Perceived Noise Level, [EPNdB]
                    EPNdB_landing                 - Landing Effective Perceived Noise Level, [EPNdB]

                Assumptions:
                    Empirical based procedure."""    

  
  # unpack
    diameter = noise_data.diameter / Units.ft
    n_blades = noise_data.n_blades
    n_propellers = noise_data.n_engines
    HP       = noise_data.HP / Units.horsepower
    RPM      = noise_data.rpm
    speed    = noise_data.speed / Units.fts
    sound_speed = noise_data.sound_speed / Units.fts
    dist    = noise_data.distance
    theta   = noise_data.angle
    
# ***********************************************
    #Correction for the number of propellers:
    if n_propellers == 1:
        NC = 0
    elif n_propellers == 2:
        NC = 3
    elif n_propellers == 4:
        NC = 6
     
# Number of points on the discretize segment   
    nsteps=len(dist) 
# Preparing matrix    
    tip_mach = np.zeros(nsteps)
    tip_speed = np.zeros(nsteps)
    Vtip = np.zeros(nsteps)
    Vtip_Mach = np.zeros(nsteps)    
    FL1 = np.zeros(nsteps)
    FL3_1 = np.zeros(nsteps)
    FL3_2 = np.zeros(nsteps)
    DI = np.zeros(nsteps)
    PNL_factor = np.zeros(nsteps)
    PNL = np.zeros(nsteps)
    PNL_dBA =np.zeros(nsteps)
    
#***************************************************************    
# Farfield Partial Noise Level Based on Blade count and Propeller diameter
# ***** Figure 4 *********************

    if n_blades == 2:
        FL2 = -8.683066391*np.log(diameter)+26.6099964054
    elif n_blades == 3:
        FL2 = -8.7158230649*np.log(diameter)+23.7536138121
    elif n_blades == 4:
        FL2 = -8.7339566501*np.log(diameter)+20.6709100907
    elif n_blades == 6:
        FL2 = -8.6437428565*np.log(diameter)+17.6059581647
    elif n_blades == 8:
        FL2 = -8.7158865083*np.log(diameter)+15.2385549706

# ***************************************************************
#                    START - MAIN LOOP
#***************************************************************
  
    for id in range(0,nsteps):
        
  #Calculate propeller tip Mach number:
        tip_speed[id]  = 3.141590*diameter*RPM[id]/60.0
        tip_mach[id]  = tip_speed[id]/sound_speed[id]  
    
    #Calculation of the helical tip speed
        Vtip[id] = np.sqrt(speed[id]**2 + tip_speed[id]**2)
        Vtip_Mach[id] = Vtip[id]/sound_speed[id]
    
# **********************************    
    # Farfield Partial Level based on Power and Tip Speed    
# ***** Figure 3 *********************
        
        if tip_mach[id] <= 0.4:
            FL1[id] = 6.7810306101*np.log(HP[id])+32.4536847808
        elif tip_mach[id] <= 0.5:
            FL1[id] = 6.7656398511*np.log(HP[id])+36.4477233368
        elif tip_mach[id] <= 0.6:
            FL1[id] = 6.6832881888*np.log(HP[id])+40.5886386121
        elif tip_mach[id] <= 0.7:
            FL1[id] = 6.7439113333*np.log(HP[id])+43.5568035724
        elif tip_mach[id] <= 0.8:
            FL1[id] = 6.8401765646*np.log(HP[id])+46.8530951754
        elif tip_mach[id] <= 0.9:
            FL1[id] = 6.8447630205*np.log(HP[id])+50.7481250789
        
# ********************************** 
    #Atmospheric Absorption and Spherical Spreading
# ***** Figure 5 *********************

        #Perceived Noise Level and dbA
        FL3_1[id] = -10.6203648721*np.log(dist[id]) + 63.9095683153
        
        #Overall sound pressure level
        FL3_2[id] = -9.0388679389*np.log(dist[id]) + 55.9177440082
        
# ********************************** 
    # Directivity Index
# ***** Figure 6 *********************

        DI[id] = 0.000000003*(theta[id]**5) - 1.5330661136843e-6*(theta[id]**4) + 0.0002748446*(theta[id]**3) - 0.0224465579*(theta[id]**2) + 0.8841215644*theta[id] -15.0851264829
    
# **********************************
    # PNL adjustment for 2 bladed propellers
# ***** Figure 7 *********************

        if n_blades == 2:

            if Vtip_Mach[id] <= 0.5:
                PNL_factor[id] = -0.0002194325*(diameter)**4 + 0.0119121744*diameter**3 - 0.1715062582*diameter**2 - 0.440364203*diameter + 6.2098100491
            
            elif Vtip_Mach[id] <= 0.6:   
                PNL_factor[id] = -0.000244323*(diameter)**4 + 0.0152241341*diameter**3 - 0.2925615622*diameter**2 + 1.1285073819*diameter + 1.4783088763
            
            elif Vtip_Mach[id] <= 0.7:   
                PNL_factor[id] = -9.19201576386925e-5*(diameter)**4 + 0.007125071*diameter**3 - 0.1605227186*diameter**2 +0.5589157992*diameter + 2.4062117898
            
            elif Vtip_Mach[id] <= 0.8:
                PNL_factor[id] = -9.71380207504857e-6*(diameter)**4 + 0.0013799409*diameter**3 - 0.0240602419*diameter**2 -0.6106811004*diameter + 6.8595599903
            
            elif Vtip_Mach[id] <= 0.85:
                PNL_factor[id] = 0.0005259053*(diameter)**3 - 0.0058445988*diameter**2 -0.6296392506*diameter + 7.3325512636
            
            elif Vtip_Mach[id] <= 0.9:
                PNL_factor[id] = 0.0001869235*(diameter)**4 - 0.0108618115*diameter**3 + 0.2292260658*diameter**2 - 2.468037691*diameter + 13.2430172278
        else:
            print 'ERROR: Method limited for 2 bladed propellers right now!!'
            return
        
# ****************** CALCULATION OF NOISE LEVELS *********************        
        OASPL = FL1[id]+FL2+FL3_2[id]+DI[id]+NC
        PNL[id] = OASPL + PNL_factor[id]
        PNL_dBA[id] = PNL[id] - 14
        
# *********************** END OF LOOP  *******************************    
        
    # Effective Perceived Noise Level for takeoff and landing:
    EPNdB_takeoff = np.max(PNL) - 4
    EPNdB_landing = np.max(PNL) - 2
    
# ********************************************************    
    # Write output file
# ********************************************************   

    if ioprint:

        fid = open('prop_test.dat','w')   # Open output file    
        
        fid.write('Reference speed =  ')
        fid.write(str('%2.2f' % (speed[-1]/Units.kts))+'  kts')
        fid.write('\n')
        fid.write('PNLT history')
        fid.write('\n')
        fid.write('time       altitude      Mach    RPM     Polar_angle    distance        PNL  	   dBA')
        fid.write('\n')
        
        for id in range (0,nsteps):
            fid.write(str('%2.2f' % noise_data.time[id])+'        ')
            fid.write(str('%2.2f' % noise_data.altitude[id])+'        ')
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
    
    
    return (np.max(PNL_dBA), EPNdB_takeoff, EPNdB_landing)


