#-------------------------------------------------------------------------------
# Name:        Fink's model
# Purpose:
#
# Author:      CARIDSIL
#
# Created:     16/06/2015
# Copyright:   (c) CARIDSIL 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core            import Data
from SUAVE.Core            import Units


from noise_clean_wing import noise_clean_wing
from noise_landing_gear import noise_landing_gear
from noise_leading_edge_slat import noise_leading_edge_slat
from noise_trailing_edge_flap import noise_trailing_edge_flap

from SUAVE.Methods.Noise.Fidelity_One import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One import epnl_noise

# package imports
import numpy as np

def noise_fidelity_one(configs, analyses,trajectory):

    """ SUAVE.Methods.Noise.Fidelity_One.noise_fidelity_one(vehicle,airport):
            Computes the noise from different sources of the airframe for a given vehicle for a constant altitute flight.

            Inputs:
                vehicle	 - SUAVE type vehicle

                includes these fields:
                    S                          - Wing Area
                    bw                         - Wing Span
                    Sht                        - Horizontal tail area
                    bht                        - Horizontal tail span
                    Svt                        - Vertical tail area
                    bvt                        - Vertical tail span
                    deltaf                     - Flap deflection
                    Sf                         - Flap area
                    cf                         - Flap chord
                    slots                      - Number of slots (Flap type)
                    Dp                         - Main landing gear tyre diameter
                    Hp                         - Main lading gear strut length
                    Dn                         - Nose landing gear tyre diameter
                    Hn                         - Nose landing gear strut length
                    wheels                     - Number of wheels

                airport   - SUAVE type airport data, with followig fields:
                    atmosphere                  - Airport atmosphere (SUAVE type)
                    altitude                    - Airport altitude
                    delta_isa                   - ISA Temperature deviation


            Outputs: One Third Octave Band SPL [dB]
                SPL_wing                         - Sound Pressure Level of the clean wing
                SPLht                            - Sound Pressure Level of the horizontal tail
                SPLvt                            - Sound Pressure Level of the vertical tail
                SPL_flap                         - Sound Pressure Level of the flaps trailing edge
                SPL_slat                         - Sound Pressure Level of the slat leading edge
                SPL_main_landing_gear            - Sound Pressure Level og the main landing gear
                SPL_nose_landing_gear            - Sound Pressure Level of the nose landing gear

            Assumptions:
                Correlation based."""


    # ==============================================
        # Unpack
    # ==============================================

    Sw  =       configs.base.wings.main_wing.areas.reference                    #wing area, sq.ft
    bw  =       configs.base.wings.main_wing.spans.projected                    #wing span, ft
    Sht =       configs.base.wings.horizontal_stabilizer.areas.reference           #horizontal tail area, sq.ft
    bht =       configs.base.wings.horizontal_stabilizer.spans.projected            #horizontal tail span, ft
    Svt =       configs.base.wings.vertical_stabilizer.areas.reference         #vertical tail area, sq.ft
    bvt =       configs.base.wings.vertical_stabilizer.spans.projected            #vertical tail span, ft
    deltaf  =   configs.base.wings.main_wing.flaps.angle                     #flap delection, rad
    Sf  =       configs.base.wings.main_wing.flaps.area                     #flap area, sq.ft
    cf=         configs.base.wings.main_wing.flaps.chord                      #flap chord, ft
    slots=      configs.base.wings.main_wing.flaps.number_slots              #Number of slots (Flap type)
    Dp=         configs.base.landing_gear.main_tire_diameter                   #MLG tyre diameter, ft
    Hp=         configs.base.landing_gear.nose_tire_diameter             #MLG strut length, ft
    Dn      =   configs.base.landing_gear.main_strut_length               #NLG tyre diameter, ft
    Hn      =   configs.base.landing_gear.nose_strut_length              #NLG strut length, ft
    gear    =   configs.base.landing_gear.gear_condition                #Gear up ==o and gear down ==1
    wheels    =   configs.base.landing_gear.number_wheels               #Number of wheels   
    
    
    #modification 20/07/2015
    velocity=configs.flight.velocity
   # altitute=configs.flight.altitute
    altitute=trajectory[:][1]
    angle=trajectory[:][3]
    distance_vector=trajectory[:][2]
    time=trajectory[:][0]
    
    nsteps=len(time)
    
    sound_speed=np.zeros(nsteps)
    density=np.zeros(nsteps)
    viscosity=np.zeros(nsteps)
    temperature=np.zeros(nsteps)
    M=np.zeros(nsteps)
    deltaw=np.zeros(nsteps)
    
    # ==============================================
        # Computing atmospheric conditions
    # ==============================================
    
    for id in range (0,nsteps):
        atmo_data = analyses.atmosphere.compute_values(altitute[id])
    
        sound_speed[id] =    np.float(atmo_data.speed_of_sound)
        density[id] =       np.float(atmo_data.density)
        viscosity[id] =     np.float(atmo_data.dynamic_viscosity*10.7639) #units converstion - m2 to ft2
        temperature[id] =   np.float(atmo_data.temperature)
        
        #Mach number
        M[id]=velocity/np.sqrt(1.4*287*temperature[id])
    
        #Wing Turbulent Boundary Layer thickness, ft
        deltaw[id]=0.37*(Sw/bw)*((velocity/Units.ft)*Sw/(bw*viscosity[id]))**(-0.2)
    


    kt2fts=1.6878098571 #Units conversion - knots to ft/s

    #Generate array with the One Third Octave Band Center Frequencies
    frequency=np.array((50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, \
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000))


    phi=0*np.pi/180                #Azimuthal angle, rad

    velocity_fst = velocity * Units.knot



    # write header of file
    filename = 'Noise_Flight_Trajectory_Takeoff.dat'
    fid = open(filename,'w')   # Open output file


    nrange = len(angle) #number of positions of the aircraft to calculate the noise
    i=0
    SPL_wing_history = np.zeros((nrange,24))
    SPLht_history= np.zeros((nrange,24))
    SPLvt_history= np.zeros((nrange,24))
    SPL_flap_history= np.zeros((nrange,24))
    SPL_slat_history= np.zeros((nrange,24))
    SPL_main_landing_gear_history= np.zeros((nrange,24))
    SPL_nose_landing_gear_history= np.zeros((nrange,24))
    SPL_total_history = np.zeros((nrange,24))
    
    
    #START LOOP FOR EACH POSITION OF AIRCRAFT   
    for i in range(0,nrange):
        
   # for angle in range(30, 180, 30):
      # i=i+1
       theta=angle[i] #*np.pi/180

       distance=distance_vector[i] #/(np.sin(theta)*np.cos(phi))   #Distance from airplane to observer, evaluated at retarded time


       SPL_wing = noise_clean_wing(Sw,bw,0,1,deltaw[i],velocity,viscosity[i],M[i],phi,theta,distance,frequency)     #Wing Noise
       SPLht=noise_clean_wing(Sht,bht,0,1,deltaw[i],velocity,viscosity[i],M[i],phi,theta,distance,frequency)      #Horizontal Tail Noise
       SPLvt=noise_clean_wing(Svt,bvt,0,0,deltaw[i],velocity,viscosity[i],M[i],phi,theta,distance,frequency)      #Vertical Tail Noise

       SPL_slat=noise_leading_edge_slat(SPL_wing,Sw,bw,velocity,deltaw[i],viscosity[i],M[i],phi,theta,distance,frequency)         #Slat leading edge

       if (deltaf==0):
         SPL_flap=np.zeros(24)
       else:
            SPL_flap=noise_trailing_edge_flap(Sf,cf,deltaf,slots,velocity,M[i],phi,theta,distance,frequency) #Trailing Edge Flaps Noise

       if gear==0:
            SPL_main_landing_gear=np.zeros(24)
            SPL_nose_landing_gear=np.zeros(24)
       else:
            SPL_main_landing_gear=noise_landing_gear(Dp,Hp,wheels,M[i],velocity,phi,theta,distance,frequency)+3       #Main Landing Gear Noise
            SPL_nose_landing_gear=noise_landing_gear(Dn,Hn,wheels,M[i],velocity,phi,theta,distance,frequency)       #Nose Landing Gear Noise



        #Total Airframe Noise
       SPL_total=10.*np.log10(10.0**(0.1*SPL_wing)+10.0**(0.1*SPLht)+10**(0.1*SPL_flap)+ \
            10.0**(0.1*SPL_slat)+10.0**(0.1*SPL_main_landing_gear)+10.0**(0.1*SPL_nose_landing_gear))
            
       
       SPL_total_history[i][:]=SPL_total[:]
       SPL_wing_history[i][:]=SPL_wing[:]
       SPLvt_history[i][:]=SPLvt[:]
       SPLht_history[i][:]=SPLht[:]
       SPL_flap_history[i][:]=SPL_flap[:]
       SPL_slat_history[i][:]=SPL_slat[:]
       SPL_nose_landing_gear_history[i][:]=SPL_nose_landing_gear[:]
       SPL_main_landing_gear_history[i][:]=SPL_main_landing_gear[:]
       
       


   
   #Calculation of the Perceived Noise Level EPNL based on the sound time history
    PNL_total               =  pnl_noise.pnl_noise(SPL_total_history)
    PNL_wing                =   pnl_noise.pnl_noise(SPL_wing_history)
    PNL_ht                  = pnl_noise.pnl_noise(SPLht_history)
    PNL_vt                  = pnl_noise.pnl_noise(SPLvt_history)
    PNL_nose_landing_gear   =  pnl_noise.pnl_noise(SPL_nose_landing_gear_history)
    PNL_main_landing_gear   =  pnl_noise.pnl_noise(SPL_main_landing_gear_history)
    PNL_slat                =   pnl_noise.pnl_noise(SPL_slat_history)
    PNL_flap                =   pnl_noise.pnl_noise(SPL_flap_history)
    
    
   #Calculation of the tones corrections on the SPL for each component and total
    tone_correction_total = noise_tone_correction.noise_tone_correction(SPL_total_history) 
    tone_correction_wing = noise_tone_correction.noise_tone_correction(SPL_wing_history)
    tone_correction_ht = noise_tone_correction.noise_tone_correction(SPLht_history)
    tone_correction_vt = noise_tone_correction.noise_tone_correction(SPLvt_history)
    tone_correction_flap = noise_tone_correction.noise_tone_correction(SPL_flap_history)
    tone_correction_slat = noise_tone_correction.noise_tone_correction(SPL_slat_history)
    tone_correction_nose_landing_gear = noise_tone_correction.noise_tone_correction(SPL_nose_landing_gear_history)
    tone_correction_main_landing_gear = noise_tone_correction.noise_tone_correction(SPL_main_landing_gear_history)
    
    #Calculation of the PLNT for each component and total
    PNLT_total=PNL_total+tone_correction_total
    PNLT_wing=PNL_wing+tone_correction_wing
    PNLT_ht=PNL_ht+tone_correction_ht
    PNLT_vt=PNL_vt+tone_correction_vt
    PNLT_nose_landing_gear=PNL_nose_landing_gear+tone_correction_nose_landing_gear
    PNLT_main_landing_gear=PNL_main_landing_gear+tone_correction_main_landing_gear
    PNLT_slat=PNL_slat+tone_correction_slat
    PNLT_flap=PNL_flap+tone_correction_flap
    
    #Calculation of the EPNL for each component and total
    EPNL_total=epnl_noise.epnl_noise(PNLT_total)
    EPNL_wing=epnl_noise.epnl_noise(PNLT_wing)
    EPNL_ht=epnl_noise.epnl_noise(PNLT_ht)
    EPNL_vt=epnl_noise.epnl_noise(PNLT_vt)    
    EPNL_nose_landing_gear=epnl_noise.epnl_noise(PNLT_nose_landing_gear)
    EPNL_main_landing_gear=epnl_noise.epnl_noise(PNLT_main_landing_gear)
    EPNL_slat=epnl_noise.epnl_noise(PNLT_slat)
    EPNL_flap=epnl_noise.epnl_noise(PNLT_flap)
    
    
    
    fid.write('PNLT history')
    fid.write('\n')
    fid.write('time     	altitude       Mach    angle   distance        wing    		ht 		 vt  			flap    	slat    	nose    	main    	total')
    fid.write('\n')
    for id in range (0,nsteps):
        fid.write(str('%2.2f' % time[id])+'        ')
        fid.write(str('%2.2f' % altitute[id])+'        ')
        fid.write(str('%2.2f' % M[id])+'        ')
        fid.write(str('%2.2f' % (angle[id]*180/np.pi))+'        ')
        fid.write(str('%2.2f' % distance_vector[id])+'        ')
        fid.write(str('%2.2f' % PNLT_wing[id])+'        ')
        fid.write(str('%2.2f' % PNLT_ht[id])+'        ')
        fid.write(str('%2.2f' % PNLT_vt[id])+'        ')
        fid.write(str('%2.2f' % PNLT_flap[id])+'        ')
        fid.write(str('%2.2f' % PNLT_slat[id])+'        ')
        fid.write(str('%2.2f' % PNLT_nose_landing_gear[id])+'        ')
        fid.write(str('%2.2f' % PNLT_main_landing_gear[id])+'        ')
        fid.write(str('%2.2f' % PNLT_total[id])+'        ')
        fid.write('\n')
    fid.write('\n')
    fid.write('EPNdB')
    fid.write('\n')
    fid.write('wing    	ht  		vt  		flap    		slat    	nose    	main    	total')
    fid.write('\n')
    fid.write(str('%2.2f' % EPNL_wing)+'        ')
    fid.write(str('%2.2f' % EPNL_ht)+'        ')
    fid.write(str('%2.2f' % EPNL_vt)+'        ')
    fid.write(str('%2.2f' % EPNL_flap)+'        ')
    fid.write(str('%2.2f' % EPNL_slat)+'        ')
    fid.write(str('%2.2f' % EPNL_nose_landing_gear)+'        ')
    fid.write(str('%2.2f' % EPNL_main_landing_gear)+'        ')
    fid.write(str('%2.2f' % EPNL_total)+'        ')
    fid.close 
    
    
    
##       fid.write('\n')
##       fid.write('Polar angle = ' + str(angle) + '\n')
##       fid.write('f		wing    		ht  		vt  	flap    	main LDG    nose LDG    slat    	total' + '\n')
##
##       for id in range(0,24):
##           fid.write(str((frequency[id])) + '       ')
##           fid.write(str('%2.2f' % SPL_wing[id]) + '       ')
##           fid.write(str('%2.2f' % SPLht[id]) + '       ')
##           fid.write(str('%2.2f' % SPLvt[id]) + '       ')
##           fid.write(str('%2.2f' % SPL_flap[id]) + '       ')
##           fid.write(str('%2.2f' % SPL_main_landing_gear[id]) + '       ')
##           fid.write(str('%2.2f' % SPL_nose_landing_gear[id]) + '       ')
##           fid.write(str('%2.2f' % SPL_slat[id]) + '       ')
##           fid.write(str('%2.2f' % SPL_total[id]) + '       ')
##           fid.write('\n')
##
## fid.close
    
    return (EPNL_total)