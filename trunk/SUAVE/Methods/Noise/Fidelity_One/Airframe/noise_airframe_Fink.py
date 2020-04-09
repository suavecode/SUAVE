## @ingroupMethods-Noise-Fidelity_One-Airframe
# noise_airframce_Fink.py
# 
# Created:  Jun 2015, Carlos Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
from SUAVE.Core            import Data
from SUAVE.Core            import Units

from .noise_clean_wing import noise_clean_wing
from .noise_landing_gear import noise_landing_gear
from .noise_leading_edge_slat import noise_leading_edge_slat
from .noise_trailing_edge_flap import noise_trailing_edge_flap

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import epnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import atmospheric_attenuation
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import dbA_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_geometric

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_counterplot
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import senel_noise

import numpy as np

# ----------------------------------------------------------------------
#  Noise Airframce Fink
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Airframe
def noise_airframe_Fink(config, analyses, noise_segment,ioprint = 0, filename=0): 

    """ SUAVE.Methods.Noise.Fidelity_One.noise_fidelity_one(config, analyses, noise_segment):
            Computes the noise from different sources of the airframe for a given vehicle for a constant altitude flight.

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
                    
                noise segment - flight path data, containing:
                    distance_vector             - distance from the source location to observer
                    angle                       - polar angle from the source to the observer
                    phi                         - azimuthal angle from the source to the observer


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
    wing     = config.wings
    flap     = wing.main_wing.control_surfaces.flap
    
    Sw       = wing.main_wing.areas.reference  / (Units.ft)**2              #wing area, sq.ft
    bw       = wing.main_wing.spans.projected / Units.ft                    #wing span, ft
    Sht      = wing.horizontal_stabilizer.areas.reference / (Units.ft)**2   #horizontal tail area, sq.ft
    bht      = wing.horizontal_stabilizer.spans.projected / Units.ft        #horizontal tail span, ft
    Svt      = wing.vertical_stabilizer.areas.reference / (Units.ft)**2     #vertical tail area, sq.ft
    bvt      = wing.vertical_stabilizer.spans.projected  / Units.ft         #vertical tail span, ft
    deltaf   = flap.deflection                                              #flap delection, rad
    Sf       = flap.area  / (Units.ft)**2                                   #flap area, sq.ft        
    cf       = flap.chord_dimensional  / Units.ft                           #flap chord, ft
    Dp       = config.landing_gear.main_tire_diameter  / Units.ft           #MLG tyre diameter, ft
    Hp       = config.landing_gear.nose_tire_diameter  / Units.ft           #MLG strut length, ft
    Dn       = config.landing_gear.main_strut_length   / Units.ft           #NLG tyre diameter, ft
    Hn       = config.landing_gear.nose_strut_length   / Units.ft           #NLG strut length, ft
    gear     = config.landing_gear.gear_condition                           #Gear up or gear down
    
    nose_wheels    =   config.landing_gear.nose_wheels                           #Number of wheels   
    main_wheels    =   config.landing_gear.main_wheels                           #Number of wheels   
    main_units     =   config.landing_gear.main_units                            #Number of main units   
    velocity       =   np.float(noise_segment.conditions.freestream.velocity[0,0]) #aircraft velocity 
    altitude       =   noise_segment.conditions.freestream.altitude[:,0]           #aircraft altitude
    time           =   noise_segment.conditions.frames.inertial.time[:,0]          #time discretization

    noise_time = np.arange(0.,time[-1],.5)  
    altitude = np.interp(noise_time,time,altitude)

    # determining flap slot number
    if wing.main_wing.control_surfaces.flap.configuration_type   == 'single_slotted':
        slots = 1
    elif wing.main_wing.control_surfaces.flap.configuration_type == 'double_slotted':
        slots = 2
    elif wing.main_wing.control_surfaces.flap.configuration_type == 'triple_slotted':
        slots = 3  

    # Geometric information from the source to observer position
    distance_vector = noise_segment.dist    
    angle = noise_segment.theta 
    phi   = noise_segment.phi
    
    distance_vector = np.interp(noise_time,time,distance_vector)
    angle = np.interp(noise_time,time,angle)
    phi   = np.interp(noise_time,time,phi)
        
    # Number of points on the discretize segment   
    nsteps=len(noise_time)
    
    # Preparing matrix for noise calculation
    sound_speed = np.zeros(nsteps)
    density     = np.zeros(nsteps)
    viscosity   = np.zeros(nsteps)
    temperature = np.zeros(nsteps)
    M           = np.zeros(nsteps)
    deltaw      = np.zeros(nsteps)
    
    # ==============================================
    #         Computing atmospheric conditions
    # ==============================================
    
    for id in range (0,nsteps):
        atmo_data = analyses.atmosphere.compute_values(altitude[id])
        
        #unpack    
        sound_speed[id] =    np.float(atmo_data.speed_of_sound)
        density[id]     =    np.float(atmo_data.density)
        viscosity[id]   =    np.float(atmo_data.dynamic_viscosity*10.7639) #units converstion - m2 to ft2
        temperature[id] =    np.float(atmo_data.temperature)
        
        #Mach number
        M[id] = velocity/np.sqrt(1.4*287*temperature[id])
    
        #Wing Turbulent Boundary Layer thickness, ft
        deltaw[id] = 0.37*(Sw/bw)*((velocity/Units.ft)*Sw/(bw*viscosity[id]))**(-0.2)
    

    #Units conversion - knots to ft/s
    kt2fts = 1.6878098571 

    #Generate array with the One Third Octave Band Center Frequencies
    frequency = np.array((50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, \
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000))


    # Velocity in fts
    velocity_fst = velocity * Units.knot
    
    #number of positions of the aircraft to calculate the noise
    nrange = len(angle) 
    i=0
    SPL_wing_history = np.zeros((nrange,24))
    SPLht_history    = np.zeros((nrange,24))
    SPLvt_history    = np.zeros((nrange,24))
    SPL_flap_history = np.zeros((nrange,24))
    SPL_slat_history = np.zeros((nrange,24))
    SPL_main_landing_gear_history = np.zeros((nrange,24))
    SPL_nose_landing_gear_history = np.zeros((nrange,24))
    SPL_total_history = np.zeros((nrange,24))
    
    #Noise history in dBA
    SPLt_dBA_history = np.zeros((nrange,24))  
    SPLt_dBA_max = np.zeros(nrange)    
    
    #START LOOP FOR EACH POSITION OF AIRCRAFT   
    for i in range(0,nrange-1):
        #Emission angle theta   
        theta = angle[i] 
        #Distance from airplane to observer, evaluated at retarded time
        distance = distance_vector[i]    
       
         #Atmospheric attenuation
        delta_atmo=atmospheric_attenuation(distance)

        #Call each noise source model
        SPL_wing = noise_clean_wing(Sw,bw,0,1,deltaw[i],velocity,viscosity[i],M[i],phi[i],theta,distance,frequency) - delta_atmo    #Wing Noise
        SPLht    = noise_clean_wing(Sht,bht,0,1,deltaw[i],velocity,viscosity[i],M[i],phi[i],theta,distance,frequency)  -delta_atmo    #Horizontal Tail Noise
        SPLvt    = noise_clean_wing(Svt,bvt,0,0,deltaw[i],velocity,viscosity[i],M[i],phi[i],theta,distance,frequency)  -delta_atmo    #Vertical Tail Noise
 
        SPL_slat = noise_leading_edge_slat(SPL_wing,Sw,bw,velocity,deltaw[i],viscosity[i],M[i],phi[i],theta,distance,frequency) -delta_atmo        #Slat leading edge
 
        if (deltaf==0):
            SPL_flap = np.zeros(24)
        else:
            SPL_flap = noise_trailing_edge_flap(Sf,cf,deltaf,slots,velocity,M[i],phi[i],theta,distance,frequency) - delta_atmo #Trailing Edge Flaps Noise
 
        if gear=='up': #0
            SPL_main_landing_gear = np.zeros(24)
            SPL_nose_landing_gear = np.zeros(24)
        else:
            SPL_main_landing_gear = noise_landing_gear(Dp,Hp,main_wheels,M[i],velocity,phi[i],theta,distance,frequency)  - delta_atmo     #Main Landing Gear Noise
            SPL_nose_landing_gear = noise_landing_gear(Dn,Hn,nose_wheels,M[i],velocity,phi[i],theta,distance,frequency)  - delta_atmo     #Nose Landing Gear Noise
        if main_units>1: #Incoherent summation of each main landing gear unit
            SPL_main_landing_gear = SPL_main_landing_gear+3*(main_units-1)
 
 
         #Total Airframe Noise
        SPL_total = 10.*np.log10(10.0**(0.1*SPL_wing)+10.0**(0.1*SPLht)+10**(0.1*SPL_flap)+ \
             10.0**(0.1*SPL_slat)+10.0**(0.1*SPL_main_landing_gear)+10.0**(0.1*SPL_nose_landing_gear))
            
        SPL_total_history[i][:] = SPL_total[:]
        SPL_wing_history[i][:]  = SPL_wing[:]
        SPLvt_history[i][:]     = SPLvt[:]
        SPLht_history[i][:]     = SPLht[:]
        SPL_flap_history[i][:]  = SPL_flap[:]
        SPL_slat_history[i][:]  = SPL_slat[:]
        SPL_nose_landing_gear_history[i][:] = SPL_nose_landing_gear[:]
        SPL_main_landing_gear_history[i][:] = SPL_main_landing_gear[:]
        
        
        #Calculation of dBA based on the sound pressure time history
        SPLt_dBA = dbA_noise(SPL_total)
        SPLt_dBA_history[i][:] = SPLt_dBA[:]
        SPLt_dBA_max[i] = max(SPLt_dBA)        
       
       
   #Calculation of dBA based on the sound pressure time history
    dbA_total               =       np.max(SPLt_dBA_history)    #(Not used to certification point)
          
   #Calculation of the Perceived Noise Level EPNL based on the sound time history
    PNL_total               =       pnl_noise(SPL_total_history)
    PNL_wing                =       pnl_noise(SPL_wing_history)
    PNL_ht                  =       pnl_noise(SPLht_history)
    PNL_vt                  =       pnl_noise(SPLvt_history)
    PNL_nose_landing_gear   =       pnl_noise(SPL_nose_landing_gear_history)
    PNL_main_landing_gear   =       pnl_noise(SPL_main_landing_gear_history)
    PNL_slat                =       pnl_noise(SPL_slat_history)
    PNL_flap                =       pnl_noise(SPL_flap_history)
    
    
   #Calculation of the tones corrections on the SPL for each component and total
    tone_correction_total = noise_tone_correction(SPL_total_history) 
    tone_correction_wing  = noise_tone_correction(SPL_wing_history)
    tone_correction_ht    = noise_tone_correction(SPLht_history)
    tone_correction_vt    = noise_tone_correction(SPLvt_history)
    tone_correction_flap  = noise_tone_correction(SPL_flap_history)
    tone_correction_slat  = noise_tone_correction(SPL_slat_history)
    tone_correction_nose_landing_gear = noise_tone_correction(SPL_nose_landing_gear_history)
    tone_correction_main_landing_gear = noise_tone_correction(SPL_main_landing_gear_history)
    
    #Calculation of the PLNT for each component and total
    PNLT_total = PNL_total+tone_correction_total
    PNLT_wing  = PNL_wing+tone_correction_wing
    PNLT_ht    = PNL_ht+tone_correction_ht
    PNLT_vt    = PNL_vt+tone_correction_vt
    PNLT_nose_landing_gear = PNL_nose_landing_gear+tone_correction_nose_landing_gear
    PNLT_main_landing_gear = PNL_main_landing_gear+tone_correction_main_landing_gear
    PNLT_slat = PNL_slat+tone_correction_slat
    PNLT_flap = PNL_flap+tone_correction_flap
    
    #Calculation of the EPNL for each component and total
    EPNL_total = epnl_noise(PNLT_total)
    EPNL_wing  = epnl_noise(PNLT_wing)
    EPNL_ht    = epnl_noise(PNLT_ht)
    EPNL_vt    = epnl_noise(PNLT_vt)    
    EPNL_nose_landing_gear = epnl_noise(PNLT_nose_landing_gear)
    EPNL_main_landing_gear = epnl_noise(PNLT_main_landing_gear)
    EPNL_slat = epnl_noise(PNLT_slat)
    EPNL_flap = epnl_noise(PNLT_flap)
    
    #Calculation of the SENEL total
    SENEL_total = senel_noise(SPLt_dBA_max)
    
    if ioprint:
        # write header of file
        if not filename:            
            filename = ('Noise_' + str(config.tag) + '.dat')
            
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
        
        
    
        filename1 = ('History_Noise_' + str(config.tag) + '.dat')
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
    
    return (EPNL_total,SPL_total_history,SENEL_total)
