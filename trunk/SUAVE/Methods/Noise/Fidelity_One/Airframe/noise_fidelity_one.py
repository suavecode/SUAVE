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


# package imports
import numpy as np

def noise_fidelity_one(configs, analyses):

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
    gear    =   configs.base.landing_gear.number_wheels                  #Gear up ==o and gear down ==1

    wheels = 2


    kt2fts=1.6878098571 #Units conversion - knots to ft/s
    #kt2ms=0.5144444 #units conversion - knots to m/s
   # ft2m2=10.7621

    #Generate array with the One Third Octave Band Center Frequencies
    frequency=np.array((50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, \
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000))

    #Input data for Fink's noise model
    velocity=198.0                    #Aircraft airspeed, kts
    altitute=500                      #Altitude, ft
    temperature=288.15               #Ambient temperature, K
    phi=0*np.pi/180                #Azimuthal angle, rad

    velocity_mps = velocity * Units.knot


    #Atmospheric conditions
    sound_speed=np.math.sqrt(1.4*287*temperature)   #sound speed, m/s 1116.44 #
    density=(1/(temperature/288.15))*1.225          #density, kg/m3 0.002377 #
    viscositySI=1.458E-6*temperature**1.5/((110.4+temperature)*density) #Kinematic viscosity, m2/s
    viscosity= viscositySI*10.7621 #units converstion - ft2 to m2

    #Mach number
    M=velocity*0.5144/np.sqrt(1.4*287*temperature)

    # ==============================================
        # Computing atmospheric conditions
    # ==============================================
  #  atmo_data = analyses.configs.base.atmosphere.compute_values(altitute)
  #  print atmo_data

  #  sound_speed =    atmo_data.speed_of_sound
  #  density =       atmo_data.density
  #  viscosity =     atmo_data.dynamic_viscosity*10.7621 #units converstion - ft2 to m2




    #Wing Turbulent Boundary Layer thickness, ft
    deltaw=0.37*(Sw/bw)*(velocity*kt2fts*Sw/(bw*viscosity))**(-0.2)


    i=0

    # write header of file
    filename = 'noise_test.dat'
    fid = open(filename,'w')   # Open output file


    for angle in range(30, 180, 30):
       i=i+1
       theta=angle*np.pi/180

       distance=altitute/(np.sin(theta)*np.cos(phi))   #Distance from airplane to observer, evaluated at retarded time


       SPL_wing = noise_clean_wing(Sw,bw,0,1,deltaw,velocity,viscosity,M,phi,theta,distance,frequency)     #Wing Noise
       SPLht=noise_clean_wing(Sht,bht,0,1,deltaw,velocity,viscosity,M,phi,theta,distance,frequency)      #Horizontal Tail Noise
       SPLvt=noise_clean_wing(Svt,bvt,0,0,deltaw,velocity,viscosity,M,phi,theta,distance,frequency)      #Vertical Tail Noise

       SPL_slat=noise_leading_edge_slat(SPL_wing,Sw,bw,velocity,deltaw,viscosity,M,phi,theta,distance,frequency)         #Slat leading edge

       if (deltaf==0):
         SPL_flap=np.zeros(24)
       else:
            SPL_flap=noise_trailing_edge_flap(Sf,cf,deltaf,slots,velocity,M,phi,theta,distance,frequency) #Trailing Edge Flaps Noise

       if gear==0:
            SPL_main_landing_gear=np.zeros(24)
            SPL_nose_landing_gear=np.zeros(24)
       else:
            SPL_main_landing_gear=noise_landing_gear(Dp,Hp,wheels,M,velocity,phi,theta,distance,frequency)+3       #Main Landing Gear Noise
            SPL_nose_landing_gear=noise_landing_gear(Dn,Hn,wheels,M,velocity,phi,theta,distance,frequency)       #Nose Landing Gear Noise



        #Total Airframe Noise
       SPL_total=10.*np.log10(10.0**(0.1*SPL_wing)+10.0**(0.1*SPLht)+10**(0.1*SPL_flap)+ \
            10.0**(0.1*SPL_slat)+10.0**(0.1*SPL_main_landing_gear)+10.0**(0.1*SPL_nose_landing_gear))

       fid.write('\n')
       fid.write('Polar angle = ' + str(angle) + '\n')
       fid.write('f		wing    		ht  		vt  	flap    	main LDG    nose LDG    slat    	total' + '\n')

       for id in range(0,24):
           fid.write(str((frequency[id])) + '       ')
           fid.write(str('%2.2f' % SPL_wing[id]) + '       ')
           fid.write(str('%2.2f' % SPLht[id]) + '       ')
           fid.write(str('%2.2f' % SPLvt[id]) + '       ')
           fid.write(str('%2.2f' % SPL_flap[id]) + '       ')
           fid.write(str('%2.2f' % SPL_main_landing_gear[id]) + '       ')
           fid.write(str('%2.2f' % SPL_nose_landing_gear[id]) + '       ')
           fid.write(str('%2.2f' % SPL_slat[id]) + '       ')
           fid.write(str('%2.2f' % SPL_total[id]) + '       ')
           fid.write('\n')

    fid.close