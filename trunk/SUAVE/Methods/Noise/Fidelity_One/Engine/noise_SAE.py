#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      CARIDSIL
#
# Created:     26/06/2015
# Copyright:   (c) CARIDSIL 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

from angle_of_attack_effect import angle_of_attack_effect
from external_plug_effect import external_plug_effect
from ground_proximity_effect import ground_proximity_effect
from jet_installation_effect import jet_installation_effect
from mixed_noise_component import mixed_noise_component
from noise_source_location import noise_source_location
from primary_noise_component import primary_noise_component
from secondary_noise_component import secondary_noise_component


def noise_SAE (turbofan):

    #SAE ARP*876D 1994
    """This method predicts the free-field 1/3 Octave Band SPL of coaxial subsonic
    jets for turbofan engines under the following conditions:
        a) Flyover (observer on ground)
        b) Static (observer on ground)
        c) In-flight or in-flow (observer on airplane or in a wind tunnel)

        Inputs:
                    vehicle	 - SUAVE type vehicle

                    includes these fields:
                        Velocity_primary           - Primary jet flow velocity
                        Temperature_primary        - Primary jet flow temperature
                        Pressure_primary           - Primary jet flow pressure
                        Area_primary               - Area of the primary nozzle
                        Velocity_secondary         - Secondary jet flow velocity
                        Temperature_secondary      - Secondary jet flow temperature
                        Pressure_secondary         - Secondary jet flow pressure
                        Area_secondary             - Area of the secondary nozzle
                        AOA                        - Angle of attack
                        Velocity_aircraft          - Aircraft velocity
                        Altitute                   - Altitude
                        N1                         - Fan rotational speed [rpm]
                        EXA                        - Distance from fan face to fan exit/ fan diameter
                        Plug_diameter              - Diameter of the engine external plug [m]
                        Engine_height              - Engine centerline height above the ground plane
                        distance_microphone        - Distance from the nozzle exhaust to the microphones
                        angles                     - Array containing the desired polar angles


                    airport   - SUAVE type airport data, with followig fields:
                        atmosphere                  - Airport atmosphere (SUAVE type)
                        altitude                    - Airport altitude
                        delta_isa                   - ISA Temperature deviation


                Outputs: One Third Octave Band SPL [dB]
                    SPL_p                           - Sound Pressure Level of the primary jet
                    SPL_s                           - Sound Pressure Level of the secondary jet
                    SPL_m                           - Sound Pressure Level of the mixed jet
                    SPL_total                       - Sound Pressure Level of the total jet noise

                Assumptions:
                    ."""


    #unpack
    Velocity_primary=np.atleast1d(turbofan.core_nozzle.outputs.velocity)
    
    
    
    #Necessary input for the code
    pressure_amb=997962 #[Pa]
    pressure_isa=101325 #[Pa]
    R_gas=287.1         #[J/kg K]
    gama_primary=1.37 #Corretion for the primary jet
    gama=1.4

    #Primary jet input information
    Velocity_primary = 217.2
    Temperature_primary = 288
    Pressure_primary=112000

    Area_primary = 0.001963
    Diameter_primary =0.05

    #Secondary jet input information
    Velocity_secondary = 216.8
    Temperature_secondary = 288.0
    Pressure_secondary=110000

    Area_secondary = 0.00394
    Diameter_secondary=0.0867

    #Aircraft input information
    Velocity_aircraft = 0.0
    Altitude=0.0
    AOA=0.0

    # Engine input information
    EXA = 1 #(distance from fan face to fan exit/ fan diameter)
    N1 = 3000 #Fan rotational speed [rpm]
    Plug_diameter = 0.1

    distance_microphone = 13.08
    angles=[60,70,80,90,100,110,120,130] #Array of desired polar angles
    Xo=1 #Acoustic center of reference [m]
    dist= 1.5 #Dist?ncia da sa?da de exaust?o at? o microfone(KQ) [m]

    #Flags for definition of near-fiel or wind-tunnel data
    near_field=0
    tunnel=1

    # Geometry information for the installation effects function
    Xe=1
    Ye=1
    Ce=2

    #Engine centerline heigh above the ground plane
    engine_height = 1

    """Starting the main program"""


    #Arrays for the calculation of atmospheric attenuation
    Acq=np.array((0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.5, 1.9, 2.5, 2.9, 3.6, 4.9, 6.8))
    Acf=np.array((0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.8, 2.5, 3.0, 4.2, 6.1, 9.0))

    #Desired frequency range for noise evaluation
    frequency=np.array((50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, \
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000))



    #Calculation of ambient properties
    temperature_ambient = 288.15 - 6.5*10**(-3)*Altitude           #temperature, K
    sound_ambient=np.sqrt(1.4*287*temperature_ambient)           #sound speed, m/s
    density_ambient = (1/(temperature_ambient/288.15))*1.225     #density, kg/m3



    # Jet Flow Parameters

    #Primary and Secondary jets
    Cpp=R_gas/(1-1/gama_primary)
    Cp=R_gas/(1-1/gama)
    density_primary=Pressure_primary/(R_gas*Temperature_primary-(0.5*R_gas*Velocity_primary**2/Cpp))
    density_secondary=Pressure_secondary/(R_gas*Temperature_secondary-(0.5*R_gas*Velocity_secondary**2/Cp))

    mass_flow_primary=Area_primary*Velocity_primary*density_primary
    mass_flow_secondary=Area_secondary*Velocity_secondary*density_secondary

    #Mach number of the external flow - based on the aircraft velocity
    Mach_aircraft=Velocity_aircraft/sound_ambient

    #Calculation Procedure for the Mixed Jet Flow Parameters
    Velocity_mixed = (mass_flow_primary*Velocity_primary+mass_flow_secondary*Velocity_secondary)/ \
            (mass_flow_primary+mass_flow_secondary)
    Temperature_mixed =(mass_flow_primary*Temperature_primary+mass_flow_secondary*Temperature_secondary)/ \
            (mass_flow_primary+mass_flow_secondary)
    density_mixed = pressure_amb/(R_gas*Temperature_mixed-(0.5*R_gas*Velocity_mixed**2/Cp))
    Area_mixed = Area_primary*density_primary*Velocity_primary*(1+(mass_flow_secondary/mass_flow_primary))/ \
            (density_mixed*Velocity_mixed)
    Diameter_mixed = (4*Area_mixed/np.pi)**0.5

    #**********************************************
    # START OF THE NOISE PROCEDURE CALCULATIONS
    #**********************************************

    XBPR = mass_flow_secondary/mass_flow_primary - 5.5
    if XBPR<0:
            XBPR=0
    elif XBPR>4:
            XBPR=4

    #Auxiliary parameter defined as DVPS
    DVPS = np.abs((Velocity_primary - (Velocity_secondary*Area_secondary+Velocity_aircraft*Area_primary)/(Area_secondary+Area_primary)))
    if DVPS<0.3:
        DVPS=0.3

    # Calculation of the Strouhal number for each jet component (p-primary, s-secondary, m-mixed)
    Str_p = frequency*Diameter_primary/(DVPS)  #Primary jet
    Str_s = frequency*Diameter_mixed/(Velocity_secondary-Velocity_aircraft) #Secondary jet
    Str_m = frequency*Diameter_mixed/(Velocity_mixed-Velocity_aircraft) #Mixed jet

    #Calculation of the Excitation adjustment parameter
    #Excitation Strouhal Number
    excitation_Strouhal = (N1/60)*(Diameter_mixed/Velocity_mixed)
    if (excitation_Strouhal > 0.25 and excitation_Strouhal < 0.5):
        SX=0.0
    else:
        SX=50*(excitation_Strouhal-0.25)*(excitation_Strouhal-0.5)

    #Effectiveness
    exps = np.exp(-SX)

    #Spectral Shape Factor
    exs=5*exps*np.exp(-(np.log10(Str_m/(2*excitation_Strouhal+0.00001)))**2)

    #Fan Duct Lenght Factor
    exd=np.exp(0.6-(EXA)**0.5)

    #Excitation source location factor (zk)
    zk=1-0.4*(exd)*(exps)

    #Defining each array before the main loop
    theta=np.zeros(24)
    B=np.zeros(24)
    theta_p=np.ones(24)*np.pi/2
    theta_s=np.ones(24)*np.pi/2
    theta_m=np.ones(24)*np.pi/2
    EX_p=np.zeros(24)
    EX_s=np.zeros(24)
    EX_m=np.zeros(24)
    exc=np.zeros(24)
    SPL_p=np.zeros(24)
    SPL_s=np.zeros(24)
    SPL_m=np.zeros(24)
    PG_p=np.zeros(24)
    PG_s=np.zeros(24)
    PG_m=np.zeros(24)

    # Open output file to print the results
    filename = 'SAE_Noise.dat'
    fid = open(filename,'w')

    #Main loop for the desired polar angles
    for jind in range (8):
        theta=angles[jind]*np.pi/180

        #Call function noise source location for the calculation of theta
        thetaj=noise_source_location(dist,B,Xo,zk,Diameter_primary,theta_p,Area_primary,Area_secondary,distance_microphone,Diameter_secondary,theta,theta_s,theta_m,Diameter_mixed,Velocity_primary,Velocity_secondary,Velocity_mixed,Velocity_aircraft,sound_ambient,Str_m,Str_s)

        # Loop for the frequency array range
        for i in range(0,24):
           #Calculation of the Directivity Factor
            if (theta_m[i] <=1.4):
                exc[i]= sound_ambient/Velocity_mixed
            elif (theta_m[i]>1.4):
                exc[i]=(sound_ambient/Velocity_mixed)*(1-(1.8/np.pi)*(theta_m[i]-1.4))

            #Acoustic excitation adjustment (EX)
            EX_m[i]=exd*exs[i]*exc[i]   #mixed component - dependant of the frequency
        EX_p=+5*exd*exps   #primary component - no frequency dependance
        EX_s=2*sound_ambient/(Velocity_secondary*(zk)) #secondary component - no frequency dependance


        distance_primary=dist #*sin(theta)/sin(thetaj(i,1));
        distance_secondary=dist #*sin(theta)/sin(thetaj(i,2));
        distance_mixed=dist #*sin(theta)/sin(thetaj(i,3));

        #Noise attenuation due to Ambient Pressure
        dspl_ambient_pressure = 20*np.log10(pressure_amb/pressure_isa)

        #Noise attenuation due to Density
        dspl_density_p=20*np.log10((density_primary+density_secondary)/(2*density_ambient))
        dspl_density_s=20*np.log10((density_secondary+density_ambient)/(2*density_ambient))
        dspl_density_m=20*np.log10((density_mixed+density_ambient)/(2*density_ambient))

        #Noise attenuation due to Spherical divergence
        dspl_spherical_p=20*np.log10(Diameter_primary/distance_primary)
        dspl_spherical_s=20*np.log10(Diameter_mixed/distance_secondary)
        dspl_spherical_m=20*np.log10(Diameter_mixed/distance_mixed)

        #Noise attenuation due to Geometric Near-Field
        if near_field ==0:
            dspl_geometric_p=0.0
            dspl_geometric_s=0.0
            dspl_geometric_m=0.0
        elif near_field ==1:
            dspl_geometric_p=-10*np.log10(1+(2*Diameter_primary+(Diameter_primary*sound_ambient/frequency))/distance_primary)
            dspl_geometric_s=-10*np.log10(1+(2*Diameter_mixed+(Diameter_mixed*sound_ambient/frequency))/distance_secondary)
            dspl_geometric_m=-10*np.log10(1+(2*Diameter_mixed+(Diameter_mixed*sound_ambient/frequency))/distance_mixed)

        #Noise attenuation due to Acoustic Near-Field
        if near_field ==0:
            dspl_acoustic_p=0.0;
            dspl_acoustic_s=0.0;
            dspl_acoustic_m=0.0;
        elif near_field ==1:
            dspl_acoustic_p=10*np.log10(1+0.13*(sound_ambient/(distance_primary*frequency))**2)
            dspl_acoustic_s=10*np.log10(1+0.13*(sound_ambient/(distance_secondary*frequency))**2)
            dspl_acoustic_m=10*np.log10(1+0.13*(sound_ambient/(distance_mixed*frequency))**2)

        #Atmospheric attenuation coefficient
        if tunnel==0:
            f_primary=frequency/(1-Mach_aircraft*np.cos(theta_p))
            f_secondary=frequency/(1-Mach_aircraft*np.cos(theta_s))
            f_mixed=frequency/(1-Mach_aircraft*np.cos(theta_m))
            Aci= Acf + ((temperature_ambient-273)-15)/10*(Acq-Acf)
            Ac_primary=np.interp(f_primary,frequency,Aci)
            Ac_secondary=np.interp(f_secondary,frequency,Aci)
            Ac_mixed=np.interp(f_mixed,frequency,Aci)
            dspl_attenuation_p=-Ac_primary*distance_primary
            dspl_attenuation_s=-Ac_secondary*distance_secondary
            dspl_attenuation_m=-Ac_mixed*distance_mixed

        elif tunnel==1: #These corrections are not applicable for jet rigs or static conditions
            dspl_attenuation_p=np.zeros(24)
            dspl_attenuation_s=np.zeros(24)
            dspl_attenuation_m=np.zeros(24)
            EX_m=np.zeros(24)
            EX_p=0
            EX_s=0

        #Calculation of the total noise attenuation (p-primary, s-secondary, m-mixed components)
        DSPL_p=dspl_ambient_pressure+dspl_density_p+dspl_geometric_p+dspl_acoustic_p+dspl_attenuation_p+dspl_spherical_p
        DSPL_s=dspl_ambient_pressure+dspl_density_s+dspl_geometric_s+dspl_acoustic_s+dspl_attenuation_s+dspl_spherical_s
        DSPL_m=dspl_ambient_pressure+dspl_density_m+dspl_geometric_m+dspl_acoustic_m+dspl_attenuation_m+dspl_spherical_m


        #Calculation of interference effects on jet noise
        ATK_m=angle_of_attack_effect(AOA,Mach_aircraft,theta_m)
        INST_s= jet_installation_effect(Xe,Ye,Ce,theta_s,Diameter_mixed)
        Plug=external_plug_effect(Velocity_primary,Velocity_secondary, Velocity_mixed, Diameter_primary,Diameter_secondary,Diameter_mixed, Plug_diameter, sound_ambient, theta_p,theta_s,theta_m)
        GPROX_m=ground_proximity_effect(Velocity_mixed,sound_ambient,theta_m,engine_height,Diameter_mixed,frequency)

        #Calculation of the sound pressure level for each jet component
        SPL_p=primary_noise_component(SPL_p,Velocity_primary,Temperature_primary,R_gas,theta_p,DVPS,sound_ambient,Velocity_secondary,Velocity_aircraft,Area_primary,Area_secondary,DSPL_p,EX_p,Str_p) + Plug[0]
        SPL_s=secondary_noise_component(SPL_s,Velocity_primary,theta_s,sound_ambient,Velocity_secondary,Velocity_aircraft,Area_primary,Area_secondary,DSPL_s,EX_s,Str_s) + Plug[1] + INST_s
        SPL_m=mixed_noise_component(SPL_m,Velocity_primary,theta_m,sound_ambient,Velocity_secondary,Velocity_aircraft,Area_primary,Area_secondary,DSPL_m,EX_m,Str_m,Velocity_mixed,XBPR) + Plug[2] + ATK_m + GPROX_m

        #Sum of the Total Noise
        SPL_total = 10 * np.log10(10**(0.1*SPL_p)+10**(0.1*SPL_s)+10**(0.1*SPL_m))

        #Printing the output solution for the engine noise calculation
        fid.write('\n')
        fid.write('Polar angle = ' + str(angles[jind]) + '\n')
        fid.write('f		Primary Jet  Secondary Jet  Mixed jet  		Total' + '\n')

        for id in range(0,24):
               fid.write(str((frequency[id])) + '       ')
               fid.write(str('%3.3f' % SPL_p[id]) + '       ')
               fid.write(str('%3.3f' % SPL_s[id]) + '       ')
               fid.write(str('%3.3f' % SPL_m[id]) + '       ')
               fid.write(str('%3.3f' % SPL_total[id]) + '       ')
               fid.write('\n')

    fid.close

    return()