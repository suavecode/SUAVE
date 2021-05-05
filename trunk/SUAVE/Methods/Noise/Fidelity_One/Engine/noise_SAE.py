## @ingroupMethods-Noise-Fidelity_One-Engine
# noise_SAE.py
# 
# Created:  May 2015, C. Ilario
# Modified: Nov 2015, C. Ilario
#           Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import numpy as np
from SUAVE.Core            import Units , Data 

from .angle_of_attack_effect    import angle_of_attack_effect
from .external_plug_effect      import external_plug_effect
from .ground_proximity_effect   import ground_proximity_effect
from .jet_installation_effect   import jet_installation_effect
from .mixed_noise_component     import mixed_noise_component
from .noise_source_location     import noise_source_location
from .primary_noise_component   import primary_noise_component
from .secondary_noise_component import secondary_noise_component

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import epnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import atmospheric_attenuation 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import senel_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import dbA_noise 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import print_engine_output

# ----------------------------------------------------------------------        
#   Noise SAE
# ----------------------------------------------------------------------    

## @ingroupMethods-Noise-Fidelity_One-Engine
def noise_SAE(turbofan,segment,analyses,config,settings,ioprint = 0, filename = 0):  
    """This method predicts the free-field 1/3 Octave Band SPL of coaxial subsonic
       jets for turbofan engines under the following conditions:
       a) Flyover (observer on ground)
       b) Static (observer on ground)
       c) In-flight or in-flow (observer on airplane or in a wind tunnel)

    Assumptions:
        SAE ARP876D: Gas Turbine Jet Exhaust Noise Prediction

    Inputs:
        vehicle	 - SUAVE type vehicle 
        includes these fields:
            Velocity_primary           - Primary jet flow velocity                           [m/s]
            Temperature_primary        - Primary jet flow temperature                        [m/s]
            Pressure_primary           - Primary jet flow pressure                           [Pa]
            Area_primary               - Area of the primary nozzle                          [m^2]
            Velocity_secondary         - Secondary jet flow velocity                         [m/s]
            Temperature_secondary      - Secondary jet flow temperature                      [m/s]
            Pressure_secondary         - Secondary jet flow pressure                         [Pa]
            Area_secondary             - Area of the secondary nozzle                        [m^2]
            AOA                        - Angle of attack                                     [rad]
            Velocity_aircraft          - Aircraft velocity                                   [m/s]
            Altitude                   - Altitude                                            [m]
            N1                         - Fan rotational speed                                [rpm]
            EXA                        - Distance from fan face to fan exit/ fan diameter    [m]
            Plug_diameter              - Diameter of the engine external plug                [m]
            Engine_height              - Engine centerline height above the ground plane     [m]
            distance_microphone        - Distance from the nozzle exhaust to the microphones [m]
            angles                     - Array containing the desired polar angles           [rad]


        airport   - SUAVE type airport data, with followig fields:
            atmosphere                  - Airport atmosphere (SUAVE type)
            altitude                    - Airport altitude
            delta_isa                   - ISA Temperature deviation


    Outputs: One Third Octave Band SPL [dB]
        SPL_p                           - Sound Pressure Level of the primary jet            [dB]
        SPL_s                           - Sound Pressure Level of the secondary jet          [dB]
        SPL_m                           - Sound Pressure Level of the mixed jet              [dB]
        SPL_total                       - Sound Pressure Level of the total jet noise        [dB]

    """ 
    # unpack 
    Velocity_primary       = turbofan.core_nozzle.noise_speed * 0.92*(turbofan.design_thrust/52700.)   
    Temperature_primary    = segment.conditions.noise.sources.turbofan.core.exit_stagnation_temperature[:,0] 
    Pressure_primary       = segment.conditions.noise.sources.turbofan.core.exit_stagnation_pressure[:,0] 

    Velocity_secondary     = turbofan.fan_nozzle.noise_speed * (turbofan.design_thrust/52700.) 
    Temperature_secondary  = segment.conditions.noise.sources.turbofan.fan.exit_stagnation_temperature[:,0] 
    Pressure_secondary     = segment.conditions.noise.sources.turbofan.fan.exit_stagnation_pressure[:,0] 

    N1                     = turbofan.fan.rotation* 0.92*(turbofan.design_thrust/52700.)
    Diameter_primary       = turbofan.core_nozzle_diameter
    Diameter_secondary     = turbofan.fan_nozzle_diameter
    engine_height          = turbofan.engine_height
    EXA                    = turbofan.exa
    Plug_diameter          = turbofan.plug_diameter 
    Xe                     = turbofan.geometry_xe
    Ye                     = turbofan.geometry_ye
    Ce                     = turbofan.geometry_Ce

    Velocity_aircraft      = np.float(segment.conditions.freestream.velocity[0,0]) 
    Altitude               = segment.conditions.freestream.altitude[:,0] 
    AOA                    = np.mean(segment.conditions.aerodynamics.angle_of_attack / Units.deg)

    noise_time             = segment.conditions.frames.inertial.time[:,0]

    # unpack
    distance_microphone = segment.dist   
    angles              = segment.theta  
    phi                 = segment.phi     

    nsteps = len(noise_time)        

    #Preparing matrix for noise calculation
    sound_ambient       = np.zeros(nsteps)
    density_ambient     = np.zeros(nsteps)
    viscosity           = np.zeros(nsteps)
    temperature_ambient = np.zeros(nsteps)
    pressure_amb        = np.zeros(nsteps)
    Mach_aircraft       = np.zeros(nsteps)

    if type(Velocity_primary) == float:
        Velocity_primary    = np.ones(nsteps)*Velocity_primary

    if type(Velocity_secondary) == float:
        Velocity_secondary  = np.ones(nsteps)*Velocity_secondary

    # ==============================================
    # Computing atmospheric conditions
    # ==============================================  
    sound_ambient       =   segment.conditions.freestream.speed_of_sound[:,0]
    density_ambient     =   segment.conditions.freestream.density[:,0]
    viscosity           =   segment.conditions.freestream.dynamic_viscosity[:,0]
    temperature_ambient =   segment.conditions.freestream.temperature[:,0]
    pressure_amb        =   segment.conditions.freestream.pressure[:,0]

    #Base parameters necessary input for the noise code
    pressure_isa = 101325 # [Pa]
    R_gas        = 287.1  # [J/kg K]
    gamma_primary = 1.37   # Corretion for the primary jet
    gamma         = 1.4

    #Calculation of nozzle areas
    Area_primary   = np.pi*(Diameter_primary/2)**2 
    Area_secondary =  np.pi*(Diameter_secondary/2)**2 

    Xo = 0 # Acoustic center of reference [m] - Used for wind tunnel acoustic data

    # Flags for definition of near-fiel or wind-tunnel data
    near_field = 0
    tunnel     = 0

    """Starting the main program"""

    #Arrays for the calculation of atmospheric attenuation
    Acq = np.array((0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.5, 1.9, 2.5, 2.9, 3.6, 4.9, 6.8))
    Acf = np.array((0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.8, 2.5, 3.0, 4.2, 6.1, 9.0))

    #Desired frequency range for noise evaluation
    frequency = settings.center_frequencies[5:] 
    num_f     = len(frequency)

    # Defining each array before the main loop
    B       = np.zeros(num_f)
    theta_p = np.ones(num_f)*np.pi/2
    theta_s = np.ones(num_f)*np.pi/2
    theta_m = np.ones(num_f)*np.pi/2
    EX_p    = np.zeros(num_f)
    EX_s    = np.zeros(num_f)
    EX_m    = np.zeros(num_f)
    exc     = np.zeros(num_f)
    SPL_p   = np.zeros(num_f)
    SPL_s   = np.zeros(num_f)
    SPL_m   = np.zeros(num_f)
    PG_p    = np.zeros(num_f)
    PG_s    = np.zeros(num_f)
    PG_m    = np.zeros(num_f)

    dspl_attenuation_p = np.zeros(num_f)
    dspl_attenuation_s = np.zeros(num_f)
    dspl_attenuation_m = np.zeros(num_f)

    SPL_total_history     = np.zeros((nsteps,num_f))
    SPL_primary_history   = np.zeros((nsteps,num_f))
    SPL_secondary_history = np.zeros((nsteps,num_f))
    SPL_mixed_history     = np.zeros((nsteps,num_f))

    # Noise history in dBA
    SPLt_dBA         = np.zeros((nsteps,num_f))
    SPLt_dBA_history = np.zeros((nsteps,num_f))
    SPLt_dBA_max     = np.zeros(nsteps)

    # Start loop for each position of aircraft 
    for id in range(0,nsteps):

        # Jet Flow Parameters

        #Primary and Secondary jets
        Cpp = R_gas/(1-1/gamma_primary)
        Cp  = R_gas/(1-1/gamma)

        density_primary   = Pressure_primary[id]/(R_gas*Temperature_primary[id]-(0.5*R_gas*Velocity_primary[id]**2/Cpp))
        density_secondary = Pressure_secondary[id]/(R_gas*Temperature_secondary[id]-(0.5*R_gas*Velocity_secondary[id]**2/Cp))

        mass_flow_primary   = Area_primary*Velocity_primary[id]*density_primary
        mass_flow_secondary = Area_secondary*Velocity_secondary[id]*density_secondary

        #Mach number of the external flow - based on the aircraft velocity
        Mach_aircraft[id] = Velocity_aircraft/sound_ambient[id]

        #Calculation Procedure for the Mixed Jet Flow Parameters
        Velocity_mixed = (mass_flow_primary*Velocity_primary[id]+mass_flow_secondary*Velocity_secondary[id])/ \
            (mass_flow_primary+mass_flow_secondary)
        Temperature_mixed =(mass_flow_primary*Temperature_primary[id]+mass_flow_secondary*Temperature_secondary[id])/ \
            (mass_flow_primary+mass_flow_secondary)
        density_mixed = pressure_amb[id]/(R_gas*Temperature_mixed-(0.5*R_gas*Velocity_mixed**2/Cp))
        Area_mixed = Area_primary*density_primary*Velocity_primary[id]*(1+(mass_flow_secondary/mass_flow_primary))/ \
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
        DVPS = np.abs((Velocity_primary[id] - (Velocity_secondary[id]*Area_secondary+Velocity_aircraft*Area_primary)/\
                       (Area_secondary+Area_primary)))
        if DVPS<0.3:
            DVPS=0.3

        # Calculation of the Strouhal number for each jet component (p-primary, s-secondary, m-mixed)
        Str_p = frequency*Diameter_primary/(DVPS)  #Primary jet
        Str_s = frequency*Diameter_mixed/(Velocity_secondary[id]-Velocity_aircraft) #Secondary jet
        Str_m = frequency*Diameter_mixed/(Velocity_mixed-Velocity_aircraft) #Mixed jet

        #Calculation of the Excitation adjustment parameter
        #Excitation Strouhal Number
        excitation_Strouhal = (N1/60)*(Diameter_mixed/Velocity_mixed)
        if (excitation_Strouhal > 0.25 and excitation_Strouhal < 0.5):
            SX = 0.0
        else:
            SX = 50*(excitation_Strouhal-0.25)*(excitation_Strouhal-0.5)

        #Effectiveness
        exps = np.exp(-SX)

        #Spectral Shape Factor
        exs = 5*exps*np.exp(-(np.log10(Str_m/(2*excitation_Strouhal+0.00001)))**2)

        #Fan Duct Lenght Factor
        exd = np.exp(0.6-(EXA)**0.5)

        #Excitation source location factor (zk)
        zk = 1-0.4*(exd)*(exps)    

        #Main loop for the desired polar angles

        theta = angles[id]

        # Call function noise source location for the calculation of theta
        thetaj = noise_source_location(B,Xo,zk,Diameter_primary,theta_p,Area_primary,Area_secondary,distance_microphone[id],
                                       Diameter_secondary,theta,theta_s,theta_m,Diameter_mixed,Velocity_primary[id],
                                       Velocity_secondary[id],Velocity_mixed,Velocity_aircraft,sound_ambient[id],Str_m,Str_s)

        # Loop for the frequency array range
        for i in range(0,num_f):
            #Calculation of the Directivity Factor
            if (theta_m[i] <=1.4):
                exc[i] = sound_ambient[id]/Velocity_mixed
            elif (theta_m[i]>1.4):
                exc[i] =(sound_ambient[id]/Velocity_mixed)*(1-(1.8/np.pi)*(theta_m[i]-1.4))

            #Acoustic excitation adjustment (EX)
            EX_m[i] = exd*exs[i]*exc[i]   #mixed component - dependant of the frequency

        EX_p = +5*exd*exps   #primary component - no frequency dependance
        EX_s = 2*sound_ambient[id]/(Velocity_secondary[id]*(zk)) #secondary component - no frequency dependance    

        distance_primary   = distance_microphone[id] 
        distance_secondary = distance_microphone[id] 
        distance_mixed     = distance_microphone[id]

        #Noise attenuation due to Ambient Pressure
        dspl_ambient_pressure = 20*np.log10(pressure_amb[id]/pressure_isa)

        #Noise attenuation due to Density Gradientes
        dspl_density_p = 20*np.log10((density_primary+density_secondary)/(2*density_ambient[id]))
        dspl_density_s = 20*np.log10((density_secondary+density_ambient[id])/(2*density_ambient[id]))
        dspl_density_m = 20*np.log10((density_mixed+density_ambient[id])/(2*density_ambient[id]))

        #Noise attenuation due to Spherical divergence
        dspl_spherical_p = 20*np.log10(Diameter_primary/distance_primary)
        dspl_spherical_s = 20*np.log10(Diameter_mixed/distance_secondary)
        dspl_spherical_m = 20*np.log10(Diameter_mixed/distance_mixed)

        # Noise attenuation due to Geometric Near-Field
        if near_field ==0:
            dspl_geometric_p = 0.0
            dspl_geometric_s = 0.0
            dspl_geometric_m = 0.0
        elif near_field ==1:
            dspl_geometric_p = -10*np.log10(1+(2*Diameter_primary+(Diameter_primary*sound_ambient[id]/frequency))/distance_primary)
            dspl_geometric_s = -10*np.log10(1+(2*Diameter_mixed+(Diameter_mixed*sound_ambient[id]/frequency))/distance_secondary)
            dspl_geometric_m = -10*np.log10(1+(2*Diameter_mixed+(Diameter_mixed*sound_ambient[id]/frequency))/distance_mixed)

        # Noise attenuation due to Acoustic Near-Field
        if near_field ==0:
            dspl_acoustic_p = 0.0;
            dspl_acoustic_s = 0.0;
            dspl_acoustic_m = 0.0;
        elif near_field ==1:
            dspl_acoustic_p = 10*np.log10(1+0.13*(sound_ambient[id]/(distance_primary*frequency))**2)
            dspl_acoustic_s = 10*np.log10(1+0.13*(sound_ambient[id]/(distance_secondary*frequency))**2)
            dspl_acoustic_m = 10*np.log10(1+0.13*(sound_ambient[id]/(distance_mixed*frequency))**2)

        # Atmospheric attenuation coefficient
        if tunnel==0:
            f_primary   = frequency/(1-Mach_aircraft[id]*np.cos(theta_p))
            f_secondary = frequency/(1-Mach_aircraft[id]*np.cos(theta_s))
            f_mixed     = frequency/(1-Mach_aircraft[id]*np.cos(theta_m))
            Aci         = Acf + ((temperature_ambient[id]-0)-15)/10*(Acq-Acf)    

            Ac_primary   = np.interp(f_primary,frequency,Aci)
            Ac_secondary = np.interp(f_secondary,frequency,Aci)
            Ac_mixed     = np.interp(f_mixed,frequency,Aci)

                #Atmospheric attenuation
            delta_atmo = atmospheric_attenuation(distance_primary)

            dspl_attenuation_p = -delta_atmo 
            dspl_attenuation_s = -delta_atmo 
            dspl_attenuation_m = -delta_atmo 

        elif tunnel==1: #These corrections are not applicable for jet rigs or static conditions
            dspl_attenuation_p = np.zeros(num_f)
            dspl_attenuation_s = np.zeros(num_f)
            dspl_attenuation_m = np.zeros(num_f)
            EX_m = np.zeros(num_f)
            EX_p = 0
            EX_s = 0

        # Calculation of the total noise attenuation (p-primary, s-secondary, m-mixed components)
        DSPL_p = dspl_ambient_pressure+dspl_density_p+dspl_geometric_p+dspl_acoustic_p+dspl_attenuation_p+dspl_spherical_p
        DSPL_s = dspl_ambient_pressure+dspl_density_s+dspl_geometric_s+dspl_acoustic_s+dspl_attenuation_s+dspl_spherical_s
        DSPL_m = dspl_ambient_pressure+dspl_density_m+dspl_geometric_m+dspl_acoustic_m+dspl_attenuation_m+dspl_spherical_m


        # Calculation of interference effects on jet noise
        ATK_m   = angle_of_attack_effect(AOA,Mach_aircraft[id],theta_m)
        INST_s  = jet_installation_effect(Xe,Ye,Ce,theta_s,Diameter_mixed)
        Plug    = external_plug_effect(Velocity_primary[id],Velocity_secondary[id], Velocity_mixed, Diameter_primary,Diameter_secondary,
                                       Diameter_mixed, Plug_diameter, sound_ambient[id], theta_p,theta_s,theta_m)

        GPROX_m = ground_proximity_effect(Velocity_mixed,sound_ambient[id],theta_m,engine_height,Diameter_mixed,frequency)

        # Calculation of the sound pressure level for each jet component
        SPL_p = primary_noise_component(SPL_p,Velocity_primary[id],Temperature_primary[id],R_gas,theta_p,DVPS,sound_ambient[id],
                                        Velocity_secondary[id],Velocity_aircraft,Area_primary,Area_secondary,DSPL_p,EX_p,Str_p) + Plug.PG_p

        SPL_s = secondary_noise_component(SPL_s,Velocity_primary[id],theta_s,sound_ambient[id],Velocity_secondary[id],
                                          Velocity_aircraft,Area_primary,Area_secondary,DSPL_s,EX_s,Str_s) + Plug.PG_s + INST_s

        SPL_m = mixed_noise_component(SPL_m,Velocity_primary[id],theta_m,sound_ambient[id],Velocity_secondary[id],
                                      Velocity_aircraft,Area_primary,Area_secondary,DSPL_m,EX_m,Str_m,Velocity_mixed,XBPR) + \
            Plug.PG_m + ATK_m + GPROX_m

        # Sum of the Total Noise
        SPL_total = 10 * np.log10(10**(0.1*SPL_p)+10**(0.1*SPL_s)+10**(0.1*SPL_m))

        # Store the SPL history     
        SPL_total_history[id][:]     = SPL_total[:]
        SPL_primary_history[id][:]   = SPL_p[:]
        SPL_secondary_history[id][:] = SPL_s[:]
        SPL_mixed_history[id][:]     = SPL_m[:]

        # Calculation of dBA based on the sound pressure time history
        SPLt_dBA                = dbA_noise(SPL_total)
        SPLt_dBA_history[id][:] = dbA_noise(SPL_total)
        SPLt_dBA_max[id]        = max(SPLt_dBA)

    # Calculation of the Perceived Noise Level EPNL based on the sound time history
    PNL_total               =  pnl_noise(SPL_total_history)    
    PNL_primary             =  pnl_noise(SPL_primary_history)  
    PNL_secondary           =  pnl_noise(SPL_secondary_history)  
    PNL_mixed               =  pnl_noise(SPL_mixed_history)  

    # Calculation of the tones corrections on the SPL for each component and total
    tone_correction_total     = noise_tone_correction(SPL_total_history) 
    tone_correction_primary   = noise_tone_correction(SPL_primary_history) 
    tone_correction_secondary = noise_tone_correction(SPL_secondary_history) 
    tone_correction_mixed     = noise_tone_correction(SPL_mixed_history) 

    # Calculation of the PLNT for each component and total
    PNLT_total     = PNL_total+tone_correction_total
    PNLT_primary   = PNL_primary+tone_correction_primary
    PNLT_secondary = PNL_secondary+tone_correction_secondary
    PNLT_mixed     = PNL_mixed+tone_correction_mixed

    # Calculation of the EPNL for each component and total
    EPNL_total     = epnl_noise(PNLT_total)
    EPNL_primary   = epnl_noise(PNLT_primary)
    EPNL_secondary = epnl_noise(PNLT_secondary)
    EPNL_mixed     = epnl_noise(PNLT_mixed)

    #Calculation of the SENEL total
    SENEL_total = senel_noise(SPLt_dBA_max)

    # Open output file to print the results
    SAE_Engine_Noise_Outputs = Data(
        filename               = filename,
        tag                    = config.tag,
        EPNL_total             = EPNL_total,
        PNLT_total             = PNLT_total,
        Velocity_aircraft      = Velocity_aircraft,
        noise_time             = noise_time,
        Altitude               = Altitude,
        Mach_aircraft          = Mach_aircraft,
        Velocity_primary       = Velocity_primary,
        Velocity_secondary     = Velocity_secondary,
        angles                 = angles,
        phi                    = phi,
        distance_microphone    = distance_microphone,
        PNLT_primary           = PNLT_primary,
        PNLT_secondary         = PNLT_secondary,
        PNLT_mixed             = PNLT_mixed,
        SPLt_dBA_max           = SPLt_dBA_max,
        EPNL_primary           = EPNL_primary,
        EPNL_secondary         = EPNL_secondary,
        EPNL_mixed             = EPNL_mixed,
        SENEL_total            = SENEL_total,
        nsteps                 = nsteps,
        frequency              = frequency,
        SPL_primary_history    = SPL_primary_history,
        SPL_secondary_history  = SPL_secondary_history,
        SPL_mixed_history      = SPL_mixed_history,
        SPL_total_history      = SPL_total_history)   
    
    if ioprint:
        print_engine_output(SAE_Engine_Noise_Outputs)

    engine_noise                   = Data()
    engine_noise.EPNL_total        = EPNL_total 
    engine_noise.SENEL_total       = SENEL_total
    engine_noise.SPL_spectrum      = SPL_total_history
    engine_noise.SPL               = SPL_arithmetic(SPL_total_history,sum_axis=1)
    engine_noise.SPL_dBA           = SPLt_dBA_max

    return engine_noise