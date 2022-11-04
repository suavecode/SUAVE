# SUAVE Imports 

# Imports    
import SUAVE
from SUAVE.Core import Units, Data 
from SUAVE.Components.Energy.Networks.Battery_Propeller                                       import Battery_Propeller
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_properties  import compute_airfoil_properties
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series        import compute_naca_4series
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity                        import propeller_mid_fidelity
from SUAVE.Analyses.Mission.Segments.Conditions                                               import Aerodynamics , Conditions
from SUAVE.Analyses.Mission.Segments.Segment                                                  import Segment
from scipy.interpolate import interp1d  

# Python Imports  
import sys 
import numpy as np 
import matplotlib.pyplot as plt  

sys.path.append('../Vehicles/Propellers')
# the analysis functions

from F8745_D4_Propeller  import F8745_D4_Propeller
from APC_11x4_Propeller import APC_11x4_Propeller 
# ----------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------- 
def main(): 
    '''This regression script is for validation and verification of the mid-fidelity acoustics
    analysis routine. "Experimental Data is obtained from Comparisons of predicted propeller
    noise with windtunnel ..." by Weir, D and Powers, J.
    '''  


    # ----------------------------------------------------------------------------------------------------------------------------------------
    #  Plots
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # Universal Plot Settings 
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.linewidth'] = 1. 
 
    PP = Data( 
        lw  = 1,                             # line_width               
        m   = 5,                             # markersize               
        lf  = 10,                            # legend_font_size         
        Slc = ['black','dimgray','silver' ], # SUAVE_line_colors        
        Slm = '^',                           # SUAVE_line_markers       
        Sls = '-',                           # SUAVE_line_styles        
        Elc = ['darkred','red','tomato'], # Experimental_line_colors 
        Elm = 's',                           # Experimental_line_markers
        Els = '',                          # Experimental_line_styles 
        Rlc = ['darkblue','blue','cyan'],    # Ref_Code_line_colors     
        Rlm = 'o',                           # Ref_Code_line_markers    
        Rls = ':',                           # Ref_Code_line_styles     
    )   
   
       
    # harmonic noise test 
    Hararmonic_Noise_Validation(PP)

    # broadband nosie test function 
    Broadband_Noise_Validation(PP)
    
    return 

    
    
# ------------------------------------------------------------------ 
# Harmonic Noise Validation
# ------------------------------------------------------------------  
def Hararmonic_Noise_Validation(PP):

    net                                = Battery_Propeller()
    net.number_of_propeller_engines    = 1                                      
    prop                               = F8745_D4_Propeller()  
    net.identical_propellers           = True  
    net.propellers.append(prop)  

    # Atmosheric & Run Conditions                                               
    a                       = 343.376
    T                       = 288.16889478  
    density                 = 1.2250	
    dynamic_viscosity       = 1.81E-5 

    theta                   = np.array([1,10,20,30.1,40,50,59.9,70,80,89.9,100,110,120.1,130,140,150.1,160,170,179])  
    S                       = 4.
    test_omega              = np.array([2390,2710,2630]) * Units.rpm    
    ctrl_pts                = len(test_omega)

    # Set twist target
    three_quarter_twist     = 21 * Units.degrees 
    n                       = len(prop.twist_distribution)
    beta                    = prop.twist_distribution
    beta_75                 = beta[round(n*0.75)] 
    delta_beta              = three_quarter_twist-beta_75
    prop.twist_distribution = beta + delta_beta 

    # microphone locations
    positions = np.zeros(( len(theta),3))
    for i in range(len(theta)):
        if theta[i]*Units.degrees < np.pi/2:
            positions[i][:] = [-S*np.cos(theta[i]*Units.degrees)  ,S*np.sin(theta[i]*Units.degrees), 0.0]
        else: 
            positions[i][:] = [S*np.sin(theta[i]*Units.degrees- np.pi/2)  ,S*np.cos(theta[i]*Units.degrees - np.pi/2), 0.0] 

    # Set up for Propeller Model

    conditions                                             = Aerodynamics()   
    conditions.freestream.density                          = np.ones((ctrl_pts,1)) * density
    conditions.freestream.dynamic_viscosity                = np.ones((ctrl_pts,1)) * dynamic_viscosity   
    conditions.freestream.speed_of_sound                   = np.ones((ctrl_pts,1)) * a 
    conditions.freestream.temperature                      = np.ones((ctrl_pts,1)) * T 
    conditions.frames.inertial.velocity_vector             = np.array([[77.2, 0. ,0.],[ 77.0,0.,0.], [ 77.2, 0. ,0.]])
    conditions.propulsion.throttle                         = np.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial           = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])
    
    prop.inputs.omega                                      = np.atleast_2d(test_omega).T
    prop.inputs.y_axis_rotation                            = np.ones_like(prop.inputs.omega)
    
    # Run Propeller model 
    F, Q, P, Cp , noise_data , etap                        = prop.spin(conditions) 

    # Prepare Inputs for Noise Model  
    conditions.noise.total_microphone_locations            = np.repeat(positions[ np.newaxis,:,: ],1,axis=0)
    conditions.aerodynamics.angle_of_attack                = np.ones((ctrl_pts,1))* 0. * Units.degrees 
    segment                                                = Segment() 
    segment.state.conditions                               = conditions
    segment.state.conditions.expand_rows(ctrl_pts)
    
    
    # Store Noise Data 
    noise                                      = SUAVE.Analyses.Noise.Fidelity_One() 
    settings                                   = noise.settings   
    num_mic                                    = len(conditions.noise.total_microphone_locations[0] )  
    conditions.noise.number_of_microphones     = num_mic
    
    # Run Fidelity One    
    propeller_noise                       = propeller_mid_fidelity(net.propellers,noise_data,segment,settings )
    F8745D4_SPL                           = propeller_noise.SPL     
    F8745D4_SPL_harmonic                  = propeller_noise.SPL_harmonic 
    F8745D4_SPL_broadband                 = propeller_noise.SPL_broadband  
    F8745D4_SPL_harmonic_bpf_spectrum     = propeller_noise.SPL_harmonic_bpf_spectrum      

    # ----------------------------------------------------------------------------------------------------------------------------------------
    #  Experimental Data
    # ----------------------------------------------------------------------------------------------------------------------------------------      

    harmonics              = np.arange(1,19)
    ANOPP_PAS_Case_1_60deg = np.array([105.82,101.08,100.13,97.581,94.035,89.095,82.957,
                                       80.609,81.052,72.718,70.772,68.023,67.072,53.949,
                                       np.inf,np.inf,np.inf,np.inf]) 

    Exp_Test_Case_1_60deg  = np.array([103.23,100.08,98.733,94.990,91.045,87.500,82.161,79.012,
                                       74.469,70.128,65.784,61.241,56.699,51.958,47.013,42.673,
                                       37.927,32.989,]) 

    ANOPP_PAS_Case_1_90deg = np.array([107.671,104.755,105.829,103.307,101.385,100.465,99.1399,
                                       96.8208,93.6988,91.7765,89.6573,86.5323,85.2098,83.4874,
                                       78.1692,75.4503,73.7248,72.0024]) 

    Exp_Test_Case_1_90deg  = np.array([108.077,107.554,105.626,103.307,100.988,100.068,98.9430,
                                       96.8208,93.6988,91.7796,89.6542,85.5295,85.0099,81.8879,
                                       77.9724,74.8566,73.1250,71.2057  ])    

    Exp_Test_Case_2_60deg = np.array([111.951,108.175,108.789,106.352,105.059,100.140,100.945,
                                      99.8430,93.9683,93.8203,91.1914,85.3167,85.3626,82.1580,
                                       78.1933,75.7552,80.1887,72.2133])

    ANOPP_PAS_Case_2_60deg = np.array([111.760,111.421,108.984,106.352,104.487,101.856,99.0369,
                                       95.4522,93.2050,89.4327,86.2313,82.6498,79.2559,75.4804,
                                       71.1356,68.1219,63.9663,60.0000]) 


    Exp_Test_Case_2_90deg  = np.array([115.587,113.363,115.520,113.868,113.365,111.331,
                                       113.491,110.507,109.999,109.873,107.649,106.949,
                                       106.822,103.079,103.715,102.633,99.6502,97.8095])

    ANOPP_PAS_Case_2_90deg = np.array([115.397,115.273,114.377,113.870,113.362,111.143,
                                       110.631,109.752,108.859,107.585,109.175,105.234,
                                       103.782,102.127,101.236,99.7790,98.7002,98.9523  ])  

    Exp_Test_Case_3_60deg  = np.array([110.93,107.28,108.60,106.28,104.17,99.377,
                                       100.69,100.28,95.688,95.094,92.975,84.365,
                                       84.533,82.224,77.622,77.411,78.152,74.312])
    ANOPP_PAS_Case_3_60deg = np.array([110.93,110.53,108.41,107.43,104.55,101.47,
                                       98.592,95.328,92.635,88.987,86.103,83.028,
                                       79.573,76.114,73.040,69.775,65.554,61.908 ])

    Exp_Test_Case_3_90deg  = np.array([114.499,112.135,114.674,112.898,112.299,111.308,
                                       112.473,110.894,109.510,109.303,107.724,107.124,
                                       106.133,102.790,103.758,101.983,99.2279,98.0404])

    ANOPP_PAS_Case_3_90deg = np.array([114.499,114.291,113.889,113.879,111.122,110.523,
                                       109.924,109.129,108.725,107.342,106.743,105.164,
                                       104.369,102.593,101.210,100.021,98.6401,96.6674])

    # plot results
    fig = plt.figure('Harmonic Test')
    fig.set_size_inches(12, 8)   
    axes = fig.add_subplot(2,3,1) 
    axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[0,6,:][:len(harmonics)]   , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_1_60deg                                      , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw, label = 'ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_1_60deg                                       , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,  label = 'Exp.')    
    axes.set_title('Case 1, $C_P$ = ' + str(round(Cp[0,0],3)))
    axes.set_ylabel('SPL (dB)') 
    axes.minorticks_on() 
    #plt.ylim((80,125))      

    # Test Case 2
    axes = fig.add_subplot(2,3,2) 
    axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[1,6,:][:len(harmonics)] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,   label = 'SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_2_60deg                                    , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw,   label = 'ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_2_60deg                                     , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,   label = 'Exp.')  
    #axes.set_ylabel('SPL (dB)') 
    axes.set_title('60 deg. 60 deg. Case 2, $C_P$ = ' +  str(round(Cp[1,0],3)))   
    axes.minorticks_on()   

    # Test Case 3
    axes = fig.add_subplot(2,3,3) 
    axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[2,6,:][:len(harmonics)] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,   label = 'SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_3_60deg                                    , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw,   label = 'ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_3_60deg                                     , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,  label = 'Exp.')        
    axes.set_title('60 deg. Case 3, $C_P$ = ' +  str(round(Cp[2,0],3))) 
    axes.minorticks_on() 
    plt.tight_layout()
 
    axes = fig.add_subplot(2,3,4)    
    axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[0,9,:][:len(harmonics)] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,  label = 'SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_1_90deg                                    , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw,   label = 'ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_1_90deg                                     , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,  label = 'Exp.')       
    axes.set_title('90 deg. Case 1, $C_P$ = ' + str(round(Cp[0,0],3)))
    axes.set_ylabel('SPL (dB)')
    axes.set_xlabel('Harmonic #')   
    axes.minorticks_on() 

    axes = fig.add_subplot(2,3,5)              
    axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[1,9,:][:len(harmonics)]  , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_2_90deg                                     , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw, label = 'ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_2_90deg                                      , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw, label = 'Exp.')   
    axes.set_title('90 deg. Case 2, $C_P$ = ' +  str(round(Cp[1,0],3)))   
    axes.set_xlabel('Harmonic #')  
    axes.legend(loc='upper center', prop={'size': PP.lf} , bbox_to_anchor=(0.5, -0.4), ncol= 3 )  
    axes.minorticks_on() 

    axes = fig.add_subplot(2,3,6)    
    axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[2,9,:][:len(harmonics)] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,   label = 'SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_3_90deg                                    , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw,   label = 'ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_3_90deg                                     , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,  label = 'Exp.')     
    axes.set_title('90 deg. Case 3, $C_P$ = ' +  str(round(Cp[2,0],3)))     
    axes.set_xlabel('Harmonic #')  
    axes.minorticks_on() 
    plt.tight_layout()

    # Polar plot of noise   
    fig2 = plt.figure('Polar')
    axis2 = fig2.add_subplot(111, projection='polar')    
    axis2.plot(theta*Units.degrees,F8745D4_SPL[0,:] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'Total'  )  
    axis2.plot(-theta*Units.degrees,F8745D4_SPL[0,:] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw  )  
    axis2.plot(theta*Units.degrees,F8745D4_SPL_harmonic[0,:] , color = PP.Slc[1] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw   )  
    axis2.plot(-theta*Units.degrees,F8745D4_SPL_harmonic[0,:] , color = PP.Slc[1] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'Harmonic'  )  
    axis2.plot(theta*Units.degrees,F8745D4_SPL_broadband[0,:] , color = PP.Slc[2] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw   )  
    axis2.plot(-theta*Units.degrees,F8745D4_SPL_broadband[0,:] , color = PP.Slc[2] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'Broadband' )     
    axis2.set_yticks(np.arange(50,150,25))     
    axis2.grid(True)  

    # Store errors 
    error = Data()
    error.SPL_Case_1_60deg  = np.max(np.abs(F8745D4_SPL_harmonic_bpf_spectrum[0,6,:][:len(harmonics)]  - Exp_Test_Case_1_60deg)/Exp_Test_Case_1_60deg)  
    error.SPL_Case_1_90deg  = np.max(np.abs(F8745D4_SPL_harmonic_bpf_spectrum[0,9,:][:len(harmonics)] - Exp_Test_Case_1_90deg)/Exp_Test_Case_1_90deg)
    
    print('Harmonic Noise Errors:')
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1E0)

    return    
 

# ------------------------------------------------------------------ 
# Broadband Noise Validation
# ------------------------------------------------------------------     
def Broadband_Noise_Validation(PP):  
    APC_SF = APC_11x4_Propeller()   
    APC_SF_inflow_ratio = 0.08 

    # Atmosheric conditions 
    a                     = 343   
    density               = 1.225
    dynamic_viscosity     = 1.78899787e-05   
    T                     = 286.16889478 

    # ---------------------------------------------------------------------------------------------------------------------------
    # APC SF Rotor
    # ---------------------------------------------------------------------------------------------------------------------------
    # Define Network
    net_APC_SF                                  = Battery_Propeller()
    net_APC_SF.number_of_propeller_engines      = 1        
    net_APC_SF.identical_propellers             = True  
    net_APC_SF.propellers.append(APC_SF)    

    # Run conditions                            
    APC_SF_RPM                                  = np.array([3600,4200,4800])
    APC_SF_omega_vector                         = APC_SF_RPM * Units.rpm 
    ctrl_pts                                    = len(APC_SF_omega_vector)   
    velocity                                    = APC_SF_inflow_ratio*APC_SF_omega_vector*APC_SF.tip_radius 
    theta                                       = np.array([45., 67.5, 90.001, 112.5 , 135.])  # np.linspace(0.1,180,100)  
    S                                           = 1.905

    # Microphone Locations 
    positions = np.zeros((len(theta),3))
    for i in range(len(theta)):
        if theta[i]*Units.degrees < np.pi/2:
            positions[i][:] = [-S*np.cos(theta[i]*Units.degrees),-S*np.sin(theta[i]*Units.degrees), 0.0]
        else: 
            positions[i][:] = [S*np.sin(theta[i]*Units.degrees- np.pi/2),-S*np.cos(theta[i]*Units.degrees - np.pi/2), 0.0] 

    # Define conditions 
    APC_SF.thrust_angle                                            = 0. * Units.degrees
    APC_SF.inputs.omega                                            = np.atleast_2d(APC_SF_omega_vector).T
    APC_SF_conditions                                              = Aerodynamics() 
    APC_SF_conditions.freestream.density                           = np.ones((ctrl_pts,1)) * density
    APC_SF_conditions.freestream.dynamic_viscosity                 = np.ones((ctrl_pts,1)) * dynamic_viscosity   
    APC_SF_conditions.freestream.speed_of_sound                    = np.ones((ctrl_pts,1)) * a 
    APC_SF_conditions.freestream.temperature                       = np.ones((ctrl_pts,1)) * T
    v_mat                                                          = np.zeros((ctrl_pts,3))
    v_mat[:,0]                                                     = velocity 
    APC_SF_conditions.frames.inertial.velocity_vector              = v_mat 
    APC_SF_conditions.propulsion.throttle                          = np.ones((ctrl_pts,1)) * 1.0 
    APC_SF_conditions.frames.body.transform_to_inertial            = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])

    # Run Propeller BEMT new model  
    APC_SF_thrust, APC_SF_torque, APC_SF_power, APC_SF_Cp, acoustic_outputs  , APC_SF_etap  =  APC_SF.spin(APC_SF_conditions)  

    # Prepare Inputs for Noise Model  
    APC_SF_conditions.noise.total_microphone_locations             = np.repeat(positions[ np.newaxis,:,: ],ctrl_pts,axis=0)
    APC_SF_conditions.aerodynamics.angle_of_attack                 = np.ones((ctrl_pts,1))* 0. * Units.degrees 
    APC_SF_segment                                                 = Segment() 
    APC_SF_segment.state.conditions                                = APC_SF_conditions
    APC_SF_segment.state.conditions.expand_rows(ctrl_pts)
    APC_SF_settings                                                = Data()
    APC_SF_settings                                                = setup_noise_settings(APC_SF_segment)   

    # Run Noise Model    
    APC_SF_propeller_noise             = propeller_mid_fidelity(net_APC_SF.propellers,acoustic_outputs,APC_SF_segment,APC_SF_settings )    
    APC_SF_1_3_Spectrum                = APC_SF_propeller_noise.SPL_1_3_spectrum 
    APC_SF_SPL_broadband_1_3_spectrum  = APC_SF_propeller_noise.SPL_broadband_1_3_spectrum  

    # ----------------------------------------------------------------------------------------------------------------------------------------
    #  Experimental Data
    # ----------------------------------------------------------------------------------------------------------------------------------------      
    Exp_APC_SF_freqency_spectrum =  np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 
                                              2000, 2500, 3150,4000, 5000, 6300, 8000, 10000])   

    Exp_APC_SF_1_3_Spectrum      = np.array([[22.149, 42.242, 24.252, 22.616, 26.121, 24.953, 28.925, 29.158, 39.205
                                              , 42.943, 39.205, 38.971, 47.149, 45.280, 40.373, 38.738, 38.037, 39.906
                                              , 41.308, 45.981, 42.710, 39.205, 41.775, 37.570, 34.065, 33.598 ],
                                             [17.943,46.214,46.214,22.850,27.056,27.990,31.495,31.261,37.336,
                                              42.242,50.186,40.373,45.280,42.476,45.747,43.878,43.878,48.084,
                                              48.317,49.252,49.018,49.018,46.214,42.242,40.140,39.205 ],
                                             [ 19.345, 18.411, 54.859, 24.018, 26.355, 34.065, 33.130, 33.130, 36.635
                                               , 45.981, 45.046, 40.841, 42.710, 43.411, 44.813, 51.588, 45.981, 46.915
                                              , 52.289, 48.551, 50.186, 48.551, 48.551, 48.317, 43.177, 41.308]])   

    Exp_broadband_APC = np.array([[24.8571428,27.7142857,28.8571428,26.5714285,27.1428571,27.7142857,30.2857142,32,35.4285714,
                                   39.1428571,40.5714285,39.4285714,40.5714285,40,41.1428571,40.2857142,41.7142857,44,44.5714285,
                                   44.8571428,45.1428571],
                                  [23.42857,26,27.14285,25.14285,25.71428,26.57142,28.28571,29.14285,34.28571,37.14285,
                                   37.99999,34.57142,39.14285,34,35.71428,34.85714,35.99999,43.42857,39.14285,41.14285,
                                   42.57142]]) 


    fig_size_width  = 8 
    fig_size_height = 6

    # ----------------------------------------------------------------------------------------------------------------------------------------
    #  Plots  
    # ----------------------------------------------------------------------------------------------------------------------------------------   

    # Figures 8 and 9 comparison 
    fig1 = plt.figure('Noise_Validation_Total_1_3_Spectrum_3600')    
    fig1.set_size_inches(fig_size_width,fig_size_height)  
    axes1 = fig1.add_subplot(1,1,1)      
    axes1.plot(Exp_APC_SF_freqency_spectrum , Exp_APC_SF_1_3_Spectrum[0,:-5]  , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw, label = 'Exp. 3600 RPM')            
    axes1.plot(Exp_APC_SF_freqency_spectrum ,     APC_SF_1_3_Spectrum[0,0,8:]  , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label =' SUAVE 3600 RPM')    
    axes1.set_xscale('log') 
    axes1.set_ylabel(r'SPL$_{1/3}$ (dB)')
    axes1.set_xlabel('Frequency (Hz)') 
    axes1.legend(loc='lower right') 
    axes1.set_ylim([15,60])   


    fig2 = plt.figure('Noise_Validation_1_3_Spectrum_4800')    
    fig2.set_size_inches(fig_size_width,fig_size_height)  
    axes2 = fig2.add_subplot(1,1,1)           
    axes2.plot(Exp_APC_SF_freqency_spectrum , Exp_APC_SF_1_3_Spectrum[2,:-5]  , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,  label = 'Exp. 4800 RPM')       
    axes2.plot(Exp_APC_SF_freqency_spectrum ,     APC_SF_1_3_Spectrum[2,0,8:]  , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,label = ' SUAVE 4800 RPM')   
    axes2.set_xscale('log') 
    axes2.set_ylabel(r'SPL$_{1/3}$ (dB)')
    axes2.set_xlabel('Frequency (Hz)') 
    axes2.legend(loc='lower right')  
    axes2.set_ylim([15,60])   


    fig3 = plt.figure('Noise_Validation_Broadband_1_3_Spectrum_45_deg')    
    fig3.set_size_inches(fig_size_width,fig_size_height)  
    axes3 = fig3.add_subplot(1,1,1)           
    axes3.plot(Exp_APC_SF_freqency_spectrum , Exp_broadband_APC[0,:], color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,    label = 'Exp. 45 $\degree$ mic.')    
    axes3.plot(Exp_APC_SF_freqency_spectrum , APC_SF_SPL_broadband_1_3_spectrum[1,4,8:] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,  label = 'SUAVE 45 $\degree$ mic')     
    axes3.set_xscale('log') 
    axes3.set_ylabel(r'SPL$_{1/3}$ (dB)')
    axes3.set_xlabel('Frequency (Hz)') 
    axes3.legend(loc='lower right')  
    axes3.set_ylim([15,50])   


    fig4 = plt.figure('Noise_Validation_Broadband_1_3_Spectrum_22_deg')    
    fig4.set_size_inches(fig_size_width,fig_size_height)  
    axes4 = fig4.add_subplot(1,1,1)            
    axes4.plot(Exp_APC_SF_freqency_spectrum , Exp_broadband_APC[1,:], color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,   label = 'Exp. 22.5 $\degree$ mic.')     
    axes4.plot(Exp_APC_SF_freqency_spectrum , APC_SF_SPL_broadband_1_3_spectrum[1,3,8:] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,  label = 'SUAVE 22.5 $\degree$ mic.')   
    axes4.set_xscale('log') 
    axes4.set_ylabel(r'SPL$_{1/3}$ (dB)')
    axes4.set_xlabel('Frequency (Hz)') 
    axes4.legend(loc='lower right') 
    axes4.set_ylim([15,50])   


    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()     
    
    
    # Store errors 
    error = Data()
    error.SPL_Case_1_60deg  = np.max(np.abs(APC_SF_SPL_broadband_1_3_spectrum[1,3,8:]  - Exp_broadband_APC[1,:])/Exp_broadband_APC[1,:])
    error.SPL_Case_1_90deg  = np.max(np.abs(APC_SF_SPL_broadband_1_3_spectrum[1,4,8:] - Exp_broadband_APC[0,:])/Exp_broadband_APC[0,:])
    
    print('Broadband Noise Errors:')
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1E0)
        
    return 

def setup_noise_settings(sts): 

    sts.ground_microphone_phi_angles   = np.array([30.,45.,60.,75.,89.9,90.1,105.,120.,135.,150.])*Units.degrees
    sts.ground_microphone_theta_angles = np.array([89.9,89.9,89.9,89.9,89.9,89.9,89.9,89.9, 89.9,89.9 ])*Units.degrees
    sts.center_frequencies             = np.array([16,20,25,31.5,40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, \
                                                   500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                                                   4000, 5000, 6300, 8000, 10000])        
    sts.lower_frequencies              = np.array([14,18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,\
                                                   900,1120,1400,1800,2240,2800,3550,4500,5600,7100,9000 ])
    sts.upper_frequencies              = np.array([18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,900,1120,\
                                                   1400,1800,2240,2800,3550,4500,5600,7100,9000,11200 ])
    sts.harmonics                      = np.arange(1,30)


    sts.broadband_spectrum_resolution        = 301
    sts.floating_point_precision             = np.float32
    sts.urban_canyon_microphone_z_resolution = 16 
    sts.sideline_x_position                  = 0     
    sts.number_of_multiprocessing_workers    = 8
    sts.parallel_computing                   = True # TO BE REMOVED
    sts.lateral_ground_distance              = 1000 * Units.feet  
    sts.level_ground_microphone_min_x        = -50
    sts.level_ground_microphone_max_x        = 1000
    sts.level_ground_microphone_min_y        = -1000 * Units.feet 
    sts.level_ground_microphone_max_y        = 1000 * Units.feet 
    sts.level_ground_microphone_x_resolution = 16 
    sts.level_ground_microphone_y_resolution = 4      
    return sts 

if __name__ == '__main__': 
    main()    
    plt.show()   
    
    