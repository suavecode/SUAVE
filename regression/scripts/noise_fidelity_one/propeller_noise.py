# SUAVE Imports 

# Imports    
import SUAVE
from SUAVE.Core import Units, Data , to_numpy, to_jnumpy
from SUAVE.Components.Energy.Networks.Battery_Propeller                                   import Battery_Propeller 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars  import compute_airfoil_polars
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry     
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity                    import propeller_mid_fidelity
from SUAVE.Analyses.Mission.Segments.Conditions                                           import Aerodynamics , Conditions
from SUAVE.Analyses.Mission.Segments.Segment                                              import Segment 
from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_Zero import Rotor_Wake_Fidelity_Zero
from scipy.interpolate import interp1d  
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_properties \
     import compute_airfoil_properties

import jax.numpy as jnp
import jax
# Python Imports 
import os 
import numpy as np 
import matplotlib.pyplot as plt    
import sys 

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

    net                                = Battery_Propeller()
    net.number_of_propeller_engines    = 1                      
    prop                               = F8745_D4_Propeller()          
    prop.Wake                          = Rotor_Wake_Fidelity_Zero()     
    net.identical_propellers           = True  
    net.propellers.append(prop)  
 

    theta                   = np.array([45])  
    S                       = 4.
    test_omega              = np.array([2390]) * Units.rpm    
    ctrl_pts                = len(test_omega)

    # Set twist target
    three_quarter_twist     = 21 * Units.degrees 
    n                       = len(prop.twist_distribution)
    beta                    = prop.twist_distribution
    beta_75                 = beta[round(n*0.75)] 
    delta_beta              = three_quarter_twist-beta_75
    prop.twist_distribution = beta + delta_beta 


    # Atmosheric & Run Conditions                                               
    a                       = 343.376
    T                       = 288.16889478  
    density                 = 1.2250	
    dynamic_viscosity       = 1.81E-5 


    # Set up for Propeller Model
    conditions                                             = Aerodynamics()   
    conditions.freestream.density                          = jnp.ones((ctrl_pts,1)) * density
    conditions.freestream.dynamic_viscosity                = jnp.ones((ctrl_pts,1)) * dynamic_viscosity   
    conditions.freestream.speed_of_sound                   = jnp.ones((ctrl_pts,1)) * a 
    conditions.freestream.temperature                      = jnp.ones((ctrl_pts,1)) * T 
    conditions.frames.inertial.velocity_vector             = jnp.array([[77.2, 0. ,0.]])
    conditions.propulsion.throttle                         = jnp.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial           = jnp.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])
    prop.inputs.omega                                      = jnp.atleast_2d(test_omega).T
    prop.inputs.y_axis_rotation                            = jnp.ones_like(prop.inputs.omega)
    
    # Run Propeller model 
    F, Q, P, Cp , noise_data , etap                        = to_numpy(prop.spin(conditions))   

    # microphone locations
    positions = np.zeros(( len(theta),3))
    for i in range(len(theta)):
        if theta[i]*Units.degrees < np.pi/2:
            positions[i][:] = [-S*np.cos(theta[i]*Units.degrees)  ,S*np.sin(theta[i]*Units.degrees), 0.0]
        else: 
            positions[i][:] = [S*np.sin(theta[i]*Units.degrees- np.pi/2)  ,S*np.cos(theta[i]*Units.degrees - np.pi/2), 0.0]   
 
    mic_positions_hover = to_jnumpy(positions)     
    
    ## Run noise model  
    conditions.noise.total_microphone_locations      = jnp.repeat(mic_positions_hover[ jnp.newaxis,:,: ],1,axis=0)
    #conditions.aerodynamics.angle_of_attack          = np.ones((ctrl_pts,1))* 0. * Units.degrees 
    #segment                                          = Segment() 
    #segment.state.conditions                         = conditions
    #segment.state.conditions.expand_rows(ctrl_pts)  
    #noise                                            = SUAVE.Analyses.Noise.Fidelity_One() 
    #settings                                         = noise.settings   
    #num_mic                                          = len(conditions.noise.total_microphone_locations[0])  
    #conditions.noise.number_of_microphones           = num_mic    
     
     

    #mic_positions_hover = jnp.array([[0.0 , S_hover*jnp.sin(theta)  ,S_hover*jnp.cos(theta)]])      
    
    # Run noise model  
    #conditions.noise.total_microphone_locations      = jnp.repeat(mic_positions_hover[ jnp.newaxis,:,: ],1,axis=0)
    conditions.aerodynamics.angle_of_attack          = np.ones((ctrl_pts,1))* 0. * Units.degrees 
    segment                                          = Segment() 
    segment.state.conditions                         = conditions
    #segment.state.conditions.expand_rows(ctrl_pts)  
    noise                                            = SUAVE.Analyses.Noise.Fidelity_One() 
    settings                                         = noise.settings   
    num_mic                                          = len(conditions.noise.total_microphone_locations[0])  
    conditions.noise.number_of_microphones           = num_mic     
    
    # Run Fidelity One    
    propeller_noise                       = to_numpy(propeller_mid_fidelity(net.propellers,noise_data,segment,settings ))
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


    # ----------------------------------------------------------------------------------------------------------------------------------------
    #  Plots
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # Universal Plot Settings
    plt.rcParams["font.family"]    = "Times New Roman"
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['axes.linewidth'] = 1. 
 
    PP = Data( 
        lw  = 2,                             # line_width               
        m = 10,                             # markersize               
        lf = 10,                            # legend_font_size         
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
   
    # plot results
    fig31 = plt.figure('Test_Case_3_p1')
    fig31.set_size_inches(16, 5)   
    axes = fig31.add_subplot(1,3,1) 
    axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[0,6,:][:len(harmonics)]   , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_1_60deg                                      , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw, label = 'ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_1_60deg                                       , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,  label = 'Exp.')    
    axes.set_title('Case 1, $C_P$ = ' + str(round(Cp[0,0],3)))
    axes.set_ylabel('SPL (dB)')
    axes.set_xlabel('Harmonic #') 
    axes.minorticks_on() 
    #plt.ylim((80,125))      

    ## Test Case 2
    #axes = fig31.add_subplot(1,3,2) 
    #axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[1,6,:][:len(harmonics)] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,   label = 'SUAVE')    
    #axes.plot(harmonics, ANOPP_PAS_Case_2_60deg                                    , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw,   label = 'ANOPP PAS')       
    #axes.plot(harmonics, Exp_Test_Case_2_60deg                                     , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,   label = 'Exp.')  
    ##axes.set_ylabel('SPL (dB)') 
    #axes.set_title('Case 2, $C_P$ = ' +  str(round(Cp[1,0],3)))  
    #axes.set_xlabel('Harmonic #') 
    #axes.minorticks_on()  
    #axes.legend(loc='upper center', prop={'size':  PP.lf} , bbox_to_anchor=(0.5, -0.2), ncol= 3 )  

    ## Test Case 3
    #axes = fig31.add_subplot(1,3,3) 
    #axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[2,6,:][:len(harmonics)] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,   label = 'SUAVE')    
    #axes.plot(harmonics, ANOPP_PAS_Case_3_60deg                                    , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw,   label = 'ANOPP PAS')       
    #axes.plot(harmonics, Exp_Test_Case_3_60deg                                     , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,  label = 'Exp.')        
    #axes.set_title('Case 3, $C_P$ = ' +  str(round(Cp[2,0],3)))  
    #axes.set_xlabel('Harmonic #')  
    #axes.minorticks_on() 
    #plt.tight_layout()

    fig32 = plt.figure('Test_Case_3_p2') 
    fig32.set_size_inches(16, 5)       
    axes = fig32.add_subplot(1,3,1)    
    axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[0,9,:][:len(harmonics)] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,  label = 'SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_1_90deg                                    , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw,   label = 'ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_1_90deg                                     , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,  label = 'Exp.')       
    axes.set_title('Case 1, $C_P$ = ' + str(round(Cp[0,0],3)))
    axes.set_ylabel('SPL (dB)')
    axes.set_xlabel('Harmonic #')   
    axes.minorticks_on() 

    #axes = fig32.add_subplot(1,3,2)              
    #axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[1,9,:][:len(harmonics)]  , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'SUAVE')    
    #axes.plot(harmonics, ANOPP_PAS_Case_2_90deg                                     , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw, label = 'ANOPP PAS')       
    #axes.plot(harmonics, Exp_Test_Case_2_90deg                                      , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw, label = 'Exp.')   
    #axes.set_title('Case 2, $C_P$ = ' +  str(round(Cp[1,0],3)))   
    #axes.set_xlabel('Harmonic #')  
    #axes.legend(loc='upper center', prop={'size': PP.lf} , bbox_to_anchor=(0.5, -0.2), ncol= 3 )  
    #axes.minorticks_on() 

    #axes = fig32.add_subplot(1,3,3)    
    #axes.plot(harmonics, F8745D4_SPL_harmonic_bpf_spectrum[2,9,:][:len(harmonics)] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw,   label = 'SUAVE')    
    #axes.plot(harmonics, ANOPP_PAS_Case_3_90deg                                    , color = PP.Rlc[0] , linestyle = PP.Rls, marker = PP.Rlm , markersize = PP.m , linewidth = PP.lw,   label = 'ANOPP PAS')       
    #axes.plot(harmonics, Exp_Test_Case_3_90deg                                     , color = PP.Elc[0] , linestyle = PP.Els, marker = PP.Elm , markersize = PP.m , linewidth = PP.lw,  label = 'Exp.')     
    #axes.set_title('Case 3, $C_P$ = ' +  str(round(Cp[2,0],3)))     
    #axes.set_xlabel('Harmonic #')  
    #axes.minorticks_on() 
    #plt.tight_layout()

    # Polar plot of noise   
    fig33 = plt.figure('Test_Case_3_p3')
    axis = fig33.add_subplot(111, projection='polar')    
    axis.plot(theta*Units.degrees,F8745D4_SPL[0,:] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'Total'  )  
    axis.plot(-theta*Units.degrees,F8745D4_SPL[0,:] , color = PP.Slc[0] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw  )  
    axis.plot(theta*Units.degrees,F8745D4_SPL_harmonic[0,:] , color = PP.Slc[1] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw   )  
    axis.plot(-theta*Units.degrees,F8745D4_SPL_harmonic[0,:] , color = PP.Slc[1] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'Harmonic'  )  
    axis.plot(theta*Units.degrees,F8745D4_SPL_broadband[0,:] , color = PP.Slc[2] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw   )  
    axis.plot(-theta*Units.degrees,F8745D4_SPL_broadband[0,:] , color = PP.Slc[2] , linestyle = PP.Sls, marker = PP.Slm , markersize = PP.m , linewidth = PP.lw, label = 'Broadband' )     
    axis.set_yticks(np.arange(50,150,25))     
    axis.grid(True)  

    # Store errors 
    error = Data()
    error.SPL_Case_1_60deg  = np.max(np.abs(F8745D4_SPL_harmonic_bpf_spectrum[0,6,:][:len(harmonics)]  - Exp_Test_Case_1_60deg))  
    error.SPL_Case_1_90deg  = np.max(np.abs(F8745D4_SPL_harmonic_bpf_spectrum[0,9,:][:len(harmonics)] - Exp_Test_Case_1_90deg))
    
    print('Errors:')
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<5E0)

    return


if __name__ == '__main__':  
    with jax.disable_jit():
        main()     
    plt.show()   