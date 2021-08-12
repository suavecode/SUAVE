# SUAVE Imports 

# Imports    
import SUAVE
from SUAVE.Core import Units, Data 
from SUAVE.Components.Energy.Networks.Battery_Propeller                                   import Battery_Propeller 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars  import compute_airfoil_polars
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity                    import propeller_mid_fidelity
from SUAVE.Analyses.Mission.Segments.Conditions                                           import Aerodynamics , Conditions
from SUAVE.Analyses.Mission.Segments.Segment                                              import Segment 
from scipy.interpolate import interp1d  

# Python Imports 
import os 
import numpy as np 
import matplotlib.pyplot as plt    
# ----------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------- 
def main(): 
    '''This regression script is for validation and verification of the mid-fidelity acoustics
    analysis routine. "Experimental Data is obtained from Comparisons of predicted propeller
    noise with windtunnel ..." by Weir, D and Powers, J.
    '''   

    # Define Network 
    net                                = Battery_Propeller()     
    net.number_of_propeller_engines    = 1
    net.identical_propellers           = True
    prop                               = design_F8745D4_prop()
    net.propellers.append(prop)

 
    # Set-up Validation Conditions 
    a                       = 343.376
    T                       = 288.16889478  
    density                 = 1.2250	
    dynamic_viscosity       = 1.81E-5  
    theta                   = np.linspace(-np.pi/2,np.pi/2,19) + 1E-8
    S                       = np.array([4]) # np.linspace(2,10,5) 
    omega                   = np.array([2390,2710,2630]) * Units.rpm     

    # Set twist target
    three_quarter_twist     = 21 * Units.degrees 
    n                       = len(prop.twist_distribution)
    beta                    = prop.twist_distribution
    beta_75                 = beta[round(n*0.75)] 
    delta_beta              = three_quarter_twist-beta_75
    prop.twist_distribution = beta + delta_beta 

    # microphone locations
    ctrl_pts                = len(omega) 
    dim_theta               = len(theta)
    dim_S                   = len(S) 
    num_mic                 = dim_S*dim_theta
                            
    theta                   = np.repeat(np.repeat(np.atleast_2d(theta).T ,dim_S , axis = 1)[np.newaxis,:,:],ctrl_pts, axis = 0)  
    S                       = np.repeat(np.repeat(np.atleast_2d(S)       ,dim_theta, axis = 0)[np.newaxis,:,:],ctrl_pts, axis = 0) 
    x_vals                  = S*np.sin(theta)
    y_vals                  = -S*np.cos(theta)
    z_vals                  = np.zeros_like(x_vals) 
                           
    mic_locations           = np.zeros((ctrl_pts,num_mic,3))   
    mic_locations[:,:,0]    = x_vals.reshape(ctrl_pts,num_mic) 
    mic_locations[:,:,1]    = y_vals.reshape(ctrl_pts,num_mic) 
    mic_locations[:,:,2]    = z_vals.reshape(ctrl_pts,num_mic)     
    
    # Set up for Propeller Model
    prop.inputs.omega                            = np.atleast_2d(omega).T
    prop.inputs.pitch_command                    = 0.
    conditions                                   = Aerodynamics()
    conditions._size                             = 3
    conditions.freestream.density                = np.ones((ctrl_pts,1)) * density
    conditions.freestream.dynamic_viscosity      = np.ones((ctrl_pts,1)) * dynamic_viscosity   
    conditions.freestream.speed_of_sound         = np.ones((ctrl_pts,1)) * a 
    conditions.freestream.temperature            = np.ones((ctrl_pts,1)) * T 
    conditions.frames.inertial.velocity_vector   = np.array([[77.2, 0. ,0.],[ 77.0,0.,0.], [ 77.2, 0. ,0.]])
    conditions.propulsion.throttle               = np.ones((ctrl_pts,1))*1.0
    conditions.aerodynamics.angle_of_attack      = np.ones((ctrl_pts,1))* 0. * Units.degrees 
    conditions.frames.body.transform_to_inertial = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])
    conditions.noise.microphone_locations        = mic_locations
    
    # Run Propeller model 
    F, Q, P, Cp , noise_data , etap                        = prop.spin(conditions)

    noise_datas = Data()
    noise_datas.propeller = noise_data
    
    # Store Noise Data 
    noise                                                  = SUAVE.Analyses.Noise.Fidelity_One()
    segment                                                = Segment()
    settings                                               = noise.settings
    conditions.noise.sources.propeller                     = noise_datas
    conditions.noise.number_of_microphones                 = num_mic  
    segment.state.conditions                               = conditions

    # Run Fidelity One   
    propeller_noise  = propeller_mid_fidelity(net,noise_datas,segment,settings)  
    SPL_dBA          = propeller_noise.SPL_dBA
    SPL_Spectrum     = propeller_noise.SPL_bpfs_spectrum
    
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
    SUAVE_SPL_Case_1_60deg = SPL_Spectrum[0,6,:][:len(harmonics)]
    SUAVE_SPL_Case_1_90deg = SPL_Spectrum[0,9,:][:len(harmonics)] 
    SUAVE_SPL_Case_2_60deg = SPL_Spectrum[1,6,:][:len(harmonics)]  
    SUAVE_SPL_Case_2_90deg = SPL_Spectrum[1,9,:][:len(harmonics)]
    SUAVE_SPL_Case_3_60deg = SPL_Spectrum[2,6,:][:len(harmonics)] 
    SUAVE_SPL_Case_3_90deg = SPL_Spectrum[2,9,:][:len(harmonics)]  
    
    # plot results
    fig = plt.figure('Validation Case')
    fig.set_size_inches(16, 6)   
    axes = fig.add_subplot(2,3,1) 
    axes.plot(harmonics, SUAVE_SPL_Case_1_60deg,'k^' ,  label = r'60 $\degree$ mic.SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_1_60deg,'bo' ,  label = r'60 $\degree$ mic. ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_1_60deg ,'rs' ,  label = r'60 $\degree$ mic. Exp.')    
    axes.set_title('Case 1, Power Coefficient: ' + str(round(Cp[0,0],3)))
    axes.set_ylabel('SPL (dB)')
    axes.set_xlabel('harmonic ') 
    axes.legend(loc='upper right')  

    axes = fig.add_subplot(2,3,4)    
    axes.plot(harmonics, SUAVE_SPL_Case_1_90deg,'k^' ,  label = r'90 $\degree$ mic.SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_1_90deg,'bo' ,  label = r'90 $\degree$ mic. ANOPP PAS')       
    axes.plot(harmonics, Exp_Test_Case_1_90deg ,'rs' ,  label = r'90 $\degree$ mic. Exp.')    
    axes.set_ylabel('SPL (dB)')
    axes.set_xlabel('harmonic ')  
    axes.legend(loc='upper right') 

    # Test Case 2
    axes = fig.add_subplot(2,3,2) 
    axes.plot(harmonics, SUAVE_SPL_Case_2_60deg,'k^' ,  label = r'60 $\degree$ mic.SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_2_60deg,'bo' ,  label = r'60 $\degree$ mic. ANOPP PAS')        
    axes.plot(harmonics, Exp_Test_Case_2_60deg ,'rs' ,  label = r'60 $\degree$ mic. Exp.')    
    axes.set_title('Case 2, Power Coefficient: ' +  str(round(Cp[1,0],3)))
    axes.set_ylabel('SPL (dB)')                
    axes.set_xlabel('harmonic ')
    axes.legend(loc='upper right') 

    axes = fig.add_subplot(2,3,5)              
    axes.plot(harmonics, SUAVE_SPL_Case_2_90deg,'k^' ,  label = r'90 $\degree$ mic.SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_2_90deg,'bo' ,  label = r'90 $\degree$ mic. ANOPP PAS')  
    axes.plot(harmonics, Exp_Test_Case_2_90deg ,'rs' ,  label = r'90 $\degree$ mic. Exp.')    
    axes.set_ylabel('SPL (dB)')                 
    axes.set_xlabel('harmonic ')  
    axes.legend(loc='upper right') 

    # Test Case 3
    axes = fig.add_subplot(2,3,3) 
    axes.plot(harmonics, SUAVE_SPL_Case_3_60deg,'k^',  label = r'60 $\degree$ mic.SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_3_60deg,'bo',  label = r'60 $\degree$ mic. ANOPP PAS')      
    axes.plot(harmonics, Exp_Test_Case_3_60deg ,'rs',  label = r'60 $\degree$ mic. Exp.')    
    axes.set_title('Case 3') 
    axes.set_title('Case 3, Power Coefficient: ' +  str(round(Cp[2,0],3)))
    axes.set_ylabel('SPL (dB)')
    axes.set_xlabel('harmonic ') 
    axes.legend(loc='upper right') 

    axes = fig.add_subplot(2,3,6)    
    axes.plot(harmonics, SUAVE_SPL_Case_3_90deg,'k^' ,  label = r'90 $\degree$ mic.SUAVE')    
    axes.plot(harmonics, ANOPP_PAS_Case_3_90deg,'bo' ,  label = r'90 $\degree$ mic. ANOPP PAS')  
    axes.plot(harmonics, Exp_Test_Case_3_90deg ,'rs' ,  label = r'90 $\degree$ mic. Exp.')    
    axes.set_ylabel('SPL (dB)')
    axes.set_xlabel('harmonic ') 
    axes.legend(loc='upper right')   
    
    # Polar plot of noise 
    theta      = np.linspace(-np.pi/2,np.pi/2,19)   
    fig, axis  = plt.subplots(subplot_kw={'projection': 'polar'})
    axis.plot(theta, SPL_dBA[0])
    axis.set_ylim([50,150])
    axis.set_yticks(np.arange(50,150,25))     
    axis.grid(True) 
    
    # ----------------------------------------------------------------------------------------------------------------------------------------
    #  Regression 
    # ----------------------------------------------------------------------------------------------------------------------------------------     
    SPL_Case_1_60deg_truth = Exp_Test_Case_1_60deg
    SPL_Case_1_90deg_truth = Exp_Test_Case_1_90deg
    
    # Store errors 
    error = Data()
    error.SPL_Case_1_60deg  = np.max(np.abs(SUAVE_SPL_Case_1_60deg -SPL_Case_1_60deg_truth))  
    error.SPL_Case_1_90deg  = np.max(np.abs(SUAVE_SPL_Case_1_90deg -SPL_Case_1_90deg_truth))
    
    print('Errors:')
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<5E0)

    return    
 
                                        
def design_F8745D4_prop():  
    prop                            = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                        = 'F8745_D4_Propeller'  
    prop.tip_radius                 = 2.03/2
    prop.hub_radius                 = prop.tip_radius*0.20
    prop.number_of_blades           = 2  
    r_R_data                        = np.array([ 0.2,0.300,0.450,0.601,0.747,0.901,0.950,0.975,0.998   ])    
    t_c_data                        = np.array([ 0.3585,0.1976,0.1148,0.0834,0.0648,0.0591,0.0562,0.0542,0.0533    ])    
    b_R_data                        = np.array([0.116,0.143,0.163,0.169,0.166,0.148,0.135,0.113,0.075  ])    
    beta_data                       = np.array([  0.362,0.286,0.216,0.170,0.135,0.112,0.105,0.101,0.098  ])* 100 
  
    dim                             = 30          
    new_radius_distribution         = np.linspace(0.2,0.98 ,dim)
    func_twist_distribution         = interp1d(r_R_data, (beta_data)* Units.degrees   , kind='cubic')
    func_chord_distribution         = interp1d(r_R_data, b_R_data * prop.tip_radius , kind='cubic')
    func_radius_distribution        = interp1d(r_R_data, r_R_data * prop.tip_radius , kind='cubic')
    func_max_thickness_distribution = interp1d(r_R_data, t_c_data * b_R_data , kind='cubic')  
    
    prop.twist_distribution         = func_twist_distribution(new_radius_distribution)     
    prop.chord_distribution         = func_chord_distribution(new_radius_distribution)         
    prop.radius_distribution        = func_radius_distribution(new_radius_distribution)        
    prop.max_thickness_distribution = func_max_thickness_distribution(new_radius_distribution) 
    prop.thickness_to_chord         = prop.max_thickness_distribution/prop.chord_distribution  
    
    ospath                          = os.path.abspath(__file__)
    separator                       = os.path.sep
    rel_path                        = ospath.split('noise_fidelity_one' + separator + 'propeller_noise.py')[0] + 'Vehicles/Airfoils' + separator
    prop.airfoil_geometry           = [ rel_path +'Clark_y.txt']
    prop.airfoil_polars             = [[rel_path +'Polars/Clark_y_polar_Re_50000.txt' ,rel_path +'Polars/Clark_y_polar_Re_100000.txt',rel_path +'Polars/Clark_y_polar_Re_200000.txt',
                                        rel_path +'Polars/Clark_y_polar_Re_500000.txt',rel_path +'Polars/Clark_y_polar_Re_1000000.txt']]
    prop.airfoil_polar_stations     = list(np.zeros(dim))  
    airfoil_polars                  = compute_airfoil_polars(prop.airfoil_geometry, prop.airfoil_polars)  
    airfoil_cl_surs                 = airfoil_polars.lift_coefficient_surrogates 
    airfoil_cd_surs                 = airfoil_polars.drag_coefficient_surrogates         
    prop.airfoil_cl_surrogates      = airfoil_cl_surs
    prop.airfoil_cd_surrogates      = airfoil_cd_surs    
    prop.mid_chord_aligment         = np.zeros_like(prop.chord_distribution) #  prop.chord_distribution/4. - prop.chord_distribution[0]/4.    
    
    return prop
if __name__ == '__main__': 
    main()    
    plt.show()   