## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# VLM.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Apr 2017, T. MacDonald
#           Oct 2017, E. Botero
#           Jun 2018, M. Clarke


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import SUAVE
import numpy as np
from SUAVE.Core import Units
import time
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_vortex_distribution    import compute_vortex_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import  compute_induced_velocity_matrix
from SUAVE.Plots import plot_vehicle_vlm_panelization
# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def VLM(conditions,configuration,geometry):
    ti = time.time()
    """Uses the vortex lattice method to compute the lift, induced drag and moment coefficients  

    Assumptions:
    None

    Source:
    1. Aerodynamics for Engineers, Sixth Edition by John Bertin & Russel Cummings 
    
    2. An Introduction to Theoretical and Computational Aerodynamics by Jack Moran
    
    3. Yahyaoui, M. "Generalized Vortex Lattice Method for Predicting Characteristics of Wings
    with Flap and Aileron Deflection" , World Academy of Science, Engineering and Technology 
    International Journal of Mechanical, Aerospace, Industrial and Mechatronics Engineering 
    Vol:8 No:10, 2014
    

    Inputs:
    geometry.
       wing.
         spans.projected                       [m]
         chords.root                           [m]
         chords.tip                            [m]
         sweeps.quarter_chord                  [radians]
         taper                                 [Unitless]
         twists.root                           [radians]
         twists.tip                            [radians]
         symmetric                             [Boolean]
         aspect_ratio                          [Unitless]
         areas.reference                       [m^2]
         vertical                              [Boolean]
         origin                                [m]
       configuration.number_panels_spanwise    [Unitless]
       configuration.number_panels_chordwise   [Unitless]
       conditions.aerodynamics.angle_of_attack [radians]

    Outputs:
    CL                                      [Unitless]
    Cl                                      [Unitless]
    CDi                                     [Unitless]
    Cdi                                     [Unitless]

    Properties Used:
    N/A
    """ 
   
    # unpack settings
    n_sw   = configuration.number_panels_spanwise    
    n_cw   = configuration.number_panels_chordwise   
    Sref   = geometry.reference_area
    
    # define point about which moment coefficient is computed 
    c_bar  = geometry.wings['main_wing'].chords.mean_aerodynamic
    x_mac  = geometry.wings['main_wing'].aerodynamic_center[0] + geometry.wings['main_wing'].origin[0]
    x_cg   = geometry.mass_properties.center_of_gravity[0] 
    if x_cg == None:
        x_m = x_mac 
    else:
        x_m = x_cg
    
    aoa = conditions.aerodynamics.angle_of_attack[0][0]   # angle of attack    
    
    # Compute vortex distribution
    data = compute_vortex_distribution(geometry,n_sw,n_cw)
    
    # Plot vortex discretization of vehicle
    #plot_vehicle_vlm_panelization(data)
    
    # Build induced velocity matrix, C_mn
    C_mn = compute_induced_velocity_matrix(data,n_sw,n_cw,aoa)

    # Compute flow tangency conditions   
    phi   = np.arctan((data.ZBC - data.ZAC)/(data.YBC - data.YAC))
    delta = np.arctan((data.ZC - data.ZCH)/(data.XC - data.XCH)) 
   
    # Build Aerodynamic Influence Coefficient Matrix
    A = C_mn[:,:,2] - np.multiply(C_mn[:,:,0],np.tan(delta))- np.multiply(C_mn[:,:,1],np.tan(phi))
    
    # Build the vector
    RHS = np.tan(delta)*np.cos(aoa) - np.sin(aoa)
    
    # Compute vortex strength  
    gamma = np.linalg.solve(A,RHS)
    
    # Compute induced velocities     
    u = np.dot(C_mn[:,:,0],gamma)
    v = np.dot(C_mn[:,:,1],gamma)
    w = np.dot(C_mn[:,:,2],gamma)    
    
    # ---------------------------------------------------------------------------------------
    # STEP 10: Compute aerodynamic coefficients 
    # --------------------------------------------------------------------------------------- 
    n_cp= data.n_cp   
    n_cppw     = n_sw*n_cw
    n_w        = data.n_w
    CS         = data.CS    
    wing_areas = data.wing_areas
    X_M        = np.ones(n_cp)*x_m  
    CL_wing    = np.zeros(n_w)
    CDi_wing   = np.zeros(n_w)
    Cl_wing    = np.zeros(n_w*n_sw)
    Cdi_wing   = np.zeros(n_w*n_sw)
    
    Del_Y = np.abs(data.YB1 - data.YA1)
    i = 0
    for j in range(n_w):
        L_wing      = np.dot((u[j*n_cppw:(j+1)*n_cppw] +1),gamma[j*n_cppw:(j+1)*n_cppw] * Del_Y[j*n_cppw:(j+1)*n_cppw]) # wing lift coefficient
        CL_wing[j]  = L_wing/(wing_areas[j])
        Di_wing     = np.dot(-w[j*n_cppw:(j+1)*n_cppw]    ,gamma[j*n_cppw:(j+1)*n_cppw] * Del_Y[j*n_cppw:(j+1)*n_cppw]) # wing induced drag coefficient
        CDi_wing[j] = Di_wing/(wing_areas[j])
        for k in range(n_sw):   
            l_wing      = np.dot((u[i*n_cw:(i+1)*n_cw] +1),gamma[i*n_cw:(i+1)*n_cw] * Del_Y[i*n_cw:(i+1)*n_cw]) # sectional lift coefficients 
            Cl_wing[i]  = 2*l_wing/(CS[i])
            di_wing     = np.dot(-w[i*n_cw:(i+1)*n_cw]  ,gamma[i*n_cw:(i+1)*n_cw] * Del_Y[i*n_cw:(i+1)*n_cw])   # sectional induced drag coefficients
            Cdi_wing[i] = 2*di_wing/(CS[i])
            i += 1
            
    # total lift and lift coefficient
    L  = np.dot((1+u),gamma*Del_Y)
    CL = 2*L/(Sref) 
    
    # total drag and drag coefficient
    D  =  -np.dot(w,gamma*Del_Y)
    CDi = 2*D/(np.pi*Sref) 
    
    # moment coefficient
    CM  = np.dot((X_M - data.XCH),Del_Y*gamma)/(Sref*c_bar)   
     
    tf = time.time()
    print ('Time taken for VLM: ' + str(tf-ti))      
    return CL, Cl_wing, CDi, Cdi_wing, CM 

