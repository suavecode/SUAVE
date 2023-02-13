## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# update_wake_position_time_marching.py
# 
# Created:  Aug 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np 
from DCode.Common.generalFunctions import save_single_prop_vehicle_vtk
from SUAVE.Core import Data
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_interpolated_velocity_field import compute_interpolated_velocity_field
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_exact_velocity_field import compute_exact_velocity_field
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_wake_induced_velocity import compute_wake_induced_velocity
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_fidelity_one_inflow_velocities import compute_fidelity_one_inflow_velocities
import copy
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations \
     import compute_airfoil_aerodynamics,compute_inflow_and_tip_loss

import time
from DCode.Common.Visualization_Tools.box_contour_field_vtk import box_contour_field_vtk
#from DCode.Common.Visualization_Tools.evaluate_induced_velocity_contour_plane import evaluate_induced_velocity_contour_plane, default_prop_contourPlane

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def update_wake_position_time_marching(wake, rotor, conditions):  
    """ Evolves the rotor wake's vortex filament positions, accounting for the wake vortex
    and external vortex components.
    
    This is an approximation method intended to account for first order effects of interactions that
    will change the rotor wake shape. 

    Assumptions:  
    
    Source:   
    
    Inputs: 
    WD     - rotor wake vortex distribution                           [Unitless] 
    VD     - external vortex distribution (ie. from lifting surfaces) [Unitless] 
    cpts   - control points in segment                            [Unitless] 

    Properties Used:
    N/A
    """  
    # Unpack
    WD = wake.vortex_distribution
    VD = wake.external_vortex_distribution
    nts = len(WD.reshaped_wake.XA2[0,0,0,0,:])
    Na  = np.shape(WD.reshaped_wake.XA2)[0]
    omega = rotor.inputs.omega[0][0]
    dt = (2 * np.pi / Na) / omega
    ctrl_pts = 1
    
    #WD_network=wake.influencing_rotor_wake_network
    
    #if WD_network == None:
        ## Using single rotor wake as the netowrk
        #print("No network specified. Using single rotor wake in evolution of wake shape for "+str(wake.tag))
        #WD_network = Data()
        #WD_network[wake.tag + "_vortex_distribution"]  = WD
          
        
    new_wake = generate_temp_wake(WD) # Shape: ( azimuthal start index, control point , blade number , radial location on blade , time step )
    
    use_exact_velocities = True #True#False
    debug = False    
    step_1_times = np.zeros(nts)
    step_2_times = np.zeros(nts)
    
    #--------------------------------------------------------------------------------------------    
    # Step 1: Time march until the wake is fully developed
    #--------------------------------------------------------------------------------------------    
    fully_develop_wake(wake, WD, VD, new_wake, use_exact_velocities,rotor, conditions, step_1_times, debug, Na, nts, step_2_times, ctrl_pts, dt)
           

    #--------------------------------------------------------------------------------------------    
    # Step 2: Iterate through one full rotation after full development is reached
    #--------------------------------------------------------------------------------------------                
    # Step through Na more times and save each time step to vortex distribution data structure
    wakeTemp = copy.deepcopy(WD)
    for a in range(Na):
        single_step_time_march(wake, WD, VD, new_wake, use_exact_velocities,rotor, conditions, step_1_times, a, debug, Na, nts, step_2_times, ctrl_pts, dt)
        for key in list(WD.keys()):
            if key == 'reshaped_wake':
                for key2 in WD[key].keys():
                    if np.shape(WD[key][key2]) == np.shape(WD[key]['XA1']):
                        wakeTemp[key][key2][a,:,:,:,:] = new_wake[key][key2]
            elif np.shape(WD[key]) == np.shape(WD.XA1):
                wakeTemp[key][a,:,:] = new_wake[key]
            
    rotor.Wake.vortex_distribution = wakeTemp
    wake.vortex_distribution = wakeTemp
    
    if True:#debug:
        for a in range(Na):
            rotor.start_angle_idx = a
            save_single_prop_vehicle_vtk(rotor, time_step=a, save_loc="/Users/rerha/Desktop/test_relaxed_wake_2/")         
    
    #--------------------------------------------------------------------------------------------    
    # Step 6: Update the position of all wake panels 
    #--------------------------------------------------------------------------------------------
    import pylab as plt
    from SUAVE.Core import Units
    alpha = rotor.orientation_euler_angles[1]/Units.deg
    plt.plot(np.linspace(0,nts-1,nts), step_2_times,label="Step 2")
    plt.plot(np.linspace(0,nts-1,nts), step_1_times,label="Step 1")
    plt.xlabel('Time Step')
    plt.ylabel('Process Runtime (sec)')
    plt.title("Time Marching Free Wake Execution Times (Alpha = "+str(alpha)+")")
    plt.legend()
    plt.grid()
    plt.savefig('/Users/rerha/Downloads/fvw_tm_runtimes_parallelized_Alpha'+str(alpha)+'.png', dpi=300)
    interpolatedBoxData = None
    return wake, rotor, interpolatedBoxData

def single_step_time_march(wake, WD, VD, new_wake, use_exact_velocities, rotor, conditions, step_1_times, i, debug, Na, nts, step_2_times, ctrl_pts, dt):
    if use_exact_velocities:
        fun_V_induced = None 
    else:
        new_wake_reduced = reduce_wake(new_wake,wake_treatment=False) # only consisting of panels that have already been shed
        WD_network = Data()
        WD_network[wake.tag + "_vortex_distribution"] = new_wake_reduced
        t0 = time.time()
        fun_V_induced, interpolatedBoxData = compute_interpolated_velocity_field(WD_network, rotor, conditions, VD)        
        step_1_times[i] = time.time()-t0
        print("\tStep 1 (compute_interpolated_velocity_field): " + str(step_1_times[i]))
        # debug
        if debug:    
            stateData = Data()
            stateData.vFreestream = 20
            stateData.alphaDeg = 0
            box_contour_field_vtk(interpolatedBoxData, stateData, filename="/Users/rerha/Desktop/test_relaxed_wake_2/time_marching_contour.vtk", iteration=i)   
    
    # Remove oldest row of panels
    remove_oldest_panels(new_wake)
    
    # Advance all panels forward one time step
    time_index = nts-2 # advance all but first panel forward (will be replaced by shed row)
    advance_panels(wake, rotor, conditions, VD, new_wake, time_index, nts, dt, fun_V_induced, use_exact_velocities )
    
    # Shed a row of trailing edge panels for this time step  
    na_start = (i % Na)   
    shed_trailing_edge_panels(WD, new_wake, nts, time_index, na_start, Na)  
    
    # Compute new circulation strengths based on prior wake
    Gamma = compute_panel_circulations(wake, rotor, conditions, nts, new_wake, na_start, ctrl_pts, i)
    new_wake.reshaped_wake.GAMMA[:,:,:,:,:] = Gamma[na_start:na_start+1,:,:,:,:]#[:,:,:,:,:] = Gamma[na_start:na_start+1,:,:,:,:]  
    new_wake.GAMMA = np.reshape(new_wake.reshaped_wake.GAMMA, np.shape(new_wake.GAMMA))
    
    wakeTemp = copy.deepcopy(wake)
    for key in ['XA1', 'XA2', 'XB1', 'XB2','YA1', 'YA2', 'YB1', 'YB2','ZA1', 'ZA2', 'ZB1', 'ZB2', 'GAMMA']:
        if i != nts-1:
            wakeTemp.vortex_distribution.reshaped_wake[key] = new_wake.reshaped_wake[key][:,:,:,:,nts-2-time_index:]
            sz = np.append(np.append(np.shape(new_wake.reshaped_wake.XA2[:,0,0,0,0]), 1), np.size(new_wake.reshaped_wake.XA2[0,:,:,:,nts-2-time_index:]))
            wakeTemp.vortex_distribution[key] = np.reshape(new_wake.reshaped_wake[key][:,:,:,:,nts-2-time_index:], sz)
        else:
            wakeTemp.vortex_distribution.reshaped_wake[key] = new_wake.reshaped_wake[key]
            wakeTemp.vortex_distribution[key] = new_wake[key]   
    rotor.Wake = wakeTemp
    wake = wakeTemp    

    if debug:
        save_single_prop_vehicle_vtk(rotor, time_step=i, save_loc="/Users/rerha/Desktop/test_relaxed_wake_2/")      
    return

def fully_develop_wake(wake, WD, VD, new_wake, use_exact_velocities, rotor, conditions, step_1_times, debug, Na, nts, step_2_times, ctrl_pts, dt):
    # Loop over all time steps until fully developed wake
    for i in range(nts):
        
        print("\nTime step "+str(i)+" out of " + str(nts-1))    
        
        if i == nts-1:
            # Last time step, just advance it forward one step to final wake position
            single_step_time_march(wake, WD, VD, new_wake, use_exact_velocities, rotor, conditions, step_1_times, i, debug, Na, nts, step_2_times, ctrl_pts, dt)
        
        else:
            #--------------------------------------------------------------------------------------------  
            # Step 1: Compute induced velocity field function, Vind = fun((x,y,z))
            #--------------------------------------------------------------------------------------------  
            if use_exact_velocities:
                fun_V_induced = None 
            else:
                new_wake_reduced = reduce_wake(new_wake,wake_treatment=False) # only consisting of panels that have already been shed
                WD_network = Data()
                WD_network[wake.tag + "_vortex_distribution"] = new_wake_reduced
                t0 = time.time()
                fun_V_induced, interpolatedBoxData = compute_interpolated_velocity_field(WD_network, rotor, conditions, VD)        
                step_1_times[i] = time.time()-t0
                print("\tStep 1 (compute_interpolated_velocity_field): " + str(step_1_times[i]))
                # debug
                if debug:    
                    stateData = Data()
                    stateData.vFreestream = 20
                    stateData.alphaDeg = 0
                    box_contour_field_vtk(interpolatedBoxData, stateData, filename="/Users/rerha/Desktop/test_relaxed_wake_2/time_marching_contour.vtk", iteration=i)   
                      
            t0 = time.time()
            # -------------------------------------------------------------------------------------------------------------------------
            # Step 2: Advance all previously shed wake elements one step forward in time
            # -------------------------------------------------------------------------------------------------------------------------        
            if i!=0:
                advance_panels(wake, rotor, conditions, VD, new_wake, i, nts, dt, fun_V_induced, use_exact_velocities )
        
            # -------------------------------------------------------------------------------------------------------------------------
            # Step 3: Shed a row of trailing edge panels for this time step
            # -------------------------------------------------------------------------------------------------------------------------  
            na_start = (i % Na)     
            shed_trailing_edge_panels(WD, new_wake, nts, i, na_start, Na)
    
            # -------------------------------------------------------------------------------------------------------------------------
            # Step 4: Compute new circulation strengths based on prior wake
            # -------------------------------------------------------------------------------------------------------------------------          
            Gamma = compute_panel_circulations(wake, rotor, conditions, nts, new_wake, na_start, ctrl_pts, i)
            new_wake.reshaped_wake.GAMMA[:,:,:,:,nts-1-i] = Gamma[na_start:na_start+1,:,:,:,0]  
            new_wake.GAMMA = np.reshape(new_wake.reshaped_wake.GAMMA, np.shape(new_wake.GAMMA))
            # -------------------------------------------------------------------------------------------
            # ----DEBUG----------------------------------------------------------------------------------        
            # -------------------------------------------------------------------------------------------
            # Step 5: Store time step vortex distribution data
            # -------------------------------------------------------------------------------------------
            # Compress Data into 1D Arrays
            reshaped_shape = np.shape(new_wake.reshaped_wake.XA1)
            m  = reshaped_shape[1]
            B  = reshaped_shape[2]
            Nr = reshaped_shape[3]
            mat6_size = (1,m,nts*B*(Nr)) 
            
            new_wake.XA1    =  np.reshape(new_wake.reshaped_wake.XA1,mat6_size)
            new_wake.YA1    =  np.reshape(new_wake.reshaped_wake.YA1,mat6_size)
            new_wake.ZA1    =  np.reshape(new_wake.reshaped_wake.ZA1,mat6_size)
            new_wake.XA2    =  np.reshape(new_wake.reshaped_wake.XA2,mat6_size)
            new_wake.YA2    =  np.reshape(new_wake.reshaped_wake.YA2,mat6_size)
            new_wake.ZA2    =  np.reshape(new_wake.reshaped_wake.ZA2,mat6_size)
            new_wake.XB1    =  np.reshape(new_wake.reshaped_wake.XB1,mat6_size)
            new_wake.YB1    =  np.reshape(new_wake.reshaped_wake.YB1,mat6_size)
            new_wake.ZB1    =  np.reshape(new_wake.reshaped_wake.ZB1,mat6_size)
            new_wake.XB2    =  np.reshape(new_wake.reshaped_wake.XB2,mat6_size)
            new_wake.YB2    =  np.reshape(new_wake.reshaped_wake.YB2,mat6_size)
            new_wake.ZB2    =  np.reshape(new_wake.reshaped_wake.ZB2,mat6_size)
            new_wake.GAMMA  =  np.reshape(new_wake.reshaped_wake.GAMMA,mat6_size)
            
            wakeTemp = copy.deepcopy(wake)
            for key in ['XA1', 'XA2', 'XB1', 'XB2','YA1', 'YA2', 'YB1', 'YB2','ZA1', 'ZA2', 'ZB1', 'ZB2', 'GAMMA']:
                if i != nts-1:
                    wakeTemp.vortex_distribution.reshaped_wake[key] = new_wake.reshaped_wake[key][:,:,:,:,nts-2-i:]
                    sz = np.append(np.append(np.shape(new_wake.reshaped_wake.XA2[:,0,0,0,0]), 1), np.size(new_wake.reshaped_wake.XA2[0,:,:,:,nts-2-i:]))
                    wakeTemp.vortex_distribution[key] = np.reshape(new_wake.reshaped_wake[key][:,:,:,:,nts-2-i:], sz)
                else:
                    wakeTemp.vortex_distribution.reshaped_wake[key] = new_wake.reshaped_wake[key]
                    wakeTemp.vortex_distribution[key] = new_wake[key]   
                    
            rotor.Wake = wakeTemp
            wake = wakeTemp
            
            step_2_times[i] = time.time()-t0
            print("\tStep 2 (updating positions): " + str(step_2_times[i]))
            if debug:
                save_single_prop_vehicle_vtk(rotor, time_step=i, save_loc="/Users/rerha/Desktop/test_relaxed_wake_2/")  
            
        
                ## xz plane contour output
                #xmin = interpolatedBoxData.Dimensions.Xmin
                #xmax = interpolatedBoxData.Dimensions.Xmax
                #zmin = interpolatedBoxData.Dimensions.Zmin
                #zmax = interpolatedBoxData.Dimensions.Zmax
                #contourPlane = default_prop_contourPlane(wMin=xmin, wMax=xmax, hMin=zmin, hMax=zmax)
                #evaluate_induced_velocity_contour_plane(WD, stateData, contourPlane, iteration=i, save_loc="/Users/rerha/Desktop/test_relaxed_wake/")
        
            # -------------------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------------------
        
        
    return

def remove_oldest_panels(new_wake):   
    
    # last row; remove oldest row of panels and shift others down
    new_wake.reshaped_wake.XA1[:,:,:,:,1:] = new_wake.reshaped_wake.XA1[:,:,:,:,:-1]
    new_wake.reshaped_wake.YA1[:,:,:,:,1:] = new_wake.reshaped_wake.YA1[:,:,:,:,:-1]
    new_wake.reshaped_wake.ZA1[:,:,:,:,1:] = new_wake.reshaped_wake.ZA1[:,:,:,:,:-1]
    new_wake.reshaped_wake.XB1[:,:,:,:,1:] = new_wake.reshaped_wake.XB1[:,:,:,:,:-1]
    new_wake.reshaped_wake.YB1[:,:,:,:,1:] = new_wake.reshaped_wake.YB1[:,:,:,:,:-1]
    new_wake.reshaped_wake.ZB1[:,:,:,:,1:] = new_wake.reshaped_wake.ZB1[:,:,:,:,:-1]
    new_wake.reshaped_wake.XA2[:,:,:,:,1:] = new_wake.reshaped_wake.XA2[:,:,:,:,:-1]
    new_wake.reshaped_wake.YA2[:,:,:,:,1:] = new_wake.reshaped_wake.YA2[:,:,:,:,:-1]
    new_wake.reshaped_wake.ZA2[:,:,:,:,1:] = new_wake.reshaped_wake.ZA2[:,:,:,:,:-1]
    new_wake.reshaped_wake.XB2[:,:,:,:,1:] = new_wake.reshaped_wake.XB2[:,:,:,:,:-1]
    new_wake.reshaped_wake.YB2[:,:,:,:,1:] = new_wake.reshaped_wake.YB2[:,:,:,:,:-1]
    new_wake.reshaped_wake.ZB2[:,:,:,:,1:] = new_wake.reshaped_wake.ZB2[:,:,:,:,:-1]             

    return    
def shed_trailing_edge_panels(WD, new_wake, nts, time_index, na_start, Na):

    # use original wake panel for first row
    x_a1 = WD.reshaped_wake.XA1[na_start:na_start+1,:,:,:,1]
    y_a1 = WD.reshaped_wake.YA1[na_start:na_start+1,:,:,:,1]
    z_a1 = WD.reshaped_wake.ZA1[na_start:na_start+1,:,:,:,1]
    x_b1 = WD.reshaped_wake.XB1[na_start:na_start+1,:,:,:,1]
    y_b1 = WD.reshaped_wake.YB1[na_start:na_start+1,:,:,:,1]
    z_b1 = WD.reshaped_wake.ZB1[na_start:na_start+1,:,:,:,1]

    # pre-pend row to new wake
    new_wake.reshaped_wake.XA1[:,:,:,:,nts-1-time_index] = x_a1
    new_wake.reshaped_wake.YA1[:,:,:,:,nts-1-time_index] = y_a1
    new_wake.reshaped_wake.ZA1[:,:,:,:,nts-1-time_index] = z_a1
    new_wake.reshaped_wake.XB1[:,:,:,:,nts-1-time_index] = x_b1
    new_wake.reshaped_wake.YB1[:,:,:,:,nts-1-time_index] = y_b1
    new_wake.reshaped_wake.ZB1[:,:,:,:,nts-1-time_index] = z_b1
    
    x_a1_start = WD.reshaped_wake.XA1[na_start:na_start+1,:,:,:,0]
    y_a1_start = WD.reshaped_wake.YA1[na_start:na_start+1,:,:,:,0]
    z_a1_start = WD.reshaped_wake.ZA1[na_start:na_start+1,:,:,:,0]
    x_b1_start = WD.reshaped_wake.XB1[na_start:na_start+1,:,:,:,0]
    y_b1_start = WD.reshaped_wake.YB1[na_start:na_start+1,:,:,:,0]
    z_b1_start = WD.reshaped_wake.ZB1[na_start:na_start+1,:,:,:,0]
    
    # get prior row of shed trailing edge nodes
    x_a2_start = WD.reshaped_wake.XA2[na_start:na_start+1,:,:,:,0]
    y_a2_start = WD.reshaped_wake.YA2[na_start:na_start+1,:,:,:,0]
    z_a2_start = WD.reshaped_wake.ZA2[na_start:na_start+1,:,:,:,0]
    x_b2_start = WD.reshaped_wake.XB2[na_start:na_start+1,:,:,:,0]
    y_b2_start = WD.reshaped_wake.YB2[na_start:na_start+1,:,:,:,0]
    z_b2_start = WD.reshaped_wake.ZB2[na_start:na_start+1,:,:,:,0]        
        
    
    # Also append original blade panel for first row
    new_wake.reshaped_wake.XA1[:,:,:,:,nts-2-time_index] = x_a1_start
    new_wake.reshaped_wake.YA1[:,:,:,:,nts-2-time_index] = y_a1_start
    new_wake.reshaped_wake.ZA1[:,:,:,:,nts-2-time_index] = z_a1_start
    new_wake.reshaped_wake.XB1[:,:,:,:,nts-2-time_index] = x_b1_start
    new_wake.reshaped_wake.YB1[:,:,:,:,nts-2-time_index] = y_b1_start
    new_wake.reshaped_wake.ZB1[:,:,:,:,nts-2-time_index] = z_b1_start
    new_wake.reshaped_wake.XA2[:,:,:,:,nts-2-time_index] = x_a2_start
    new_wake.reshaped_wake.YA2[:,:,:,:,nts-2-time_index] = y_a2_start
    new_wake.reshaped_wake.ZA2[:,:,:,:,nts-2-time_index] = z_a2_start
    new_wake.reshaped_wake.XB2[:,:,:,:,nts-2-time_index] = x_b2_start
    new_wake.reshaped_wake.YB2[:,:,:,:,nts-2-time_index] = y_b2_start
    new_wake.reshaped_wake.ZB2[:,:,:,:,nts-2-time_index] = z_b2_start
    
    return

def advance_panels(wake, rotor, conditions, VD, new_wake, time_index, nts, dt, fun_V_induced, use_exact_velocities, use_RK2=True):
    """
    Use a second order Runge-Kutta integration scheme algorithm for advection of wake panels. Steps are as follows
       Step 1: Use a forward Euler as the predictor (x_t_plus_delta_t = x_t + U_t * delta_t)
       Step 2: Compute velocities at new locations (U_t_plus_delta_t = f(x_t_plus_delta_t) )
       Step 3: Correct prediction (x_t_plus_delta_t = x_t + (0.5 delta_t * (U_t_plus_delta_t + U_t) ) )
       
    """
    # -------------------------------------------------------------------------------------------------------------------------
    # Step 1: Forward Euler Predictor ; Advance all previously shed wake elements one step in time
    # -------------------------------------------------------------------------------------------------------------------------    
    # extract all previously shed panel positions

    shape_1 = np.shape(new_wake.reshaped_wake.XA1[:,:,:,:,nts-time_index:])
    shape_2 = np.shape(new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-time_index:])
    a = shape_1[0]
    m = shape_1[1]
    B = shape_1[2]
    Nr = shape_1[3] + 1
    nts_cur_1 = shape_1[4]
    nts_cur_2 = shape_2[4]
    
    # only take unique lagrangian markers (repeated on connecting panel corner nodes)
    xpts_1 = np.ravel(np.append(new_wake.reshaped_wake.XA1[:,:,:,:,nts-time_index:], new_wake.reshaped_wake.XB1[:,:,:,-1:,nts-time_index:], axis=3))
    ypts_1 = np.ravel(np.append(new_wake.reshaped_wake.YA1[:,:,:,:,nts-time_index:], new_wake.reshaped_wake.YB1[:,:,:,-1:,nts-time_index:], axis=3))
    zpts_1 = np.ravel(np.append(new_wake.reshaped_wake.ZA1[:,:,:,:,nts-time_index:], new_wake.reshaped_wake.ZB1[:,:,:,-1:,nts-time_index:], axis=3))

    xpts_2 = np.ravel(np.append(new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-time_index:], new_wake.reshaped_wake.XB2[:,:,:,-1:,nts-1-time_index:], axis=3))
    ypts_2 = np.ravel(np.append(new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-time_index:], new_wake.reshaped_wake.YB2[:,:,:,-1:,nts-1-time_index:], axis=3))
    zpts_2 = np.ravel(np.append(new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-time_index:], new_wake.reshaped_wake.ZB2[:,:,:,-1:,nts-1-time_index:], axis=3))   
    
    if use_exact_velocities:
        v1, v2 = get_exact_panel_velocities(rotor, wake, conditions, VD, new_wake, xpts_1, xpts_2, ypts_1, ypts_2, zpts_1, zpts_2)                     
    else:
        v1 = fun_V_induced((xpts_1, ypts_1, zpts_1))
        v2 = fun_V_induced((xpts_2, ypts_2, zpts_2)) 
        
        
    # compute the velocities at all panel nodes
    V1 = np.reshape( v1, (a, m, B, Nr, nts_cur_1, 3))
    V2 = np.reshape( v2, (a, m, B, Nr, nts_cur_2, 3))


    if use_RK2:
        X1_next_predictor = xpts_1 + np.ravel(V1[:,:,:,:,:,0] * dt)
        Y1_next_predictor = ypts_1 + np.ravel(V1[:,:,:,:,:,1] * dt)      
        Z1_next_predictor = zpts_1 + np.ravel(V1[:,:,:,:,:,2] * dt)
        X2_next_predictor = xpts_2 + np.ravel(V2[:,:,:,:,:,0] * dt)
        Y2_next_predictor = ypts_2 + np.ravel(V2[:,:,:,:,:,1] * dt)      
        Z2_next_predictor = zpts_2 + np.ravel(V2[:,:,:,:,:,2] * dt)

        
        new_wake2 = copy.deepcopy(new_wake)
        new_wake2.reshaped_wake.XA1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.XA1[:,:,:,:,nts-time_index:] + (V1[:,:,:,0:-1,:,0] * dt)
        new_wake2.reshaped_wake.YA1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.YA1[:,:,:,:,nts-time_index:] + (V1[:,:,:,0:-1,:,1] * dt)
        new_wake2.reshaped_wake.ZA1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.ZA1[:,:,:,:,nts-time_index:] + (V1[:,:,:,0:-1,:,2] * dt)
        
        new_wake2.reshaped_wake.XB1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.XB1[:,:,:,:,nts-time_index:] + (V1[:,:,:,1:,:,0] * dt)  
        new_wake2.reshaped_wake.YB1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.YB1[:,:,:,:,nts-time_index:] + (V1[:,:,:,1:,:,1] * dt)
        new_wake2.reshaped_wake.ZB1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.ZB1[:,:,:,:,nts-time_index:] + (V1[:,:,:,1:,:,2] * dt)            
        
        new_wake2.reshaped_wake.XA2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,0:-1,:,0] * dt)
        new_wake2.reshaped_wake.YA2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,0:-1,:,1] * dt)
        new_wake2.reshaped_wake.ZA2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,0:-1,:,2] * dt)
        
        new_wake2.reshaped_wake.XB2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,1:,:,0] * dt)
        new_wake2.reshaped_wake.YB2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,1:,:,1] * dt)
        new_wake2.reshaped_wake.ZB2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,1:,:,2] * dt)               
        
        # -------------------------------------------------------------------------------------------------------------------------
        # Step 2: Compute velocities at predictor locations 
        # -------------------------------------------------------------------------------------------------------------------------    
        if use_exact_velocities: # FIX: use next time step wake geometry, not just probe at those locations with current wake
            v1_plus, v2_plus = get_exact_panel_velocities(rotor, wake, conditions, VD, new_wake2, 
                                                        X1_next_predictor, X2_next_predictor, 
                                                        Y1_next_predictor, Y2_next_predictor,  
                                                        Z1_next_predictor, Z2_next_predictor)                                 
        else:
            v1_plus = fun_V_induced((X1_next_predictor, Y1_next_predictor, Z1_next_predictor))
            v2_plus = fun_V_induced((X2_next_predictor, Y2_next_predictor, Z2_next_predictor))
        
    
        V1_plus = np.reshape( v1_plus, (a, m, B, Nr, nts_cur_1, 3) )
        V2_plus = np.reshape( v2_plus, (a, m, B, Nr, nts_cur_2, 3) )
        

        # -------------------------------------------------------------------------------------------------------------------------
        # Step 3: Correct panel positions with 2nd order Runge Kutta integration scheme
        # -------------------------------------------------------------------------------------------------------------------------    

        new_wake.reshaped_wake.XA1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.XA1[:,:,:,:,nts-time_index:] + (V1[:,:,:,0:-1,:,0] + V1_plus[:,:,:,0:-1,:,0])* dt/2
        new_wake.reshaped_wake.YA1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.YA1[:,:,:,:,nts-time_index:] + (V1[:,:,:,0:-1,:,1] + V1_plus[:,:,:,0:-1,:,1])* dt/2
        new_wake.reshaped_wake.ZA1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.ZA1[:,:,:,:,nts-time_index:] + (V1[:,:,:,0:-1,:,2] + V1_plus[:,:,:,0:-1,:,2])* dt/2
        
        new_wake.reshaped_wake.XB1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.XB1[:,:,:,:,nts-time_index:] + (V1[:,:,:,1:,:,0] + V1_plus[:,:,:,1:,:,0])* dt/2          
        new_wake.reshaped_wake.YB1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.YB1[:,:,:,:,nts-time_index:] + (V1[:,:,:,1:,:,1] + V1_plus[:,:,:,1:,:,1])* dt/2
        new_wake.reshaped_wake.ZB1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.ZB1[:,:,:,:,nts-time_index:] + (V1[:,:,:,1:,:,2] + V1_plus[:,:,:,1:,:,2])* dt/2            
        
        new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,0:-1,:,0] + V2_plus[:,:,:,0:-1,:,0])* dt/2
        new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,0:-1,:,1] + V2_plus[:,:,:,0:-1,:,1])* dt/2
        new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,0:-1,:,2] + V2_plus[:,:,:,0:-1,:,2])* dt/2
        
        new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,1:,:,0] + V2_plus[:,:,:,1:,:,0])* dt/2
        new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,1:,:,1] + V2_plus[:,:,:,1:,:,1])* dt/2
        new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,1:,:,2] + V2_plus[:,:,:,1:,:,2])* dt/2               
        
    else:
        # use simple forward euler advection scheme
        # Update the new panel node positions
        new_wake.reshaped_wake.XA1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.XA1[:,:,:,:,nts-time_index:] + (V1[:,:,:,0:-1,:,0] * dt)
        new_wake.reshaped_wake.YA1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.YA1[:,:,:,:,nts-time_index:] + (V1[:,:,:,0:-1,:,1] * dt)
        new_wake.reshaped_wake.ZA1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.ZA1[:,:,:,:,nts-time_index:] + (V1[:,:,:,0:-1,:,2] * dt)
        
        new_wake.reshaped_wake.XB1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.XB1[:,:,:,:,nts-time_index:] + (V1[:,:,:,1:,:,0] * dt)  
        new_wake.reshaped_wake.YB1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.YB1[:,:,:,:,nts-time_index:] + (V1[:,:,:,1:,:,1] * dt)
        new_wake.reshaped_wake.ZB1[:,:,:,:,nts-time_index:] = new_wake.reshaped_wake.ZB1[:,:,:,:,nts-time_index:] + (V1[:,:,:,1:,:,2] * dt)            
        
        new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,0:-1,:,0] * dt)
        new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,0:-1,:,1] * dt)
        new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,0:-1,:,2] * dt)
        
        new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,1:,:,0] * dt)
        new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,1:,:,1] * dt)
        new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-time_index:] = new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-time_index:] + (V2[:,:,:,1:,:,2] * dt)       
    return
  
def get_exact_panel_velocities(rotor, wake, conditions, VD, new_wake, xpts_1, xpts_2, ypts_1, ypts_2, zpts_1, zpts_2):
    # extract panel nodes shed that are affected by velocity influences
    collocationPoints      = Data()
    collocationPoints.XC   = np.concatenate([xpts_1, xpts_2])
    collocationPoints.YC   = np.concatenate([ypts_1, ypts_2])
    collocationPoints.ZC   = np.concatenate([zpts_1, zpts_2])
    collocationPoints.n_cp = np.size(collocationPoints.XC)

    # extract only already shed vortices for influences
    new_wake_reduced = reduce_wake(new_wake,wake_treatment=False)
    WD_Network = Data()
    WD_Network[wake.tag + "_vortex_distribution"] = new_wake_reduced
    V_induced = compute_exact_velocity_field(WD_Network, rotor, conditions, VD_VLM=VD, GridPointsFull=collocationPoints)        

    v1 = V_induced[0, 0 : np.size(xpts_1), :]
    v2 = V_induced[0, np.size(xpts_1) :, :]
    return v1, v2

def generate_temp_wake(WD):
    temp_wake = Data()
    temp_wake.reshaped_wake = Data()
    refReshapedData   = WD.reshaped_wake.XA1[0:1,:,:,:,:]
    refData           = WD.XA1[0:1,:,:]
    
    temp_wake.reshaped_wake.XA1 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.XA2 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.YA1 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.YA2 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.ZA1 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.ZA2 = np.zeros_like(refReshapedData)
    
    temp_wake.reshaped_wake.XB1 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.XB2 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.YB1 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.YB2 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.ZB1 = np.zeros_like(refReshapedData)
    temp_wake.reshaped_wake.ZB2 = np.zeros_like(refReshapedData)

    temp_wake.reshaped_wake.GAMMA = np.zeros_like(refReshapedData)# copy.deepcopy(WD.reshaped_wake.GAMMA)    
    temp_wake.GAMMA = np.zeros_like(refData)#copy.deepcopy(WD.GAMMA)
    
    temp_wake.XA1 = np.zeros_like(refData)
    temp_wake.XA2 = np.zeros_like(refData)
    temp_wake.YA1 = np.zeros_like(refData)
    temp_wake.YA2 = np.zeros_like(refData)
    temp_wake.ZA1 = np.zeros_like(refData)
    temp_wake.ZA2 = np.zeros_like(refData)
    
    temp_wake.XB1 = np.zeros_like(refData)
    temp_wake.XB2 = np.zeros_like(refData)
    temp_wake.YB1 = np.zeros_like(refData)
    temp_wake.YB2 = np.zeros_like(refData)
    temp_wake.ZB1 = np.zeros_like(refData)
    temp_wake.ZB2 = np.zeros_like(refData)    
    
    return temp_wake

def reduce_wake(WD_total,wake_treatment=False):
    WD = copy.deepcopy(WD_total)
    WD_reduced = Data()
    WD_reduced.reshaped_wake = Data()
    
    # get number of panels shed in free wake so far
    oShape      = np.shape(WD.reshaped_wake.XA1)
    shed_panels = np.shape(WD.reshaped_wake.XA1[WD.reshaped_wake.XA1 != 0])[0] # total elements shed
    Nshed       = shed_panels // np.size(WD.reshaped_wake.XA1[:,:,:,:,0])       # number of time steps shed
    
    mat_size   = (oShape[0],oShape[1],Nshed*oShape[2]*oShape[3])  
    
    for key in list(WD.reshaped_wake.keys()):
        if Nshed == 0:
            # no panels shed, return zeros along 5th axis
            WD_reduced.reshaped_wake[key] = WD.reshaped_wake[key][:,:,:,:,oShape[-1]-1][:,:,:,:,None]
        else:
            WD_reduced.reshaped_wake[key] = WD.reshaped_wake[key][:,:,:,:,oShape[-1]-Nshed:]
            
            
    for key in list(WD.keys()):
        if key == 'reshaped_wake':
            pass
        else:
                       
            if (int(Nshed)) == 0:
                # no panels shed, return zeros
                WD_reduced[key] = np.empty([0,0,0])#np.reshape(WD_reduced.reshaped_wake[key], mat_size) #
            else:
            
                WD_reduced[key] = np.reshape(WD_reduced.reshaped_wake[key], mat_size)    
    # Wake treatment: Remove influences with circulation less than 10% of tip circulation
    if wake_treatment:
        WD_reduced = wake_treatment_removal(WD_reduced)
    return WD_reduced


def compute_panel_circulations(wake, rotor, conditions, nts, new_wake, na_start, ctrl_pts, i):
    va, vt = compute_fidelity_one_inflow_velocities(wake,rotor,ctrl_pts)        
    
    # Unpack rotor parameters
    R = rotor.tip_radius
    Rh = rotor.hub_radius
    B = rotor.number_of_blades
    tc = rotor.thickness_to_chord
    
    # Unpack rotor airfoil data
    a_geo   = rotor.airfoil_geometry
    a_loc   = rotor.airfoil_polar_stations
    cl_sur  = rotor.airfoil_cl_surrogates
    cd_sur  = rotor.airfoil_cd_surrogates    
    
    # Unpack wake inputs
    wake_inputs = rotor.outputs.wake_inputs
    Ua = wake_inputs.velocity_axial
    Ut  = wake_inputs.velocity_tangential
    ctrl_pts = wake_inputs.ctrl_pts
    Nr = wake_inputs.Nr
    Na = wake_inputs.Na    
    use_2d_analysis = wake_inputs.use_2d_analysis     
    beta = wake_inputs.twist_distribution
    c = wake_inputs.chord_distribution
    r = wake_inputs.radius_distribution
    a = wake_inputs.speed_of_sounds
    nu = wake_inputs.dynamic_viscosities
    
    # compute new blade velocities
    Wa   = va + Ua
    Wt   = Ut - vt

    lamdaw, F, _ = compute_inflow_and_tip_loss(r,R, Rh,Wa,Wt,B)

    # Compute aerodynamic forces based on specified input airfoil or surrogate
    Cl, Cdval, alpha, Ma, W, Re = compute_airfoil_aerodynamics(beta,c,r,R,B,F,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis)
    
    
    # compute HFW circulation at the blade
    gamma = 0.5*W*c*Cl        
    
    gfit = gamma
    #gfit = np.zeros_like(r)
    #for cpt in range(ctrl_pts):
        ## FILTER OUTLIER DATA
        #for a in range(Na):
            #gPoly = np.poly1d(np.polyfit(r[0,:,0], gamma[cpt,:,a], 4))
            #gfit[cpt,:,a] = F[cpt,:,a]*gPoly(r[0,:,0])  
            
    
    gamma_new = (gfit[:,:-1,:] + gfit[:,1:,:])*0.5  # [control points, Nr-1, Na ] one less radial station because ring
    num       = Na//B
    time_idx  = np.arange(nts)
    Gamma     = np.zeros((Na,ctrl_pts,B,Nr-1,nts))
    
    # generate Gamma for each start angle
    for ito in range(Na):
        t_idx     = np.atleast_2d(time_idx).T 
        B_idx     = np.arange(B) 
        B_loc     = (ito + B_idx*num - t_idx )%Na 
        Gamma1    = gamma_new[:,:,B_loc]  
        Gamma1    = Gamma1.transpose(0,3,1,2) 
        Gamma[ito,:,:,:,:] = Gamma1
    
    return Gamma


def wake_treatment_removal(WD_reduced):
    # Step 1: Remove influences from panels that have low strengths
    # Middle half of panels have much lower circulation strength, so reduce
    
    
    oShape = np.shape(WD_reduced.reshaped_wake.XA1)
    Nr = oShape[3]
    Nr_range = np.append(np.linspace(0,(Nr-1)//4, (Nr-1)//4 +1).astype(int), np.linspace(3*Nr//4,Nr-1, (Nr-1) - (3*Nr//4) + 1).astype(int))
    newShape = np.array(oShape)
    newShape[3] = len(Nr_range)

    mat_size   = (newShape[0],newShape[1],newShape[2]*newShape[3]*newShape[4])      
    for key in list(WD_reduced.reshaped_wake.keys()):
        WD_reduced.reshaped_wake[key] = WD_reduced.reshaped_wake[key][:,:,:,Nr_range,:]    
        WD_reduced[key] = np.reshape(WD_reduced.reshaped_wake[key], mat_size)
    
    return WD_reduced