## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# update_wake_position.py
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
    
    use_exact_velocities = False
    debug = False    
    step_1_times = np.zeros(nts-1)
    step_2_times = np.zeros(nts-1)
    
    # Time marching approach, loop over number of time steps
    for i in range(nts-1):
        
        print("\nTime step "+str(i)+" out of " + str(nts-1))
        ##--------------------------------------------------------------------------------------------  
        ## Step 1: Compute interpolated induced velocity field function, Vind = fun((x,y,z))
        ##--------------------------------------------------------------------------------------------  
        if not use_exact_velocities:
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
            
            
            
        #--------------------------------------------------------------------------------------------    
        # Step 2: Compute the position of new trailing edge panel under all velocity influences
        #--------------------------------------------------------------------------------------------            
        na_start = (i % Na)
        rolled_start = np.roll(np.linspace(0, Na-1, Na).astype(int), -na_start)
        t0 = time.time()
        if i!=0:
            # -------------------------------------------------------------------------------------------------------------------------
            # Step 1: Advance all previously shed wake elements one step in time
            # -------------------------------------------------------------------------------------------------------------------------
            # extract all previously shed panel positions
            xpts_a2 = np.reshape(new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-i:]))
            ypts_a2 = np.reshape(new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-i:]))
            zpts_a2 = np.reshape(new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-i:]))
            xpts_b2 = np.reshape(new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-i:]))
            ypts_b2 = np.reshape(new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-i:]))
            zpts_b2 = np.reshape(new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-i:]))    
            
            xpts_a1 = np.reshape(new_wake.reshaped_wake.XA1[:,:,:,:,nts-i:], np.size(new_wake.reshaped_wake.XA1[:,:,:,:,nts-i:]))
            ypts_a1 = np.reshape(new_wake.reshaped_wake.YA1[:,:,:,:,nts-i:], np.size(new_wake.reshaped_wake.YA1[:,:,:,:,nts-i:]))
            zpts_a1 = np.reshape(new_wake.reshaped_wake.ZA1[:,:,:,:,nts-i:], np.size(new_wake.reshaped_wake.ZA1[:,:,:,:,nts-i:]))
            xpts_b1 = np.reshape(new_wake.reshaped_wake.XB1[:,:,:,:,nts-i:], np.size(new_wake.reshaped_wake.XB1[:,:,:,:,nts-i:]))
            ypts_b1 = np.reshape(new_wake.reshaped_wake.YB1[:,:,:,:,nts-i:], np.size(new_wake.reshaped_wake.YB1[:,:,:,:,nts-i:]))
            zpts_b1 = np.reshape(new_wake.reshaped_wake.ZB1[:,:,:,:,nts-i:], np.size(new_wake.reshaped_wake.ZB1[:,:,:,:,nts-i:]))
             
        

            if not use_exact_velocities:
                va1 = fun_V_induced((xpts_a1, ypts_a1, zpts_a1))
                va2 = fun_V_induced((xpts_a2, ypts_a2, zpts_a2))
                vb1 = fun_V_induced((xpts_b1, ypts_b1, zpts_b1))
                vb2 = fun_V_induced((xpts_b2, ypts_b2, zpts_b2))
            else:
                # extract panel nodes shed that are affected by velocity influences
                collocationPoints      = Data()
                collocationPoints.XC   = np.concatenate([np.reshape(xpts_a1, np.size(xpts_a1)), np.reshape(xpts_a2, np.size(xpts_a2)) , np.reshape(xpts_b1, np.size(xpts_b1)), np.reshape(xpts_b2, np.size(xpts_b2))])
                collocationPoints.YC   = np.concatenate([np.reshape(ypts_a1, np.size(ypts_a1)), np.reshape(ypts_a2, np.size(ypts_a2)) , np.reshape(ypts_b1, np.size(ypts_b1)), np.reshape(ypts_b2, np.size(ypts_b2))])
                collocationPoints.ZC   = np.concatenate([np.reshape(zpts_a1, np.size(zpts_a1)), np.reshape(zpts_a2, np.size(zpts_a2)) , np.reshape(zpts_b1, np.size(zpts_b1)), np.reshape(zpts_b2, np.size(zpts_b2))])
                collocationPoints.n_cp = np.size(collocationPoints.XC)
                
                # extract only already shed vortices for influences
                new_wake_reduced = reduce_wake(new_wake,wake_treatment=False)
                WD_Network = Data()
                WD_Network[wake.tag + "_vortex_distribution"] = new_wake_reduced
                V_induced = compute_exact_velocity_field(WD_Network, rotor, conditions, VD_VLM=VD, GridPointsFull=collocationPoints)        
                
                va1 = V_induced[0, 0 : np.size(xpts_a1), :]
                va2 = V_induced[0, np.size(xpts_a1): (np.size(xpts_a1) + np.size(xpts_a2)), :]
                vb1 = V_induced[0, (np.size(xpts_a1) + np.size(xpts_a2)) : (np.size(xpts_a1) + np.size(xpts_a2) + np.size(xpts_b1)),:]
                vb2 = V_induced[0, (np.size(xpts_a1) + np.size(xpts_a2) + np.size(xpts_b1)) : (np.size(xpts_a1) + np.size(xpts_a2) + np.size(xpts_b1) + np.size(xpts_b2)),:]                         
                
            
            # compute the velocities at all panel nodes
            Va1 = np.reshape( va1, np.append( np.shape(new_wake.reshaped_wake.XA1[:,:,:,:,nts-i:]),3) )
            Vb1 = np.reshape( vb1, np.append( np.shape(new_wake.reshaped_wake.XB1[:,:,:,:,nts-i:]),3) )
            Va2 = np.reshape( va2, np.append( np.shape(new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-i:]),3) )
            Vb2 = np.reshape( vb2, np.append( np.shape(new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-i:]),3) )
    
            # Update the new panel node positions
            new_wake.reshaped_wake.XA1[:,:,:,:,nts-i:] = new_wake.reshaped_wake.XA1[:,:,:,:,nts-i:] + (Va1[:,:,:,:,:,0] * dt)
            new_wake.reshaped_wake.XB1[:,:,:,:,nts-i:] = new_wake.reshaped_wake.XB1[:,:,:,:,nts-i:] + (Vb1[:,:,:,:,:,0] * dt)  
            new_wake.reshaped_wake.YA1[:,:,:,:,nts-i:] = new_wake.reshaped_wake.YA1[:,:,:,:,nts-i:] + (Va1[:,:,:,:,:,1] * dt)
            new_wake.reshaped_wake.YB1[:,:,:,:,nts-i:] = new_wake.reshaped_wake.YB1[:,:,:,:,nts-i:] + (Vb1[:,:,:,:,:,1] * dt)
            new_wake.reshaped_wake.ZA1[:,:,:,:,nts-i:] = new_wake.reshaped_wake.ZA1[:,:,:,:,nts-i:] + (Va1[:,:,:,:,:,2] * dt)
            new_wake.reshaped_wake.ZB1[:,:,:,:,nts-i:] = new_wake.reshaped_wake.ZB1[:,:,:,:,nts-i:] + (Vb1[:,:,:,:,:,2] * dt)            
            new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-i:] + (Va2[:,:,:,:,:,0] * dt)
            new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-i:] + (Vb2[:,:,:,:,:,0] * dt)
            new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-i:] + (Va2[:,:,:,:,:,1] * dt)
            new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-i:] + (Vb2[:,:,:,:,:,1] * dt)
            new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-i:] + (Va2[:,:,:,:,:,2] * dt)
            new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-i:] + (Vb2[:,:,:,:,:,2] * dt)   
                
            # connect trailing edge of first panel
        
        # use original wake panel for first row
        
        x_a1 = WD.reshaped_wake.XA1[rolled_start,:,:,:,1]
        y_a1 = WD.reshaped_wake.YA1[rolled_start,:,:,:,1]
        z_a1 = WD.reshaped_wake.ZA1[rolled_start,:,:,:,1]
        x_b1 = WD.reshaped_wake.XB1[rolled_start,:,:,:,1]
        y_b1 = WD.reshaped_wake.YB1[rolled_start,:,:,:,1]
        z_b1 = WD.reshaped_wake.ZB1[rolled_start,:,:,:,1]

        # pre-pend row to new wake
        new_wake.reshaped_wake.XA1[:,:,:,:,nts-1-i] = x_a1
        new_wake.reshaped_wake.YA1[:,:,:,:,nts-1-i] = y_a1
        new_wake.reshaped_wake.ZA1[:,:,:,:,nts-1-i] = z_a1
        new_wake.reshaped_wake.XB1[:,:,:,:,nts-1-i] = x_b1
        new_wake.reshaped_wake.YB1[:,:,:,:,nts-1-i] = y_b1
        new_wake.reshaped_wake.ZB1[:,:,:,:,nts-1-i] = z_b1
        
        # compute new circulation based on prior wake
        va, vt = compute_fidelity_one_inflow_velocities(wake,rotor,ctrl_pts)   
        GAMMA = compute_blade_circulation(wake, rotor, conditions, nts, va=va, vt=vt)
        new_wake.reshaped_wake.GAMMA[:,:,:,:,nts-1-i] = GAMMA[rolled_start,:,:,:,1]          
        
        x_a1_start = WD.reshaped_wake.XA1[rolled_start,:,:,:,0]
        y_a1_start = WD.reshaped_wake.YA1[rolled_start,:,:,:,0]
        z_a1_start = WD.reshaped_wake.ZA1[rolled_start,:,:,:,0]
        x_b1_start = WD.reshaped_wake.XB1[rolled_start,:,:,:,0]
        y_b1_start = WD.reshaped_wake.YB1[rolled_start,:,:,:,0]
        z_b1_start = WD.reshaped_wake.ZB1[rolled_start,:,:,:,0]
        
        # get prior row of shed trailing edge nodes
        x_a2_start = WD.reshaped_wake.XA2[rolled_start,:,:,:,0]
        y_a2_start = WD.reshaped_wake.YA2[rolled_start,:,:,:,0]
        z_a2_start = WD.reshaped_wake.ZA2[rolled_start,:,:,:,0]
        x_b2_start = WD.reshaped_wake.XB2[rolled_start,:,:,:,0]
        y_b2_start = WD.reshaped_wake.YB2[rolled_start,:,:,:,0]
        z_b2_start = WD.reshaped_wake.ZB2[rolled_start,:,:,:,0]        
            
        
        if i != nts-1:
            # Also append original blade panel for first row

            new_wake.reshaped_wake.XA1[:,:,:,:,nts-2-i] = x_a1_start
            new_wake.reshaped_wake.YA1[:,:,:,:,nts-2-i] = y_a1_start
            new_wake.reshaped_wake.ZA1[:,:,:,:,nts-2-i] = z_a1_start
            new_wake.reshaped_wake.XB1[:,:,:,:,nts-2-i] = x_b1_start
            new_wake.reshaped_wake.YB1[:,:,:,:,nts-2-i] = y_b1_start
            new_wake.reshaped_wake.ZB1[:,:,:,:,nts-2-i] = z_b1_start
            new_wake.reshaped_wake.XA2[:,:,:,:,nts-2-i] = x_a2_start
            new_wake.reshaped_wake.YA2[:,:,:,:,nts-2-i] = y_a2_start
            new_wake.reshaped_wake.ZA2[:,:,:,:,nts-2-i] = z_a2_start
            new_wake.reshaped_wake.XB2[:,:,:,:,nts-2-i] = x_b2_start
            new_wake.reshaped_wake.YB2[:,:,:,:,nts-2-i] = y_b2_start
            new_wake.reshaped_wake.ZB2[:,:,:,:,nts-2-i] = z_b2_start
        else:
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
            
            
            new_wake.reshaped_wake.XA1[:,:,:,:,0] = x_a1_start
            new_wake.reshaped_wake.YA1[:,:,:,:,0] = y_a1_start
            new_wake.reshaped_wake.ZA1[:,:,:,:,0] = z_a1_start
            new_wake.reshaped_wake.XB1[:,:,:,:,0] = x_b1_start
            new_wake.reshaped_wake.YB1[:,:,:,:,0] = y_b1_start
            new_wake.reshaped_wake.ZB1[:,:,:,:,0] = z_b1_start
            new_wake.reshaped_wake.XA2[:,:,:,:,0] = x_a2_start
            new_wake.reshaped_wake.YA2[:,:,:,:,0] = y_a2_start
            new_wake.reshaped_wake.ZA2[:,:,:,:,0] = z_a2_start
            new_wake.reshaped_wake.XB2[:,:,:,:,0] = x_b2_start
            new_wake.reshaped_wake.YB2[:,:,:,:,0] = y_b2_start
            new_wake.reshaped_wake.ZB2[:,:,:,:,0] = z_b2_start        
            
            # rotate wake to start
            start_roll = np.roll(np.linspace(0, Na-1, Na).astype(int), -1)
            for key in ['XA1', 'XA2', 'XB1', 'XB2','YA1', 'YA2', 'YB1', 'YB2','ZA1', 'ZA2', 'ZB1', 'ZB2','GAMMA']:
                new_wake.reshaped_wake[key] = new_wake.reshaped_wake[key][start_roll,:,:,:,:]
            

        # -------------------------------------------------------------------------------------------
        # -----DEBUG-----temp store wake to monitor development
        # -------------------------------------------------------------------------------------------
        
        #     # Compress Data into 1D Arrays
        reshaped_shape = np.shape(new_wake.reshaped_wake.XA1)
        m  = reshaped_shape[1]
        B  = reshaped_shape[2]
        Nr = reshaped_shape[3]
        mat6_size = (Na,m,nts*B*(Nr)) 
        
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
        
        # Connect the newest shed panel to the prior shed panel
        
        
    #for key in ['XA1', 'XA2', 'XB1', 'XB2','YA1', 'YA2', 'YB1', 'YB2','ZA1', 'ZA2', 'ZB1', 'ZB2']:
        #wake.vortex_distribution.reshaped_wake[key] = new_wake[key]
    #rotor.Wake = wake
    #--------------------------------------------------------------------------------------------    
    # Step 6: Update the position of all wake panels 
    #--------------------------------------------------------------------------------------------

    return wake, rotor, interpolatedBoxData
  
def generate_temp_wake(WD):
    temp_wake = Data()
    temp_wake.reshaped_wake = Data()
    
    temp_wake.reshaped_wake.XA1 = np.zeros_like(WD.reshaped_wake.XA1)
    temp_wake.reshaped_wake.XA2 = np.zeros_like(WD.reshaped_wake.XA2)
    temp_wake.reshaped_wake.YA1 = np.zeros_like(WD.reshaped_wake.YA1)
    temp_wake.reshaped_wake.YA2 = np.zeros_like(WD.reshaped_wake.YA2)
    temp_wake.reshaped_wake.ZA1 = np.zeros_like(WD.reshaped_wake.ZA1)
    temp_wake.reshaped_wake.ZA2 = np.zeros_like(WD.reshaped_wake.ZA2)
    
    temp_wake.reshaped_wake.XB1 = np.zeros_like(WD.reshaped_wake.XB1)
    temp_wake.reshaped_wake.XB2 = np.zeros_like(WD.reshaped_wake.XB2)
    temp_wake.reshaped_wake.YB1 = np.zeros_like(WD.reshaped_wake.YB1)
    temp_wake.reshaped_wake.YB2 = np.zeros_like(WD.reshaped_wake.YB2)
    temp_wake.reshaped_wake.ZB1 = np.zeros_like(WD.reshaped_wake.ZB1)
    temp_wake.reshaped_wake.ZB2 = np.zeros_like(WD.reshaped_wake.ZB2)

    temp_wake.reshaped_wake.GAMMA = np.zeros_like(WD.reshaped_wake.GAMMA)# copy.deepcopy(WD.reshaped_wake.GAMMA)    
    temp_wake.GAMMA = np.zeros_like(WD.GAMMA)#copy.deepcopy(WD.GAMMA)
    

    temp_wake.XA1 = np.zeros_like(WD.XA1)
    temp_wake.XA2 = np.zeros_like(WD.XA2)
    temp_wake.YA1 = np.zeros_like(WD.YA1)
    temp_wake.YA2 = np.zeros_like(WD.YA2)
    temp_wake.ZA1 = np.zeros_like(WD.ZA1)
    temp_wake.ZA2 = np.zeros_like(WD.ZA2)
    
    temp_wake.XB1 = np.zeros_like(WD.XB1)
    temp_wake.XB2 = np.zeros_like(WD.XB2)
    temp_wake.YB1 = np.zeros_like(WD.YB1)
    temp_wake.YB2 = np.zeros_like(WD.YB2)
    temp_wake.ZB1 = np.zeros_like(WD.ZB1)
    temp_wake.ZB2 = np.zeros_like(WD.ZB2)    
    
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


def compute_blade_circulation(wake, rotor, conditions, nts, va=0, vt=0):
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
    Cl, Cdval, alpha, Ma,W = compute_airfoil_aerodynamics(beta,c,r,R,B,F,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis)
    
    
    # compute HFW circulation at the blade
    gamma = 0.5*W*c*Cl        
    
    gfit = np.zeros_like(r)
    for cpt in range(ctrl_pts):
        # FILTER OUTLIER DATA
        for a in range(Na):
            gPoly = np.poly1d(np.polyfit(r[0,:,0], gamma[cpt,:,a], 4))
            gfit[cpt,:,a] = F[cpt,:,a]*gPoly(r[0,:,0])  
            
    
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