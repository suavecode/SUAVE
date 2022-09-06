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
import copy

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def update_wake_position(wake, rotor, conditions, VD=None):  
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
    nts = len(WD.reshaped_wake.XA2[0,0,0,0,:])
    Na  = np.shape(WD.reshaped_wake.XA2)[0]
    omega = rotor.inputs.omega[0][0]
    dt = (2 * np.pi / Na) / omega
    
    #--------------------------------------------------------------------------------------------  
    # Step 1: Compute interpolated induced velocity field function, Vind = fun((x,y,z))
    #--------------------------------------------------------------------------------------------  
    try:
        fun_V_induced = rotor.Wake.fluid_domain_velocity_field
        interpolatedBoxData = rotor.Wake.fluid_domain_interpolated_box_data
    except:
        fun_V_induced, interpolatedBoxData = compute_interpolated_velocity_field(WD, rotor, conditions, VD)        
    
    #--------------------------------------------------------------------------------------------    
    # Step 2: Compute the position of new trailing edge panel under all velocity influences
    #--------------------------------------------------------------------------------------------
    new_wake = generate_temp_wake(WD) # Shape: ( azimuthal start index, control point , blade number , radial location on blade , time step )
    for i in range(nts):
        na_start = (i % Na)
        rolled_start = np.roll(np.linspace(0, Na-1, Na).astype(int), -na_start)

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
            
            xpts_a1 = np.reshape(new_wake.reshaped_wake.XA1[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.XA1[:,:,:,:,nts-1-i:]))
            ypts_a1 = np.reshape(new_wake.reshaped_wake.YA1[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.YA1[:,:,:,:,nts-1-i:]))
            zpts_a1 = np.reshape(new_wake.reshaped_wake.ZA1[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.ZA1[:,:,:,:,nts-1-i:]))
            xpts_b1 = np.reshape(new_wake.reshaped_wake.XB1[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.XB1[:,:,:,:,nts-1-i:]))
            ypts_b1 = np.reshape(new_wake.reshaped_wake.YB1[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.YB1[:,:,:,:,nts-1-i:]))
            zpts_b1 = np.reshape(new_wake.reshaped_wake.ZB1[:,:,:,:,nts-1-i:], np.size(new_wake.reshaped_wake.ZB1[:,:,:,:,nts-1-i:]))
            
            
            # compute the velocities at all panel nodes
            Va1 = np.reshape( fun_V_induced((xpts_a1, ypts_a1, zpts_a1)), np.append( np.shape(new_wake.reshaped_wake.XA1[:,:,:,:,nts-1-i:]),3) )
            Vb1 = np.reshape( fun_V_induced((xpts_b1, ypts_b1, zpts_b1)), np.append( np.shape(new_wake.reshaped_wake.XB1[:,:,:,:,nts-1-i:]),3) )
            Va2 = np.reshape( fun_V_induced((xpts_a2, ypts_a2, zpts_a2)), np.append( np.shape(new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-i:]),3) )
            Vb2 = np.reshape( fun_V_induced((xpts_b2, ypts_b2, zpts_b2)), np.append( np.shape(new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-i:]),3) )
 
            # Update the new panel node positions
            new_wake.reshaped_wake.XA1[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.XA1[:,:,:,:,nts-1-i:] + (Va1[:,:,:,:,:,0] * dt)
            new_wake.reshaped_wake.XB1[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.XB1[:,:,:,:,nts-1-i:] + (Vb1[:,:,:,:,:,0] * dt)  
            new_wake.reshaped_wake.YA1[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.YA1[:,:,:,:,nts-1-i:] + (Va1[:,:,:,:,:,1] * dt)
            new_wake.reshaped_wake.YB1[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.YB1[:,:,:,:,nts-1-i:] + (Vb1[:,:,:,:,:,1] * dt)
            new_wake.reshaped_wake.ZA1[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.ZA1[:,:,:,:,nts-1-i:] + (Va1[:,:,:,:,:,2] * dt)
            new_wake.reshaped_wake.ZB1[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.ZB1[:,:,:,:,nts-1-i:] + (Vb1[:,:,:,:,:,2] * dt)            
            new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-i:] + (Va2[:,:,:,:,:,0] * dt)
            new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-i:] + (Vb2[:,:,:,:,:,0] * dt)
            new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-i:] + (Va2[:,:,:,:,:,1] * dt)
            new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-i:] + (Vb2[:,:,:,:,:,1] * dt)
            new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-i:] + (Va2[:,:,:,:,:,2] * dt)
            new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-i:] = new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-i:] + (Vb2[:,:,:,:,:,2] * dt)   
        
        # use original wake panel for first row
        
        x_a1 = WD.reshaped_wake.XA1[rolled_start,:,:,:,1]
        y_a1 = WD.reshaped_wake.YA1[rolled_start,:,:,:,1]
        z_a1 = WD.reshaped_wake.ZA1[rolled_start,:,:,:,1]
        x_b1 = WD.reshaped_wake.XB1[rolled_start,:,:,:,1]
        y_b1 = WD.reshaped_wake.YB1[rolled_start,:,:,:,1]
        z_b1 = WD.reshaped_wake.ZB1[rolled_start,:,:,:,1]
        
        # get prior row of shed trailing edge nodes
        x_a2 = WD.reshaped_wake.XA2[rolled_start,:,:,:,1]
        y_a2 = WD.reshaped_wake.YA2[rolled_start,:,:,:,1]
        z_a2 = WD.reshaped_wake.ZA2[rolled_start,:,:,:,1]
        x_b2 = WD.reshaped_wake.XB2[rolled_start,:,:,:,1]
        y_b2 = WD.reshaped_wake.YB2[rolled_start,:,:,:,1]
        z_b2 = WD.reshaped_wake.ZB2[rolled_start,:,:,:,1]

        # pre-pend row to new wake
        new_wake.reshaped_wake.XA1[:,:,:,:,nts-1-i] = x_a1
        new_wake.reshaped_wake.YA1[:,:,:,:,nts-1-i] = y_a1
        new_wake.reshaped_wake.ZA1[:,:,:,:,nts-1-i] = z_a1
        new_wake.reshaped_wake.XB1[:,:,:,:,nts-1-i] = x_b1
        new_wake.reshaped_wake.YB1[:,:,:,:,nts-1-i] = y_b1
        new_wake.reshaped_wake.ZB1[:,:,:,:,nts-1-i] = z_b1
        new_wake.reshaped_wake.XA2[:,:,:,:,nts-1-i] = x_a2
        new_wake.reshaped_wake.YA2[:,:,:,:,nts-1-i] = y_a2
        new_wake.reshaped_wake.ZA2[:,:,:,:,nts-1-i] = z_a2
        new_wake.reshaped_wake.XB2[:,:,:,:,nts-1-i] = x_b2
        new_wake.reshaped_wake.YB2[:,:,:,:,nts-1-i] = y_b2
        new_wake.reshaped_wake.ZB2[:,:,:,:,nts-1-i] = z_b2
        
        
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
            for key in ['XA1', 'XA2', 'XB1', 'XB2','YA1', 'YA2', 'YB1', 'YB2','ZA1', 'ZA2', 'ZB1', 'ZB2']:
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
        
        
        wakeTemp = copy.deepcopy(wake)
        for key in ['XA1', 'XA2', 'XB1', 'XB2','YA1', 'YA2', 'YB1', 'YB2','ZA1', 'ZA2', 'ZB1', 'ZB2']:
            wakeTemp.vortex_distribution.reshaped_wake[key] = new_wake.reshaped_wake[key]
            wakeTemp.vortex_distribution[key] = new_wake[key]
        rotor.Wake = wakeTemp
        wake = wakeTemp
        #save_single_prop_vehicle_vtk(rotor, iteration=i, save_loc="/Users/rerha/Desktop/test_relaxed_wake/")   

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