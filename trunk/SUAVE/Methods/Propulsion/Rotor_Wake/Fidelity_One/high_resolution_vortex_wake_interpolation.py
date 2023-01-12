## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# high_resolution_vortex_wake_interpolation.py
# 
# Created:  Jan 2023, R. Erhard 
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
from SUAVE.Core import Data
import numpy as np 
from DCode.Common.plottingFunctions import *
from DCode.Common.generalFunctions import *

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def high_resolution_vortex_wake_interpolation(wake,rotor):  
    """
    This takes the vortex wake system and interpolates the geometry onto a higher resolution azimuthal
    grid. This is necessary for BVI analysis.
    
    """
    # Extract wake parameters and vortex distribution
    Na_high_res = wake.wake_settings.high_resolution_azimuthals
    
    VD_low_res = wake.vortex_distribution
    VD_lr_shape = np.shape(VD_low_res.reshaped_wake.XA1)
    Na_low_res = VD_lr_shape[0]
    m = VD_lr_shape[1]
    B = VD_lr_shape[2]
    nmax = VD_lr_shape[3]
    nts = VD_lr_shape[4]
    Nr = nmax + 1

    delta_psi_high_res = 360 * (1/(Na_high_res))
    delta_psi_low_res = 360 * (1/(Na_low_res))
    
    # Initialize new wake with high resolution azimuthal stations
    VD_high_res = initialize_distributions(VD_low_res, Nr, Na_high_res, B, nts, m)
    
    # For each azimuthal start location, interpolate from low-resolution VD
    keys = list(VD_high_res.keys())
    keys.remove('n_cp')
    keys.remove('XC')
    keys.remove('YC')
    keys.remove('ZC')
    
    for a in range(Na_high_res):
        print("a={}".format(a))
        # get closest low-res points
        if  a * delta_psi_high_res % delta_psi_low_res == 0:
            # on a low-res grid point, no interpolation necessary
            print("\tLow res value: {}".format(a))

            for element in keys:
                if element == 'reshaped_wake':
                    for element2 in list(VD_high_res.reshaped_wake.keys()):
                        VD_high_res.reshaped_wake[element2][a] = VD_low_res.reshaped_wake[element2][a * (Na_low_res) // (Na_high_res)]
                else:
                    VD_high_res[element][a] = VD_low_res[element][a * (Na_low_res) // (Na_high_res)]
            
                
        else:
            low_res_idx_lower = round(np.floor((a * delta_psi_high_res / 360) * (Na_low_res)))
            low_res_idx_upper = (low_res_idx_lower + 1) % Na_low_res
            print("\t{}".format(low_res_idx_lower))
            print("\t{}".format(low_res_idx_upper))
            
            # interpolate wake between low resolution grid points
            for element in keys:
                if element == 'reshaped_wake':
                    for element2 in list(VD_high_res.reshaped_wake.keys()):
                        A_lr_lower = VD_low_res.reshaped_wake[element2][low_res_idx_lower]
                        A_lr_upper = VD_low_res.reshaped_wake[element2][low_res_idx_upper]
                        
                        # simple linear interpolation
                        VD_high_res.reshaped_wake[element2][a] = A_lr_lower + a * (Na_low_res) % (Na_high_res) / (Na_high_res) * (A_lr_upper - A_lr_lower)
                else:
                    A_lr_lower = VD_low_res[element][low_res_idx_lower]
                    A_lr_upper = VD_low_res[element][low_res_idx_upper]
                    
                    # simple linear interpolation
                    VD_high_res[element][a] = A_lr_lower + (a * (Na_low_res) % (Na_high_res) / (Na_high_res))* (A_lr_upper - A_lr_lower)  
    
    # export vtk of wake over each Na_high_res (debug step)
    rotor.Wake.vortex_distribution = VD_high_res
    debug = True
    if debug:
        for a in range(Na_high_res):
            rotor.number_azimuthal_stations = Na_high_res
            rotor.start_angle_idx = a
            save_single_prop_vehicle_vtk(rotor, time_step=a, save_loc="/Users/rerha/Desktop/test_relaxed_wake_2/")  
    
    
    
    return
  


## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def initialize_distributions(VD_low_res, Nr, Na, B, n_wts, m):
    """
    Initializes the matrices for the wake vortex distributions.
    
    Assumptions:
        None

    Source:
        N/A
        
    Inputs:
       Nr    - number of radial blade elemnts
       Na    - number of azimuthal start positions
       B     - number of rotor blades
       n_wts - total number of wake time steps in wake simulation
       m     - number of control points to evaluate
       VD    - vehicle vortex distribution
       
    Outputs:
       VD  - Vortex distribution
       WD  - Wake vortex distribution
    
    Properties:
       N/A
       
    """
    
    VD = Data()
    VD.reshaped_wake       = Data()
    
    for element in list(VD_low_res.keys()):
        if element == 'reshaped_wake':
            for element2 in list(VD_low_res.reshaped_wake.keys()):
                shape = list(np.shape(VD_low_res.reshaped_wake[element2])[1:])
                shape.insert(0,Na) 
                VD.reshaped_wake[element2] = np.zeros((shape))
        elif element in ['n_cp','XC','YC','ZC']:
            VD[element] = VD_low_res[element]
        else:
            shape = list(np.shape(VD_low_res[element])[1:])
            shape.insert(0,Na) 
            VD[element] = np.zeros((shape))
    
    return VD