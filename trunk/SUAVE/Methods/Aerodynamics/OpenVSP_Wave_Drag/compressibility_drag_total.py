## @ingroup Methods-Aerodynamics-OpenVSP_Wave_Drag
# compressibility_drag_total.py
# 
# Created:  Aug 2014, T. MacDonald
# Modified: Jun 2017, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Analyses import Results
from SUAVE.Core import (
    Data, Container,
)

from wave_drag_lift import wave_drag_lift
from wave_drag_volume import wave_drag_volume

import copy

# package imports
import autograd.numpy as np

# ----------------------------------------------------------------------
#  Compressibility Drag Total
# ----------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-OpenVSP_Wave_Drag
def compressibility_drag_total(state,settings,geometry):
    """Computes compressibility drag for full aircraft including volume drag through OpenVSP

    Assumptions:
    None

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    settings.number_slices
    settings.number_rotations
    state.conditions.aerodynamics.
      lift_breakdown.compressible_wings      [-]
    state.conditions.freestream.mach_number  [-]
    geometry.wings.*.tag                       

    Outputs:
    drag_breakdown.compressible[wing.tag].
      divergence_mach                        [-]
    drag_breakdown.compressible.total        [-]                    
    drag_breakdown.compressible.total_volume [-]
    drag_breakdown.compressible.total_lift   [-]
    cd_c                                     [-] Total compressibility drag

    Properties Used:
    N/A
    """     

    # Unpack
    conditions       = state.conditions
    configuration    = settings
    number_slices    = settings.number_slices
    number_rotations = settings.number_rotations
    
    wings          = geometry.wings
    fuselages      = geometry.fuselages
    propulsor_name = geometry.propulsors.keys()[0] #obtain the key for the propulsor for assignment purposes
    propulsor      = geometry.propulsors[propulsor_name]

    Mc             = conditions.freestream.mach_number
    drag_breakdown = conditions.aerodynamics.drag_breakdown

    # Initialize result
    drag_breakdown.compressible = Results()
    
    # Use main wing reference area for drag coefficients
    Sref_main = geometry.reference_area
    

    drag99_total  = np.zeros(np.shape(Mc))
    drag105_total = np.zeros(np.shape(Mc))

    # Iterate through wings
    for k in wings.keys():
        
        wing = wings[k]

        # initialize array to correct length
        cd_c = np.array([[0.0]] * len(Mc))
        mcc  = np.array([[0.0]] * len(Mc))
        MDiv = np.array([[0.0]] * len(Mc))     


        # Get main fuselage data - note that name of fuselage is important here
        # This should be changed to be general 
        main_fuselage = fuselages['fuselage']

        # Get number of engines data
        num_engines = propulsor.number_of_engines

        # Get the lift coefficient of the wing.
        # Note that this is not the total CL
        cl = conditions.aerodynamics.lift_breakdown.compressible_wings

        # Calculate compressibility drag at Mach 0.99 and 1.05 for interpolation between
        # dummy variables are unused function outputs
        (drag99,dummy1,dummy2) = drag_div(np.array([[0.99]] * len(Mc)),wing,k,cl,Sref_main)
        cdc_l = lift_wave_drag(conditions, 
                                  configuration, 
                                  wing, 
                                  k,Sref_main,True)
        
        drag99_total  = drag99_total + drag99
        drag105_total = drag105_total + cdc_l
        
    try:
        old_array = np.load('volume_drag_data_' + geometry.tag + '.npy')
        file_exists = True
    except:
        file_exists = False
        old_array = np.array([[-1,-1]]) 
          
    if np.any(old_array[:,0]==1.05):
        cd_c_v = np.array([[float(old_array[old_array[:,0]==1.05,1])]])
    else:    
        cd_c_v = wave_drag_volume(conditions,geometry, True)
    
    if file_exists:
        pass
    else:
        new_save_row = np.array([[1.05,cd_c_v]])
        np.save('volume_drag_data_' + geometry.tag + '.npy', new_save_row)    
    

    drag105 = drag105_total + cd_c_v*np.ones(np.shape(Mc))
    drag99  = drag99_total
    cd_c_l  = np.array([[0.0]] * len(Mc))

    # For subsonic mach numbers, use drag divergence correlations to find the drag
    for k in wings.keys():
        wing = wings[k]    
        (a,b,c) = drag_div(Mc[Mc <= 0.99],wing,k,cl[Mc <= 0.99],Sref_main)
        cd_c[Mc <= 0.99] = cd_c[Mc <= 0.99] + a
        mcc[Mc <= 0.99]  = b
        MDiv[Mc <= 0.99] = c
        drag_breakdown.compressible[wing.tag]    = Data()
        drag_breakdown.compressible[wing.tag].divergence_mach = MDiv
        cd_c_l = lift_wave_drag(conditions, 
                                configuration, 
                                wing, 
                                k,Sref_main,False) + cd_c_l     

    # For mach numbers close to 1, use an interpolation to avoid intensive calculations
    cd_c[Mc > 0.99] = drag99[Mc > 0.99] + (drag105[Mc > 0.99]-drag99[Mc > 0.99])*(Mc[Mc > 0.99]-0.99)/(1.05-0.99)

    # Use wave drag equations at supersonic values. The cutoff for this function is 1.05
    # Only the supsonic results are returned with nonzero values

        
    cd_c_v = wave_drag_volume(conditions, geometry, False,num_slices=number_slices,num_rots=number_rotations)
        
    cd_c[Mc >= 1.05] = cd_c_l[Mc >= 1.05] + cd_c_v[Mc >= 1.05]

    
    # Save drag breakdown

    drag_breakdown.compressible.total        = cd_c
    drag_breakdown.compressible.total_volume = cd_c_v
    drag_breakdown.compressible.total_lift   = cd_c_l
    

    return cd_c

## @ingroup Methods-Aerodynamics-OpenVSP_Wave_Drag
def drag_div(Mc_ii,wing,k,cl,Sref_main):
    """Use drag divergence mach number to determine drag for subsonic speeds

    Assumptions:
    Basic fit, subsonic

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    wing.
      thickness_to_chord    [-]     
      sweeps.quarter_chord  [radians]
      high_mach             [Boolean]
      areas.reference       [m^2]

    Outputs:
    cd_c                    [-]
    mcc                     [-]
    MDiv                    [-]

    Properties Used:
    N/A
    """         
    # Use drag divergence mach number to determine drag for subsonic speeds

    # Check if the wing is designed for high subsonic cruise
    # If so use arbitrary divergence point as correlation will not work
    if wing.high_mach is True:

        # Divergence mach number
        MDiv = np.array([0.95] * len(Mc_ii))
        mcc = np.array([0.93] * len(Mc_ii))

    else:
        # Unpack wing
        t_c_w   = wing.thickness_to_chord
        sweep_w = wing.sweeps.quarter_chord

        # Check if this is the main wing, other wings are assumed to have no lift
        if k == 'main_wing':
            cl_w = cl
        else:
            cl_w = 0

        # Get effective Cl and sweep
        cos_sweep = np.cos(sweep_w)
        tc = t_c_w / cos_sweep
        cl = cl_w / (cos_sweep*cos_sweep)

        # Compressibility drag based on regressed fits from AA241
        mcc_cos_ws = 0.922321524499352       \
            - 1.153885166170620*tc    \
            - 0.304541067183461*cl    \
            + 0.332881324404729*tc*tc \
            + 0.467317361111105*tc*cl \
            + 0.087490431201549*cl*cl

        # Crest-critical mach number, corrected for wing sweep
        mcc = mcc_cos_ws / cos_sweep

        # Divergence mach number
        MDiv = mcc * ( 1.02 + 0.08*(1 - cos_sweep) )        

    # Divergence ratio
    mo_mc = Mc_ii/mcc

    # Compressibility correlation, Shevell
    dcdc_cos3g = 0.0019*mo_mc**14.641

    # Compressibility drag

    # Sweep correlation cannot be used if the wing has a high mach design
    if wing.high_mach is True:
        cd_c = dcdc_cos3g
    else:
        cd_c = dcdc_cos3g * (np.cos(sweep_w))**3
        
    cd_c = cd_c*wing.areas.reference/Sref_main    

    return (cd_c,mcc,MDiv)

## @ingroup Methods-Aerodynamics-OpenVSP_Wave_Drag
def lift_wave_drag(conditions,configuration,wing,k,Sref_main,flag105):
    """Determine lift wave drag for supersonic speeds

    Assumptions:
    Basic fit

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    conditions.freestream.mach_number [-]
    configuration                     (passed to another function)
    wing.areas.reference              [m^2]
    k                                 (unused)
    Sref_main                         [m^2] Main reference area
    flag105                           <boolean> Check if calcs are for Mach 1.05

    Outputs:
    cd_c_l                            [-] Wave drag CD due to lift

    Properties Used:
    N/A
    """       
    # Use wave drag to determine compressibility drag for supersonic speeds

    # Unpack mach number
    mach       = conditions.freestream.mach_number

    # Create a copy that can be modified
    Mc         = mach*1.

    # This flag is for the interpolation mode
    if flag105 is True:
        # Change conditions and short hand for calculations
        conditions.freestream.mach_number = np.array([[1.05]] * len(Mc))
        mach = conditions.freestream.mach_number

    # Initalize cd arrays
    cd_c_l = np.array([[0.0]] * len(mach)) # lift wave drag

    # Calculate wing values at all mach numbers
    # Note that these functions arrange the supersonic values at the beginning of the array
    cd_lift_wave = wave_drag_lift(conditions,configuration,wing)

    # Pack supersonic results into correct elements
    cd_c_l[mach >= 1.05] = cd_lift_wave[0:len(mach[mach >= 1.05]),0]

    # Convert coefficient to full aircraft value
    cd_c_l = cd_c_l*wing.areas.reference/Sref_main

    # Reset mach number to real values
    conditions.freestream.mach_number = Mc

    return cd_c_l
