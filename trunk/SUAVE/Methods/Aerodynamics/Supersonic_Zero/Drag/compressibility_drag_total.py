## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# compressibility_drag_total.py
# 
# Created:  Aug 2014, T. MacDonald
# Modified: Jun 2017, T. MacDonald
#           Jul 2017, T. MacDonald
#           Aug 2018, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag import \
      wave_drag_volume, wave_drag_body_of_rev
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import wave_drag_lift

import copy

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compressibility Drag Total
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def compressibility_drag_total(state,settings,geometry):
    """Computes compressibility drag for full aircraft

    Assumptions:
    Drag is only calculated for the wings, main fuselage, and propulsors
    Main fuselage must have tag 'fuselage'
    No lift on wings other than main wing

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    state.conditions.aerodynamics.lift_breakdown.compressible_wings  [Unitless]
    state.conditions.freestream.mach_number                          [Unitless]
    geometry.wings                             
    geometry.fuselages['fuselage'].length_total                      [m]
    geometry.fuselages['fuselage'].effective_diameter                [m]
    geometry.propulsors[geometry.propulsors.keys()[0]].
      nacelle_diameter                                               [m]
      engine_length                                                  [m]
      number_of_engines                                              [m]

    Outputs:
    total_compressibility_drag                                       [Unitless]

    Properties Used:
    N/A
    """     

    # Unpack
    conditions    = state.conditions
    configuration = settings
    
    wings          = geometry.wings
    fuselages      = geometry.fuselages
    propulsor_name = list(geometry.propulsors.keys())[0] #obtain the key for the propulsor for assignment purposes
    propulsor      = geometry.propulsors[propulsor_name]

    Mc             = conditions.freestream.mach_number
    drag_breakdown = conditions.aerodynamics.drag_breakdown

    # Initialize result
    drag_breakdown.compressible = Data()
    
    # Use the vehicle for drag coefficients
    Sref_main = geometry.reference_area
    

    # Iterate through wings
    for k in wings.keys():
        
        wing = wings[k]

        # initialize array to correct length
        cd_c = np.array([[0.0]] * len(Mc))
        mcc = np.array([[0.0]] * len(Mc))
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
        (drag99,a,b) = drag_div(np.array([[0.99]] * len(Mc)),wing,k,cl,Sref_main)
        (drag105,a,b,cd_c_l_105,cd_c_v_105) = wave_drag(conditions, 
                                  configuration, 
                                  main_fuselage, 
                                  propulsor, 
                                  wing, 
                                  num_engines,k,Sref_main,True)

        # For subsonic mach numbers, use drag divergence correlations to find the drag
        (cd_c[Mc[:,0] <= 0.99],mcc[Mc[:,0] <= 0.99], MDiv[Mc[:,0] <= 0.99]) = drag_div(Mc[Mc[:,0] <= 0.99],wing,k,cl[Mc[:,0] <= 0.99],Sref_main)

        # For mach numbers close to 1, use an interpolation to avoid intensive calculations
        cd_c[Mc > 0.99] = drag99[Mc > 0.99] + (drag105[Mc > 0.99]-drag99[Mc > 0.99])*(Mc[Mc > 0.99]-0.99)/(1.05-0.99)

        # Use wave drag equations at supersonic values. The cutoff for this function is 1.05
        # Only the supsonic results are returned with nonzero values
        (cd_c_sup,mcc_sup,MDiv_sup,cd_c_l,cd_c_v) = wave_drag(conditions, 
                                                configuration, 
                                                main_fuselage, 
                                                propulsor, 
                                                wing, 
                                                num_engines,k,Sref_main,False)
        
        # assume compressibility drag at .99 is due to volume wave drag
        c_inds = np.logical_and((Mc > 0.99),(Mc < 1.05))     
        cd_c_l[c_inds] = 0. + (cd_c_l_105[c_inds])*(Mc[c_inds]-0.99)/(1.05-0.99)
        cd_c_v[c_inds] = drag99[c_inds] + (cd_c_v_105[c_inds]-drag99[c_inds])*(Mc[c_inds]-0.99)/(1.05-0.99)

        # Incorporate supersonic results into total compressibility drag coefficient
        (cd_c[Mc >= 1.05],mcc[Mc >= 1.05], MDiv[Mc >= 1.05]) = (cd_c_sup[Mc >= 1.05],mcc_sup[Mc >= 1.05],MDiv_sup[Mc >= 1.05])

        # Dump data to conditions
        wing_results = Data(
            compressibility_drag      = cd_c    ,
            volume_wave_drag          = cd_c_v  ,
            lift_wave_drag            = cd_c_l  ,
            crest_critical            = mcc     ,
            divergence_mach           = MDiv    ,
        )
        drag_breakdown.compressible[k] = wing_results        
    
    # Initialize arrays
    mach       = conditions.freestream.mach_number
    prop_drag = np.array([[0.0]] * len(mach))
    fuse_drag = np.array([[0.0]] * len(mach))

    # Fuselage wave drag
    if len(main_fuselage) > 0:
        fuse_wave = wave_drag_body_of_rev(main_fuselage.lengths.total,main_fuselage.effective_diameter/2.0,Sref_main)
        fuse_drag[mach >= .99]  = fuse_wave*(mach[mach>=.99]-.99)/(1.05-.99)
        fuse_drag[mach >= 1.05] = fuse_wave
    else:
        raise ValueError('Main fuselage does not have a total length')

    # Propulsor wave drag	
    Dn                      = propulsor.nacelle_diameter
    Di                      = propulsor.inlet_diameter
    effective_area          = (Dn*Dn-Di*Di)/4.*np.pi
    effective_radius        = np.sqrt(effective_area/np.pi)
    prop_wave               = wave_drag_body_of_rev(propulsor.engine_length,effective_radius,Sref_main)*propulsor.number_of_engines
    prop_drag[mach >= .99]  = prop_wave*(mach[mach>=.99]-.99)/(1.05-.99)
    prop_drag[mach >= 1.05] = prop_wave    
    
    drag_breakdown.compressible[main_fuselage.tag] = fuse_drag
    drag_breakdown.compressible[propulsor.tag] = prop_drag

    # Dump total comp drag
    total_compressibility_drag = 0.0
    total_volume_wave_drag     = 0.0
    total_lift_wave_drag       = 0.0
        
    for k in wings.keys():
        total_compressibility_drag = drag_breakdown.compressible[k].compressibility_drag + total_compressibility_drag
        total_volume_wave_drag     = drag_breakdown.compressible[k].volume_wave_drag + total_volume_wave_drag
        total_lift_wave_drag       = drag_breakdown.compressible[k].lift_wave_drag + total_lift_wave_drag
        
    total_compressibility_drag               = total_compressibility_drag + fuse_drag
    total_compressibility_drag               = total_compressibility_drag + prop_drag
    total_volume_wave_drag                   = total_volume_wave_drag + fuse_drag + prop_drag
    drag_breakdown.compressible.total        = total_compressibility_drag
    drag_breakdown.compressible.total_volume = total_volume_wave_drag
    drag_breakdown.compressible.total_lift   = total_lift_wave_drag

    return total_compressibility_drag


## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
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
        MDiv = np.array([[0.95]] * len(Mc_ii))
        mcc = np.array([[0.93]] * len(Mc_ii))

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
    
    # Change empty format to avoid errors in assignment of returned values
    if np.shape(cd_c) == (0,0):
        cd_c = np.reshape(cd_c,[0,1]) 
        mcc  = np.reshape(mcc,[0,1]) 
        MDiv = np.reshape(MDiv,[0,1]) 

    return (cd_c,mcc,MDiv)

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def wave_drag(conditions,configuration,main_fuselage,propulsor,wing,num_engines,k,Sref_main,flag105):
    """Use wave drag to determine compressibility drag for supersonic speeds

    Assumptions:
    Basic fit

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    conditions.freestream.mach_number            [Unitless]
    configuration
    main_fuselage (unused)
    propulsor     (unused)
    wing.areas.reference                         [m^2]
    num_engines   (unused)
    k             (tag for wing)                 [String]
    Sref_main (main reference area)              [m^2]
    flag105   (check if calcs are for Mach 1.05) [Boolean]
    

    Outputs:
    cd_c                    [Unitless]
    mcc                     [Unitless]
    MDiv                    [Unitless]

    Properties Used:
    N/A
    """    

    # Unpack mach number
    mach       = conditions.freestream.mach_number

    # Create a copy that can be modified
    Mc         = copy.copy(mach)

    # This flag is for the interpolation mode
    if flag105 is True:
        # Change conditions and short hand for calculations
        conditions.freestream.mach_number = np.array([[1.05]] * len(Mc))
        mach = conditions.freestream.mach_number

    # Initalize cd arrays
    cd_c   = np.array([[0.0]] * len(mach))
    cd_c_l = np.array([[0.0]] * len(mach)) # lift wave drag
    cd_c_v = np.array([[0.0]] * len(mach)) # vol wave drag

    # Calculate wing values at all mach numbers
    # Note that these functions arrange the supersonic values at the beginning of the array
    cd_lift_wave = wave_drag_lift(conditions,configuration,wing)
    cd_volume_wave = wave_drag_volume(conditions,configuration,wing)

    # Pack supersonic results into correct elements
    cd_c[mach >= 1.05] = cd_lift_wave[0:len(mach[mach >= 1.05]),0] + cd_volume_wave[0:len(mach[mach >= 1.05]),0]
    cd_c_l[mach >= 1.05] = cd_lift_wave[0:len(mach[mach >= 1.05]),0]
    cd_c_v[mach >= 1.05] = cd_volume_wave[0:len(mach[mach >= 1.05]),0]

    # Convert coefficient to full aircraft value
    cd_c = cd_c*wing.areas.reference/Sref_main
    cd_c_l = cd_c_l*wing.areas.reference/Sref_main
    cd_c_v = cd_c_v*wing.areas.reference/Sref_main

    # Include fuselage and propulsors for one iteration

    # Return dummy values for mcc and MDiv
    mcc = np.array([[0.0]] * len(mach))
    MDiv = np.array([[0.0]] * len(mach))

    # Reset mach number to real values
    conditions.freestream.mach_number = Mc

    return (cd_c,mcc,MDiv,cd_c_l,cd_c_v)

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def wave_drag_body_of_rev(total_length,Rmax,Sref):
    """Use wave drag to determine compressibility drag a body of revolution

    Assumptions:
    Corrected Sear-Haack body 

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    total_length                    [m]
    Rmax (max radius)               [m]
    Sref (main wing reference area) [m^2]

    Outputs:
    wave_drag_body_of_rev*1.15      [Unitless]

    Properties Used:
    N/A
    """    


    # Computations - takes drag of Sears-Haack and use wing reference area for CD
    wave_drag_body_of_rev = (9.0*(np.pi)**3.0*Rmax**4.0/(4.0*total_length**2.0))/(0.5*Sref)  

    # Apply correction for imperfect body
    wave_drag_body_of_rev_corrected = wave_drag_body_of_rev*1.15
    
    return wave_drag_body_of_rev_corrected
