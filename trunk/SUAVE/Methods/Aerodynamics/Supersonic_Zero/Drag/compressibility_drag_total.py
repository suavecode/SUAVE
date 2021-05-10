## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# compressibility_drag_total.py
# 
# Created:  Jan 2019, T. MacDonald
# Modified: Jan 2020, T. MacDonald
#           May 2021, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Data

from .wave_drag_lift import wave_drag_lift
from .wave_drag_volume_raymer import wave_drag_volume_raymer
from .wave_drag_volume_sears_haack import wave_drag_volume_sears_haack
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender
from SUAVE.Components.Wings import Main_Wing

import copy

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compressibility Drag Total
# ----------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def compressibility_drag_total(state,settings,geometry):
    """Computes compressibility drag for full aircraft including volume drag

    Assumptions:
    None

    Source:
    N/A

    Inputs:   
    settings.
      begin_drag_rise_mach_number                                    [Unitless]
      end_drag_rise_mach_number                                      [Unitless]
      peak_mach_number                                               [Unitless]
      transonic_drag_multiplier                                      [Unitless]
      volume_wave_drag_scaling                                       [Unitless]
    state.conditions.aerodynamics.lift_breakdown.compressible_wings  [Unitless]
    state.conditions.freestream.mach_number                          [Unitless]
    geometry.maximum_cross_sectional_area                            [m^2] (used in subfunctions)
    geometry.total_length                                            [m]   (used in subfunctions)
    geometry.reference_area                                          [m^2]
    geometry.wings                             

    Outputs:
    total_compressibility_drag                                       [Unitless]

    Properties Used:
    N/A
    """     

    # Unpack
    conditions       = state.conditions
    configuration    = settings
    low_mach_cutoff  = settings.begin_drag_rise_mach_number
    high_mach_cutoff = settings.end_drag_rise_mach_number
    peak_mach        = settings.peak_mach_number
    peak_factor      = settings.transonic_drag_multiplier
    scaling_factor   = settings.volume_wave_drag_scaling
    
    if settings.wave_drag_type == 'Raymer':
        wave_drag_volume = wave_drag_volume_raymer
    elif settings.wave_drag_type == 'Sears-Haack':
        wave_drag_volume = wave_drag_volume_sears_haack
    else:
        raise NotImplementedError    
    
    if settings.cross_sectional_area_calculation_type != 'Fixed':
        raise NotImplementedError
    
    wings          = geometry.wings
    Mc             = conditions.freestream.mach_number
    drag_breakdown = conditions.aerodynamics.drag_breakdown

    # Initialize result
    drag_breakdown.compressible = Data()
    
    # Use vehicle reference area for drag coefficients
    Sref_main = geometry.reference_area
    

    low_cutoff_volume_total  = np.zeros_like(Mc)
    high_cutoff_volume_total = np.zeros_like(Mc)
    
    # Get the lift coefficient
    cl = conditions.aerodynamics.lift_breakdown.compressible_wings    
    
    for wing in geometry.wings:
        low_cutoff_volume_total += drag_div(low_mach_cutoff*np.ones([1]), wing, cl[wing.tag], Sref_main)[0]
    high_cutoff_volume_total = wave_drag_volume(geometry,low_mach_cutoff*np.ones([1]),scaling_factor)
    
    peak_volume_total = high_cutoff_volume_total*peak_factor
    
    # fit the drag rise using piecewise parabolas y = a*(x-x_peak)**2+y_peak
    
    # subsonic side
    a1 = (low_cutoff_volume_total-peak_volume_total)/(low_mach_cutoff-peak_mach)/(low_mach_cutoff-peak_mach)
    
    # supersonic side
    a2 = (high_cutoff_volume_total-peak_volume_total)/(high_mach_cutoff-peak_mach)/(high_mach_cutoff-peak_mach) 
    
    def CD_v_para(M,a_vertex): # parabolic approximation of drag rise in the transonic region
        ret = a_vertex*(M-peak_mach)*(M-peak_mach)+peak_volume_total
        ret = ret.reshape(np.shape(M))
        return ret
    
    # Shorten cubic Hermite spline
    sub_spline = Cubic_Spline_Blender(low_mach_cutoff, peak_mach-(peak_mach-low_mach_cutoff)*3/4)
    sup_spline = Cubic_Spline_Blender(peak_mach,high_mach_cutoff)
    sub_h00 = lambda M:sub_spline.compute(M)
    sup_h00 = lambda M:sup_spline.compute(M)
    
    cd_c_v_base = np.zeros_like(Mc)
    
    low_inds = Mc[:,0]<peak_mach
    hi_inds  = Mc[:,0]>=peak_mach
    
    for wing in geometry.wings:
        cd_c_v_base[low_inds] += drag_div(Mc[low_inds], wing, cl[wing.tag][low_inds], Sref_main)[0]
    cd_c_v_base[Mc>=peak_mach] = wave_drag_volume(geometry, Mc[Mc>=peak_mach], scaling_factor)
    
    cd_c_l_base = lift_wave_drag(conditions, configuration, geometry.wings.main_wing, Sref_main)
    
    cd_c_v = np.zeros_like(Mc)
    
    cd_c_v[low_inds] = cd_c_v_base[low_inds]*(sub_h00(Mc[low_inds])) + CD_v_para(Mc[low_inds],a1[low_inds])*(1-sub_h00(Mc[low_inds]))
    cd_c_v[hi_inds]  = CD_v_para(Mc[hi_inds],a2)*(sup_h00(Mc[hi_inds])) + cd_c_v_base[hi_inds]*(1-sup_h00(Mc[hi_inds]))

    if peak_mach<1.01:
        print('Warning: a peak mach number of less than 1.01 will cause a small discontinuity in lift wave drag')
    cd_c_l           = cd_c_l_base*(1-sup_h00(Mc))
    
    cd_c = cd_c_v + cd_c_l

    
    # Save drag breakdown

    drag_breakdown.compressible.total        = cd_c
    drag_breakdown.compressible.total_volume = cd_c_v
    drag_breakdown.compressible.total_lift   = cd_c_l
    

    return cd_c

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def lift_wave_drag(conditions,configuration,wing,Sref_main):
    """Determine lift wave drag for supersonic speeds

    Assumptions:
    Basic fit

    Source:
    http://aerodesign.stanford.edu/aircraftdesign/aircraftdesign.html (Stanford AA241 A/B Course Notes)

    Inputs:
    conditions.freestream.mach_number [-]
    configuration                     (passed to another function)
    wing.areas.reference              [m^2]
    Sref_main                         [m^2] Main reference area

    Outputs:
    cd_c_l                            [-] Wave drag CD due to lift

    Properties Used:
    N/A
    """       
    # Use wave drag to determine compressibility drag for supersonic speeds

    # Unpack mach number
    mach       = conditions.freestream.mach_number

    # Initalize cd arrays
    cd_c_l = np.array([[0.0]] * len(mach)) # lift wave drag

    # Calculate wing values at all mach numbers
    # Note that these functions arrange the supersonic values at the beginning of the array
    cd_lift_wave = wave_drag_lift(conditions,configuration,wing)

    # Pack supersonic results into correct elements
    cd_c_l[mach >= 1.01] = cd_lift_wave[0:len(mach[mach >= 1.01]),0]

    # Convert coefficient to full aircraft value
    cd_c_l = cd_c_l*wing.areas.reference/Sref_main

    return cd_c_l

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def drag_div(Mc_ii,wing,cl,Sref_main):
    """Use drag divergence mach number to determine drag for subsonic speeds

    Assumptions:
    Basic fit, subsonic

    Source:
    http://aerodesign.stanford.edu/aircraftdesign/aircraftdesign.html (Stanford AA241 A/B Course Notes)
    Concorde data can be found in "Supersonic drag reduction technology in the scaled supersonic 
    experimental airplane project by JAXA" by Kenji Yoshida

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

        # Divergence mach number, fit to Concorde data
        MDiv = np.array([[0.98]] * len(Mc_ii))
        mcc  = np.array([[0.95]] * len(Mc_ii))

    else:
        # Unpack wing
        t_c_w   = wing.thickness_to_chord
        sweep_w = wing.sweeps.quarter_chord

        # Check if this is the main wing, other wings are assumed to have no lift
        if isinstance(wing, Main_Wing):
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
