# compressibility_drag_wing.py
# 
# Created:  
# Modified: 7/2014  Tim MacDonald        

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Attributes.Results.Result import Result
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)
from SUAVE.Methods.Aerodynamics.Drag.Correlations import \
     wave_drag_lift, wave_drag_volume, wave_drag_body_of_rev

from wave_drag_lift import wave_drag_lift
from wave_drag_volume import wave_drag_volume
from wave_drag_fuselage import wave_drag_fuselage
from wave_drag_propulsor import wave_drag_propulsor

# python imports
import os, sys, shutil
import copy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def compressibility_drag_total_supersonic(conditions,configuration,geometry):
    """ SUAVE.Methods.compressibility_drag_wing(conditions,configuration,geometry)
        computes the induced drag associated with a wing 
        
        Inputs:
        
        Outputs:
        
        Assumptions:
            based on a set of fits
            
    """

    # unpack
    wings      = geometry.wings
    fuselages   = geometry.fuselages
    propulsor = geometry.Propulsors[0]
    #print geometry
    #w = input("Press any key to continue")
    #try:
        #wing_lifts = conditions.aerodynamics.lift_breakdown.compressible_wings # currently the total aircraft lift
    #except:
    wing_lifts = conditions.aerodynamics.lift_coefficient
    mach       = conditions.freestream.mach_number
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    # start result
    total_compressibility_drag = 0.0
    drag_breakdown.compressible = Result()

    # go go go
    for i_wing, wing, in enumerate(wings.values()):
        
        # unpack wing
        t_c_w   = wing.t_c
        sweep_w = wing.sweep
        Mc = copy.copy(mach)
        
        cd_c = np.array([[0.0]] * len(Mc))
        mcc = np.array([[0.0]] * len(Mc))
        MDiv = np.array([[0.0]] * len(Mc))
        
        if i_wing == 0:
            cl_w = wing_lifts
            Sref_main = wing.sref
        else:
            cl_w = 0          
        
        for ii in range(len(Mc)):          
            
            if Mc[ii] > 0.99 and Mc[ii] < 1.05:
                Mc[ii] = 0.99
            
            
            if Mc[ii] <= 1.05:
        
            
                # get effective Cl and sweep
                tc = t_c_w /(np.cos(sweep_w))
                try:
                    cl = cl_w[ii] / (np.cos(sweep_w))**2
                    # compressibility drag based on regressed fits from AA241
                    mcc_cos_ws = 0.922321524499352       \
                               - 1.153885166170620*tc    \
                               - 0.304541067183461*cl    \
                               + 0.332881324404729*tc**2 \
                               + 0.467317361111105*tc*cl \
                               + 0.087490431201549*cl**2
                        
                    # crest-critical mach number, corrected for wing sweep
                    mcc[ii] = mcc_cos_ws / np.cos(sweep_w)
                    
                    # divergence mach number
                    MDiv[ii] = mcc[ii] * ( 1.02 + 0.08*(1 - np.cos(sweep_w)) )
                    MDiv[ii] = 0.95
                    mcc[ii] = MDiv[ii] / ( 1.02 + 0.08*(1 - np.cos(sweep_w)) )
                    mcc[ii] = 0.93
                    
                    # divergence ratio
                    mo_mc = Mc[ii]/mcc[ii]
                    
                    # compressibility correlation, Shevell
                    dcdc_cos3g = 0.0019*mo_mc**14.641
                    
                    # compressibility drag

                    if Mc[ii] > 0.95 and cl > 0.1:
                        h = 0
                    
                    cd_c[ii] = dcdc_cos3g #* (np.cos(sweep_w))**3                    
                except TypeError:
                    cl = 0
                    mcc_cos_ws = 0.922321524499352 - 1.153885166170620*tc    \
                                                   + 0.332881324404729*tc**2      
                    mcc[ii] = mcc_cos_ws / np.cos(sweep_w)
                    MDiv[ii] = mcc[ii] * ( 1.02 + 0.08*(1 - np.cos(sweep_w)) )
                    mo_mc = Mc[ii]/mcc[ii]
                    dcdc_cos3g = 0.0019*mo_mc**14.641
                    cd_c[ii] = dcdc_cos3g * (np.cos(sweep_w))**3                    
            

                
            else:
                cd_lift_wave = wave_drag_lift(conditions,configuration,wing)
                cd_volume_wave = wave_drag_volume(conditions,configuration,wing)
                cd_c[ii] = cd_lift_wave[ii] + cd_volume_wave[ii]
                if i_wing != 0:
                    cd_c[ii] = cd_c[ii]*wing.sref/Sref_main
                if i_wing == 0:
                    cd_c[ii] = cd_c[ii] + wave_drag_fuselage(conditions,configuration,fuselages.Fuselage,wing)
                    cd_c[ii] = cd_c[ii] + wave_drag_propulsor(conditions,configuration,propulsor,wing)*propulsor.no_of_engines
                tc = 0
                #print("oh no wave drag")
                sweep_w = 0
                mcc = 0
                MDiv = 0
        
            
            # increment
            #total_compressibility_drag += cd_c  ## todo when there is a lift break down by wing
            
            # dump data to conditions
        wing_results = Result(
            compressibility_drag      = cd_c    ,
            thickness_to_chord        = tc      , 
            wing_sweep                = sweep_w , 
            crest_critical            = mcc     ,
            divergence_mach           = MDiv    ,
        )
        drag_breakdown.compressible[wing.tag] = wing_results
        #print cd_c[1]
    #: for each wing
    
    # dump total comp drag
    total_compressibility_drag = 0.0
    for jj in range(1,i_wing+2):
        total_compressibility_drag = drag_breakdown.compressible[jj].compressibility_drag + total_compressibility_drag
    #print total_compressibility_drag[1]
    #w = input("Press any key")
    drag_breakdown.compressible.total = total_compressibility_drag

    return total_compressibility_drag
