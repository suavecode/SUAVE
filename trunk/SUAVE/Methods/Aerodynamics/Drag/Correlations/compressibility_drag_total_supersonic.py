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
    """ SUAVE.Methods.compressibility_drag_total_supersonic(conditions,configuration,geometry)
        computes the compressibility drag on a full aircraft
        
        Inputs:
            wings
	    fuselages
	    propulsors
	    freestream conditions
        
        Outputs:
	    compressibility drag coefficient
        
        Assumptions:
            drag is only calculated for the wings, main fuselage, and propulsors
            
    """

    # unpack
    wings      = geometry.Wings
    fuselages   = geometry.Fuselages
    propulsor = geometry.Propulsors[0]

    mach       = conditions.freestream.mach_number
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    # start result
    total_compressibility_drag = 0.0
    drag_breakdown.compressible = Result()

    # go go go
    for i_wing, wing, in enumerate(wings.values()):
        
        # unpack wing
        Mc = copy.copy(mach)
        
        cd_c = np.array([[0.0]] * len(Mc))
        mcc = np.array([[0.0]] * len(Mc))
        MDiv = np.array([[0.0]] * len(Mc))
        
        if i_wing == 0:
            Sref_main = wing.sref  
            
        if len(geometry.Fuselages) > 0:
            main_fuselage = fuselages.Fuselage
        else:
            main_fuselage = geometry.Fuselages
        num_engines = propulsor.no_of_engines
        
        wings_lift = conditions.aerodynamics.lift_breakdown.compressible_wings
        cl = wings_lift
        (drag99,a,b) = drag_div(np.array([[0.99]] * len(Mc)),wing,i_wing,cl)
        (drag105,a,b) = wave_drag(conditions, 
                            configuration, 
                            main_fuselage, 
                            propulsor, 
                            wing, 
                            num_engines,i_wing,Sref_main,True)
        
	(cd_c[Mc <= 0.99],mcc[Mc <= 0.99], MDiv[Mc <= 0.99]) = drag_div(Mc[Mc <= 0.99],wing,i_wing,cl[Mc <= 0.99])
	
	
	if type(drag99) is float:
		cd_c[Mc > 0.99] = drag99 + (drag105-drag99)*(Mc[Mc > 0.99]-0.99)/(1.05-0.99)
	else:
		cd_c[Mc > 0.99] = drag99[Mc > 0.99] + (drag105[Mc > 0.99]-drag99[Mc > 0.99])*(Mc[Mc > 0.99]-0.99)/(1.05-0.99)
	(cd_c_sup,mcc_sup,MDiv_sup) = wave_drag(conditions, 
                                               configuration, 
                                               main_fuselage, 
                                               propulsor, 
                                               wing, 
                                               num_engines,i_wing,Sref_main,False)
	(cd_c[Mc >= 1.05],mcc[Mc >= 1.05], MDiv[Mc >= 1.05]) = (cd_c_sup[Mc >= 1.05],mcc_sup[Mc >= 1.05],MDiv_sup[Mc >= 1.05])
		
		

        
            
	# increment
            
	# dump data to conditions
        wing_results = Result(
            compressibility_drag      = cd_c    ,
            crest_critical            = mcc     ,
            divergence_mach           = MDiv    ,
        )
        drag_breakdown.compressible[wing.tag] = wing_results

    #: for each wing
    
    # dump total comp drag
    total_compressibility_drag = 0.0
    for jj in range(1,i_wing+2):
        total_compressibility_drag = drag_breakdown.compressible[jj].compressibility_drag + total_compressibility_drag
    drag_breakdown.compressible.total = total_compressibility_drag

    return total_compressibility_drag


def drag_div(Mc_ii,wing,i_wing,cl):

    if wing.highmach is True:
        
        # divergence mach number
        MDiv = np.array([0.95] * len(Mc_ii))
        mcc = np.array([0.93] * len(Mc_ii))
        
    else:
        # unpack wing
        t_c_w   = wing.t_c
        sweep_w = wing.sweep
        
        if i_wing == 0:
            cl_w = cl
        else:
            cl_w = 0
    
        # get effective Cl and sweep
        tc = t_c_w /(np.cos(sweep_w))
        cl = cl_w / (np.cos(sweep_w))**2
    
        # compressibility drag based on regressed fits from AA241
        mcc_cos_ws = 0.922321524499352       \
                   - 1.153885166170620*tc    \
                   - 0.304541067183461*cl    \
                   + 0.332881324404729*tc**2 \
                   + 0.467317361111105*tc*cl \
                   + 0.087490431201549*cl**2
            
        # crest-critical mach number, corrected for wing sweep
        mcc = mcc_cos_ws / np.cos(sweep_w)
        
        # divergence mach number
        MDiv = mcc * ( 1.02 + 0.08*(1 - np.cos(sweep_w)) )        
    
    # divergence ratio
    mo_mc = Mc_ii/mcc
    
    # compressibility correlation, Shevell
    dcdc_cos3g = 0.0019*mo_mc**14.641
    
    # compressibility drag
    
    if wing.highmach is True:
	cd_c = dcdc_cos3g
    else:
	cd_c = dcdc_cos3g * (np.cos(sweep_w))**3
    
    return (cd_c,mcc,MDiv)

def wave_drag(conditions,configuration,main_fuselage,propulsor,wing,num_engines,i_wing,Sref_main,flag105):

    mach       = conditions.freestream.mach_number
    Mc         = copy.copy(mach)
    if flag105 is True:
        conditions.freestream.mach_number = np.array([[1.05]] * len(Mc))
	mach = conditions.freestream.mach_number
    
    
    cd_c = np.array([[0.0]] * len(mach))
    cd_lift_wave = wave_drag_lift(conditions,configuration,wing)
    cd_volume_wave = wave_drag_volume(conditions,configuration,wing)
    cd_c[mach >= 1.05] = cd_lift_wave[0:len(mach[mach >= 1.05]),0] + cd_volume_wave[0:len(mach[mach >= 1.05]),0]
    if i_wing != 0:
        cd_c = cd_c*wing.sref/Sref_main
    if i_wing == 0:
	prop_drag = np.array([[0.0]] * len(mach))
	fuse_drag = np.array([[0.0]] * len(mach))
        if len(main_fuselage) > 0:
            fuse_drag[mach >= 1.05] = wave_drag_body_of_rev(main_fuselage.length_total,main_fuselage.Deff/2.0,Sref_main)
        else:
            fuse_drag[mach >= 1.05] = 0.0
        prop_drag[mach >= 1.05] = wave_drag_body_of_rev(propulsor.engine_length,propulsor.nacelle_dia/2.0,Sref_main)*propulsor.no_of_engines
        cd_c[mach >= 1.05] = cd_c[mach >= 1.05] + fuse_drag[mach >= 1.05]
        cd_c[mach >= 1.05] = cd_c[mach >= 1.05] + prop_drag[mach >= 1.05]
    mcc = np.array([[0.0]] * len(mach))
    MDiv = np.array([[0.0]] * len(mach))

    if len(Mc[mach < .4]) > 0:
	a = 0

    conditions.freestream.mach_number = Mc
    
    return (cd_c,mcc,MDiv)

def wave_drag_body_of_rev(total_length,Rmax,Sref):
    if ((total_length == 0.0) or (Rmax == 0.0) or (Sref == 0.0)):
	a = 0

    # Computations - takes drag of Sears-Haack and use wing reference area for CD
    wave_drag_body_of_rev = (9.0*(np.pi)**3.0*Rmax**4.0/(4.0*total_length**2.0))/(0.5*Sref)  
    
    return wave_drag_body_of_rev*1.15