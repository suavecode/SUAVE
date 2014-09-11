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
        Mc = copy.copy(mach)
        
        cd_c = np.array([[0.0]] * len(Mc))
        mcc = np.array([[0.0]] * len(Mc))
        MDiv = np.array([[0.0]] * len(Mc))
        
        if i_wing == 0:
            Sref_main = wing.sref  
            
        main_fuselage = fuselages.Fuselage
        num_engines = propulsor.no_of_engines
        
        drag99 = drag_div(0.99)
        (drag105,a,b) = wave_drag(conditions, 
                            configuration, 
                            main_fuselage, 
                            propulsor, 
                            wing, 
                            num_engines,1,i_wing,Sref_main,True)
        
        for ii in range(len(Mc)):          
            
            
            if Mc[ii] <= 0.99:
                cd_c[ii] = drag_div(Mc[ii])
            
            elif Mc[ii] > 0.99 and Mc[ii] < 1.05:
                Mc[ii] = 0.99
                cd_c[ii] = drag99 + (drag105-drag99)*(Mc[ii]-0.99)/(1.05-0.99)
                           
                
            else:
                
                (cd_c[ii],mcc[ii],MDiv[ii]) = wave_drag(conditions, 
                                                       configuration, 
                                                       main_fuselage, 
                                                       propulsor, 
                                                       wing, 
                                                       num_engines,ii,i_wing,Sref_main,False)
        
            
            # increment
            
            # dump data to conditions
        wing_results = Result(
            compressibility_drag      = cd_c    ,
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
    
    #if max(total_compressibility_drag) > 0.014:
        #h = 0

    return total_compressibility_drag


def drag_div(Mc_ii):

    # divergence mach number
    MDiv = 0.95
    mcc = 0.93
    
    # divergence ratio
    mo_mc = Mc_ii/mcc
    
    # compressibility correlation, Shevell
    dcdc_cos3g = 0.0019*mo_mc**14.641
    
    # compressibility drag
    
    cd_c = dcdc_cos3g 
    
    return cd_c

def wave_drag(conditions,configuration,main_fuselage,propulsor,wing,num_engines,ii,i_wing,Sref_main,flag105):

    mach       = conditions.freestream.mach_number
    Mc         = copy.copy(mach)
    if flag105 is True:
        conditions.freestream.mach_number = np.array([[1.05]] * len(Mc))
    
    cd_c = np.array([[0.0]] * len(Mc))
    cd_lift_wave = wave_drag_lift(conditions,configuration,wing)
    cd_volume_wave = wave_drag_volume(conditions,configuration,wing)
    cd_c[ii] = cd_lift_wave[ii] + cd_volume_wave[ii]
    if i_wing != 0:
        cd_c[ii] = cd_c[ii]*wing.sref/Sref_main
    if i_wing == 0:
        fuse_drag = wave_drag_body_of_rev(main_fuselage.length_total,main_fuselage.Deff/2.0,Sref_main)
        prop_drag = wave_drag_body_of_rev(propulsor.engine_length,propulsor.nacelle_dia,Sref_main)*propulsor.no_of_engines
        cd_c[ii] = cd_c[ii] + wave_drag_body_of_rev(main_fuselage.length_total,main_fuselage.Deff/2.0,Sref_main)
        cd_c[ii] = cd_c[ii] + wave_drag_body_of_rev(propulsor.engine_length,propulsor.nacelle_dia/2.0,Sref_main)*propulsor.no_of_engines
    mcc = 0
    MDiv = 0

    conditions.freestream.mach_number = Mc
    
    return (cd_c[ii],mcc,MDiv)

def wave_drag_body_of_rev(total_length,Rmax,Sref):

    # Computations - takes drag of Sears-Haack and use wing reference area for CD
    wave_drag_body_of_rev = (9.0*(np.pi)**3.0*Rmax**4.0/(4.0*total_length**2.0))/(0.5*Sref)  
    
    return wave_drag_body_of_rev*1.15