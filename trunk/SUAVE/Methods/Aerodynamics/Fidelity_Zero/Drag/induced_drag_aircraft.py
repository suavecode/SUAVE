
# induced_drag_aircraft.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified:     2016, A. Variyar
          

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Analyses import Results

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

#def induced_drag_aircraft(conditions,configuration,geometry):
def induced_drag_aircraft(state,settings,geometry):
    """ SUAVE.Methods.induced_drag_aircraft(conditions,configuration,geometry)
        computes the induced drag associated with a wing 
        
        Inputs:
        
        Outputs:
        
        Assumptions:
            based on a set of fits
            
    """

    # unpack inputs
    conditions = state.conditions
    configuration = settings
    
    
    # unpack common variables
    aircraft_lift = conditions.aerodynamics.lift_coefficient
    ar            = geometry.wings['main_wing'].aspect_ratio
    CDp           = state.conditions.aerodynamics.drag_breakdown.parasite.total
    wingleth      = 0.   #winglet length #geometry.wings['main_wing'].wingleth
    tc            = geometry.wings['main_wing'].thickness_to_chord
    
    
    # Base SUAVE implementation --------
    #e             = configuration.oswald_efficiency_factor
    #K             = configuration.viscous_lift_dependent_drag_factor 
    #wing_e        = geometry.wings['main_wing'].span_efficiency 
    #if e == None:
        #e = 1/((1/wing_e)+np.pi*ar*K*CDp)
    
    
    # Start the result
    total_induced_drag = 0.0
    
    if geometry.fuselages.has_key('fuselage'):
        fusewidth = geometry.fuselages['fuselage'].effective_diameter
        fusefraction = fusewidth/geometry.wings['main_wing'].spans.projected
        if(fusefraction > 0.5):
            fusefraction = 0.5 
        #s = 1.0 - 0.041599999999999998 * fusefraction - 1.7986 * fusefraction * fusefraction
        s = 1.0 - 2.0 * fusefraction * fusefraction
    else:
        s = 1.
    wing_e = s
    

    
    cossw = np.cos(geometry.wings['main_wing'].sweeps.quarter_chord)
    
    
    # Raymer effect of winglets
    arweff = ar * (1.0 + 1.9 * wingleth)    
    
    K_visc = 0.38/(cossw * cossw)
    
    # e_tot modified/shevell ------
    #e_tot = 1.0/((1.0/wing_e)) #+ np.pi*ar*K_visc*CDp)
    e_tot = 1.0/((1.0/wing_e)+ np.pi*arweff*K_visc*CDp)
        
    #arweff = ar * (1.0 + 0.80000000000000004 * wingleth)**2.0
    #arweff = ar
    
    ## wing_e raymer ----------
    #e_tot = 4.61*(1.0-0.045*arweff**0.68)*(cossw**0.15) - 3.1 
    
    ## e_tot Grosu ----------
    #e_tot = 1.0/(1.08 + 0.028*np.pi*arweff*tc/aircraft_lift**2.0)
    
    ## e_tot Howe ----------
    #lamda         = geometry.wings['main_wing'].taper
    #mach_number   = conditions.freestream.mach_number
    #Ne            = 0.0 # Number of engines    
    #fl = 0.005*(1.0+1.5*(lamda-0.6)**2.0)
    #t1 = 1.0 + 0.12*mach_number**2.0
    #t2_1 = (0.142 + fl*arweff*(10.0*tc)**0.33)/(cossw*cossw)
    #t2_2 = 0.1*(3.0*Ne + 1.0)/((4.0 + arweff)**0.8)
    #e_tot = 1.0/(t1*(1.0 + t2_1 + t2_2))

    
    #total_induced_drag = aircraft_lift**2.0 / (np.pi*ar*wing_e) + aircraft_lift**2.0*CDp*K_visc
    #total_induced_drag = aircraft_lift**2.0 / (np.pi*ar*e_tot)
    total_induced_drag = aircraft_lift**2.0 / (np.pi*arweff*e_tot)
    
    
    
    ##given e
    #e_tot = geometry.wings['main_wing'].span_efficiency
    #wing_e = e_tot
    #total_induced_drag = aircraft_lift**2.0 / (np.pi*ar*e_tot)
    
        
    # Store data
    conditions.aerodynamics.drag_breakdown.induced = Results(
        total             = total_induced_drag ,
        aspect_ratio      = ar                 ,
        e_total           = e_tot            ,
        e_inviscid        = wing_e          ,
    )
    
    # done!

    return total_induced_drag