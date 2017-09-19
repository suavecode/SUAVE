## @ingroup Methods-Propulsion
# fm_id.py
# 
# Created:  ### ####, SUAVE Team
# Modified: Feb 2016, E. Botero

import numpy as np
from scipy.optimize import fsolve


# ----------------------------------------------------------------------
#  fm_id
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion
def exit_Mach_shock(area_ratio, gamma, Pt_out, P0):
    """ Determines the output Mach number of a nozzle with a normal shock taking
    place inside of it, through pressure ratio between the nozzle stagnation
    pressure and the freestream pressure"""
    func = lambda Me : (Pt_out/P0)*(1/area_ratio)-(((gamma+1)/2)**((gamma+1)/(2*(gamma-1))))*Me*((1+(gamma-1)/2*Me**2)**0.5)

    #Initializing the array
    Me_initial_guess = 1.0*Pt_out/Pt_out
        
    i_sol = Me_initial_guess < 10.0
    
    Me_initial_guess[i_sol] = 0.1

    Me = fsolve(func,Me_initial_guess)
        
    return Me
        
        
def mach_area(area_ratio, gamma, subsonic):
    """ Returns the Mach number given an area ratio and isentropic conditions"""
    func = lambda Me : area_ratio**2 - ((1/Me)**2)*(((2/(gamma+1))*(1+((gamma-1)/2)*Me**2))**((gamma+1)/((gamma-1))))
    if subsonic:
        Me_initial_guess = 0.01
    else:
        Me_initial_guess = 2.0         
        
    Me = fsolve(func,Me_initial_guess)

    return Me

    
def normal_shock(M1, gamma):  
    """ Returns the Mach number after normal shock"""
    M2 = np.sqrt((((gamma-1)*M1**2)+2)/(2*gamma*M1**2-(gamma-1)))
    
    return M2
    
def pressure_ratio_isentropic(area_ratio, gamma, subsonic):
    """ Determines the pressure ratio for isentropic flow throughout the entire
    nozzle"""
    #yields pressure ratio for isentropic conditions given area ratio
    Me = mach_area(area_ratio,gamma, subsonic)
    
    pr_isentropic = (1+((gamma-1)/2)*Me**2)**(-gamma/(gamma-1))
    
    return pr_isentropic

def pressure_ratio_shock_in_nozzle(area_ratio, gamma):
    """ Determines the lower value of pressure ratio responsible for a 
    normal shock taking place inside the nozzle"""
    #yields maximium pressure ratio where shock takes place inside the nozzle, given area ratio
    Me = mach_area(area_ratio, gamma, False)
    M2 = normal_shock(Me, gamma)
    
    pr_shock_in_nozzle = ((area_ratio)*(((gamma+1)/2)**((gamma+1)/(2*(gamma-1))))*M2*((1+((gamma-1)/2)*M2**2)**0.5))**(-1)
    return pr_shock_in_nozzle