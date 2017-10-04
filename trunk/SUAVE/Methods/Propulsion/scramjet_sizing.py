## @ingroup Methods-Propulsion
# fm_id.py
# 
# Created:  ### ####, SUAVE Team
# Modified: Feb 2016, E. Botero
import SUAVE

from scipy.optimize import fsolve
import numpy as np
from SUAVE.Core import Data
from SUAVE.Methods.Propulsion.oblique_shock import theta_beta_mach, oblique_shock_relation

# ----------------------------------------------------------------------
#  fm_id
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion  
    
def inlet_conditions(M_entry,gamma, nbr_oblique_shock,theta):
    """
    Function that takes the Mach, gamma, number of expected oblique shocks
    and wedge angle of the inlet. It outputs the flow properties after interacting
    with the inlet ; it verifies if all oblique shocks actually take place depending
    on entry conditions
    
    Inputs:
    M_entry                 [dimensionless]
    gamma                   [dimensionless]
    nbr_oblique_shock       [dimensionless]
    theta                   [rad]
    
    
    Outputs:
    T_ratio                 [dimensionless]
    Pt_ratio                [dimensionless]
   
    
    """
    shock = 0
    T_ratio = 1.
    P_ratio = 1.
    Pt_ratio = 1.
    
    while shock < nbr_oblique_shock :
        #-- Enter n-th oblique shock
            
        beta = theta_beta_mach(M_entry, theta, gamma, 1)
        M1, Tr, Pr, Ptr = oblique_shock_relation(M_entry,gamma, theta, beta)    
        T_ratio = T_ratio*(1/Tr)
        P_ratio = P_ratio*(1/Pr)
        Pt_ratio = Pt_ratio*Ptr
        M_entry = M1
        shock = shock +1
        
#        if np.any(M1 <= 1.0) :
#            break
    
    return T_ratio, Pt_ratio

def scramjet_sizing(scramjet,mach_number = None, altitude = None, delta_isa = 0, conditions = None):  
    """ create and evaluate a gas turbine network
    """    
    
    #Unpack components
    #check if altitude is passed or conditions is passed
    if(conditions):
        #use conditions
        pass
        
    else:
        #check if mach number and temperature are passed
        if(mach_number==None or altitude==None):
            
            #raise an error
            raise NameError('The sizing conditions require an altitude and a Mach number')
        
        else:
            #call the atmospheric model to get the conditions at the specified altitude
            atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
            atmo_data = atmosphere.compute_values(altitude,delta_isa)

            p   = atmo_data.pressure          
            T   = atmo_data.temperature       
            rho = atmo_data.density          
            a   = atmo_data.speed_of_sound    
            mu  = atmo_data.dynamic_viscosity   
        
            # setup conditions
            conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()            
        
            # freestream conditions    
            conditions.freestream.altitude           = np.atleast_1d(altitude)
            conditions.freestream.mach_number        = np.atleast_1d(mach_number)
            conditions.freestream.pressure           = np.atleast_1d(p)
            conditions.freestream.temperature        = np.atleast_1d(T)
            conditions.freestream.density            = np.atleast_1d(rho)
            conditions.freestream.dynamic_viscosity  = np.atleast_1d(mu)
            conditions.freestream.gravity            = np.atleast_1d(9.81)
            conditions.freestream.gamma              = np.atleast_1d(1.4)
            conditions.freestream.Cp                 = 1.4*(p/(rho*T))/(1.4-1)
            conditions.freestream.R                  = p/(rho*T)
            conditions.freestream.speed_of_sound     = np.atleast_1d(a)
            conditions.freestream.velocity           = np.atleast_1d(a*mach_number)
            
            # propulsion conditions
            conditions.propulsion.throttle           =  np.atleast_1d(1.0)
    
    ram                       = scramjet.ram
    inlet_nozzle              = scramjet.inlet_nozzle
    combustor                 = scramjet.combustor
    core_nozzle               = scramjet.core_nozzle
    thrust                    = scramjet.thrust
    number_of_engines         = scramjet.number_of_engines
    
    #Creating the network by manually linking the different components
    #set the working fluid to determine the fluid properties
    ram.inputs.working_fluid                             = scramjet.working_fluid
    
    #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
    ram(conditions)
    
    #link inlet nozzle to ram 
    inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature #conditions.freestream.stagnation_temperature
    inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure #conditions.freestream.stagnation_pressure
    
    #Flow through the inlet nozzle
    inlet_nozzle.compute_scramjet(conditions)

    #link the combustor to the inlet nozzle
    combustor.inputs.stagnation_temperature                = inlet_nozzle.outputs.stagnation_temperature
    combustor.inputs.stagnation_pressure                   = inlet_nozzle.outputs.stagnation_pressure
    combustor.inputs.inlet_nozzle                          = inlet_nozzle.outputs
    
    #flow through the combustor
    combustor.compute_scramjet(conditions)

    
    #link the core nozzle to combustor
    core_nozzle.inputs.stagnation_temperature              = combustor.outputs.stagnation_temperature
    core_nozzle.inputs.stagnation_pressure                 = combustor.outputs.stagnation_pressure
    core_nozzle.inputs.static_temperature                  = combustor.outputs.static_temperature
    core_nozzle.inputs.static_pressure                     = combustor.outputs.static_pressure
    core_nozzle.inputs.velocity                            = combustor.outputs.velocity
    core_nozzle.inputs.fuel_to_air_ratio                   = combustor.outputs.fuel_to_air_ratio

    
    #flow through the core nozzle
    core_nozzle.compute_scramjet(conditions)

    # compute the thrust using the thrust component
    #link the thrust component to the core nozzle
    thrust.inputs.core_exit_pressure                       = core_nozzle.outputs.pressure
    thrust.inputs.core_exit_temperature                    = core_nozzle.outputs.temperature 
    thrust.inputs.core_exit_velocity                       = core_nozzle.outputs.velocity
    thrust.inputs.core_area_ratio                          = core_nozzle.outputs.area_ratio
    thrust.inputs.core_nozzle                              = core_nozzle.outputs
    
    #link the thrust component to the combustor
    thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio
    
    #link the thrust component to the core nozzle 
    thrust.inputs.stag_temp_lpt_exit                       = core_nozzle.outputs.stagnation_temperature
    thrust.inputs.stag_press_lpt_exit                      = core_nozzle.outputs.stagnation_pressure
    thrust.inputs.number_of_engines                        = number_of_engines
    thrust.inputs.total_temperature_reference              = core_nozzle.outputs.stagnation_temperature
    thrust.inputs.total_pressure_reference                 = core_nozzle.outputs.stagnation_pressure

    #compute the thrust
    thrust.inputs.fan_nozzle = Data()
    thrust.inputs.fan_nozzle.velocity = 0.0
    thrust.inputs.fan_nozzle.area_ratio = 0.0
    thrust.inputs.fan_nozzle.static_pressure = 0.0
    thrust.inputs.bypass_ratio = 0.0
    thrust.inputs.flow_through_core                        =  1.0 #scaled constant to turn on core thrust computation
    thrust.inputs.flow_through_fan                         =  0.0 #scaled constant to turn on fan thrust computation     
    thrust.size_stream_thrust(conditions)
    
    #update the design thrust value
    scramjet.design_thrust = thrust.total_design

    #compute the sls_thrust
    #call the atmospheric model to get the conditions at the specified altitude
    atmosphere_sls = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmosphere_sls.compute_values(0.0,0.0)
    
    p   = atmo_data.pressure          
    T   = atmo_data.temperature       
    rho = atmo_data.density          
    a   = atmo_data.speed_of_sound    
    mu  = atmo_data.dynamic_viscosity   

    # setup conditions
    conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()            

    # freestream conditions    
    conditions_sls.freestream.altitude           = np.atleast_1d(0.)
    conditions_sls.freestream.mach_number        = np.atleast_1d(0.01)
    conditions_sls.freestream.pressure           = np.atleast_1d(p)
    conditions_sls.freestream.temperature        = np.atleast_1d(T)
    conditions_sls.freestream.density            = np.atleast_1d(rho)
    conditions_sls.freestream.dynamic_viscosity  = np.atleast_1d(mu)
    conditions_sls.freestream.gravity            = np.atleast_1d(9.81)
    conditions_sls.freestream.gamma              = np.atleast_1d(1.4)
    conditions_sls.freestream.Cp                 = 1.4*(p/(rho*T))/(1.4-1)
    conditions_sls.freestream.R                  = p/(rho*T)
    conditions_sls.freestream.speed_of_sound     = np.atleast_1d(a)
    conditions_sls.freestream.velocity           = np.atleast_1d(a*0.01)
    
    # propulsion conditions
    conditions_sls.propulsion.throttle           =  np.atleast_1d(1.0)    
    
    state_sls = Data()
    state_sls.numerics = Data()
    state_sls.conditions = conditions_sls   
    results_sls = scramjet.evaluate_thrust(state_sls)
