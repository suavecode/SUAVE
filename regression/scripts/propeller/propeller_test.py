# propeller_test.py
# 
# Created:  Sep 2014, E. Botero
# Modified: Feb 2020, M. Clarke  
#           Sep 2020, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Plots.Geometry_Plots import plot_propeller
import matplotlib.pyplot as plt  
from SUAVE.Core import (
Data, Container,
)

import numpy as np
import copy, time
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller

def main():
    
    # This script could fail if either the design or analysis scripts fail,
    # in case of failure check both. The design and analysis powers will 
    # differ because of karman-tsien compressibility corrections in the 
    # analysis scripts
    
    net                       = Battery_Propeller()   
    net.number_of_engines     = 2    
    
    # Design Gearbox 
    gearbox                   = SUAVE.Components.Energy.Converters.Gearbox()
    gearbox.gearwheel_radius1 = 1
    gearbox.gearwheel_radius2 = 1
    gearbox.efficiency        = 0.95
    gearbox.inputs.torque     = 885.550158704757
    gearbox.inputs.speed      = 207.16160479940007
    gearbox.inputs.power      = 183451.9920076409
    gearbox.compute()
     
    
    # Design the Propeller with airfoil  geometry defined                      
    bad_prop                          = SUAVE.Components.Energy.Converters.Propeller() 
    bad_prop.tag                      = "Prop_W_Aifoil"
    bad_prop.number_of_blades         = 2
    bad_prop.number_of_engines        = 1
    bad_prop.freestream_velocity      = 1
    bad_prop.tip_radius               = 0.3
    bad_prop.hub_radius               = 0.21336 
    bad_prop.design_tip_mach          = 0.1
    bad_prop.angular_velocity         = gearbox.inputs.speed  
    bad_prop.design_Cl                = 0.7
    bad_prop.design_altitude          = 1. * Units.km        
    bad_prop.design_thrust           = 100000
    bad_prop                         = propeller_design(bad_prop)  
    
    prop_a                          = SUAVE.Components.Energy.Converters.Propeller() 
    prop_a.tag                      = "Prop_W_Aifoil"
    prop_a.number_of_blades         = 3
    prop_a.number_of_engines        = 1
    prop_a.freestream_velocity      = 49.1744 
    prop_a.tip_radius               = 1.0668
    prop_a.hub_radius               = 0.21336 
    prop_a.design_tip_mach          = 0.65
    prop_a.angular_velocity         = gearbox.inputs.speed # 207.16160479940007 
    prop_a.design_Cl                = 0.7
    prop_a.design_altitude          = 1. * Units.km      
    prop_a.rotation                 = [1] 
    prop_a.design_thrust            = 3054.4809132125697
    prop_a                          = propeller_design(prop_a)  
    
    # plot propeller 
    plot_propeller(prop_a)
 
    # Design a Rotor with airfoil  geometry defined  
    rot_a                          = SUAVE.Components.Energy.Converters.Rotor() 
    rot_a.tag                      = "Rot_W_Aifoil"
    rot_a.tip_radius               = 2.8 * Units.feet
    rot_a.hub_radius               = 0.35 * Units.feet      
    rot_a.number_of_blades         = 2   
    rot_a.design_tip_mach          = 0.65
    rot_a.number_of_engines        = 12
    rot_a.disc_area                = np.pi*(rot_a.tip_radius**2)        
    rot_a.induced_hover_velocity   = 12.756071638899549
    rot_a.freestream_velocity      = 500. * Units['ft/min']  
    rot_a.angular_velocity         = 258.9520059992501
    rot_a.design_Cl                = 0.7
    rot_a.design_altitude          = 20 * Units.feet                            
    rot_a.design_thrust            = 2271.2220451593753 
    rot_a                          = propeller_design(rot_a) 
    
    # Find the operating conditions
    atmosphere            = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions =  atmosphere.compute_values(rot_a.design_altitude)
    
    V  = prop_a.freestream_velocity
    Vr = rot_a.freestream_velocity
    
    conditions                                          = Data()
    conditions.freestream                               = Data()
    conditions.propulsion                               = Data()
    conditions.frames                                   = Data()
    conditions.frames.body                              = Data()
    conditions.frames.inertial                          = Data()
    conditions.freestream.update(atmosphere_conditions)
    conditions.freestream.dynamic_viscosity             = atmosphere_conditions.dynamic_viscosity
    conditions.frames.inertial.velocity_vector          = np.array([[V,0,0]])
    conditions.propulsion.throttle                      = np.array([[1.0]])
    conditions.frames.body.transform_to_inertial        = np.array([np.eye(3)])
    
    conditions_r = copy.deepcopy(conditions)
    conditions.frames.inertial.velocity_vector   = np.array([[V,0,0]])
    conditions_r.frames.inertial.velocity_vector = np.array([[0,Vr,0]])
    
    # Create and attach this propeller 
    prop_a.inputs.omega  = np.array(prop_a.angular_velocity,ndmin=2)
    rot_a.inputs.omega   = copy.copy(prop_a.inputs.omega)
    
    # propeller with airfoil results 
    F_a, Q_a, P_a, Cplast_a ,output_a , etap_a = prop_a.spin(conditions)  
    plot_results(output_a, prop_a,'blue','-','s')
    
    # rotor with airfoil results 
    Fr_a, Qr_a, Pr_a, Cplastr_a ,outputr_a , etapr = rot_a.spin(conditions_r)
    plot_results(outputr_a, rot_a,'green','-','^')
    
    # Truth values for propeller with airfoil geometry defined 
    F_a_truth       = 3383.41788013
    Q_a_truth       = 998.54011272
    P_a_truth       = 206859.17220762
    Cplast_a_truth  = 0.10662024
    
    # Truth values for rotor with airfoil geometry defined 
    Fr_a_truth      = 1447.95378214
    Qr_a_truth      = 191.08844467
    Pr_a_truth      = 39586.18885581
    Cplastr_a_truth = 0.0622671

    # Store errors 
    error = Data()
    error.Thrust_a  = np.max(np.abs(F_a -F_a_truth))
    error.Torque_a  = np.max(np.abs(Q_a -Q_a_truth))    
    error.Power_a   = np.max(np.abs(P_a -P_a_truth))
    error.Cp_a      = np.max(np.abs(Cplast_a -Cplast_a_truth))  
    error.Thrustr_a = np.max(np.abs(Fr_a-Fr_a_truth))
    error.Torquer_a = np.max(np.abs(Qr_a-Qr_a_truth))    
    error.Powerr_a  = np.max(np.abs(Pr_a-Pr_a_truth))
    error.Cpr_a     = np.max(np.abs(Cplastr_a-Cplastr_a_truth))  
    
    print('Errors:')
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)

    return

def plot_results(results,prop,c,ls,m):
    
    tag                = prop.tag
    va_ind             = results.blade_axial_induced_velocity[0]  
    vt_ind             = results.blade_tangential_induced_velocity[0]  
    r                  = prop.radius_distribution
    T_distribution     = results.blade_thrust_distribution[0] 
    vt                 = results.blade_tangential_velocity[0]  
    va                 = results.blade_axial_velocity[0] 
    Q_distribution     = results.blade_torque_distribution[0] 
        
    # ----------------------------------------------------------------------------
    # 2D - Plots  Plots    
    # ---------------------------------------------------------------------------- 
    # perpendicular velocity, up Plot 
    fig = plt.figure('va_ind')         
    plt.plot(r  , va_ind ,color = c  , marker = m, linestyle = ls , label =  tag)          
    plt.xlabel('Radial Location')
    plt.ylabel('Induced Axial Velocity') 
    plt.legend(loc='lower right') 
    
    fig = plt.figure('vt_ind')          
    plt.plot(r  , vt_ind ,color = c ,marker = m, linestyle = ls , label =  tag )       
    plt.xlabel('Radial Location')
    plt.ylabel('Induced Tangential Velocity') 
    plt.legend(loc='lower right')  
        
    fig = plt.figure('T')     
    plt.plot(r , T_distribution ,color = c ,marker = m, linestyle = ls, label =  tag  )    
    plt.xlabel('Radial Location')
    plt.ylabel('Trust, N')
    plt.legend(loc='lower right')
    
    fig = plt.figure('Q')
    plt.plot(r , Q_distribution ,color = c ,marker = m, linestyle = ls, label =  tag)            
    plt.xlabel('Radial Location')
    plt.ylabel('Torque, N-m')
    plt.legend(loc='lower right')
    
    fig = plt.figure('Va')     
    plt.plot(r , va ,color = c  ,marker =m, linestyle = ls, label =  tag + 'axial vel')          
    plt.xlabel('Radial Location')
    plt.ylabel('Axial Velocity') 
    plt.legend(loc='lower right') 
    
    fig = plt.figure('Vt')       
    plt.plot(r , vt ,color = c ,marker = m, linestyle = ls, label =  tag )         
    plt.xlabel('Radial Location')
    plt.ylabel('Tangential Velocity') 
    plt.legend(loc='lower right')  
    
    return 

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()