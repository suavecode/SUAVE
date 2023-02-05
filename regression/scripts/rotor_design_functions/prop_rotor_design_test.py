# prop_rotor_design_test.py
# 
# Created:  Sep 2014, E. Botero
# Modified: Feb 2020, M. Clarke  
#           Sep 2020, M. Clarke 
#           Nov 2022, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Visualization.Geometry.Three_Dimensional import plot_3d_rotor
import matplotlib.pyplot as plt  
from SUAVE.Core import Data 

import numpy as np
import copy 
from SUAVE.Methods.Propulsion import  prop_rotor_design  

def main(): 
    
    prop_rotor                                          = SUAVE.Components.Energy.Converters.Prop_Rotor() 
    prop_rotor.tag                                      = 'prop_rotor'     
    prop_rotor.tip_radius                               = 3/2
    prop_rotor.hub_radius                               = 0.2 * prop_rotor.tip_radius
    prop_rotor.number_of_blades                         = 3  
          
    # HOVER       
    prop_rotor.hover.design_altitude                    = 20 * Units.feet                  
    prop_rotor.hover.design_thrust                      = 21356/(6)  
    prop_rotor.hover.design_freestream_velocity         = np.sqrt(prop_rotor.hover.design_thrust/(2*1.2*np.pi*(prop_rotor.tip_radius**2))) # 
        
    # HOVER       
    prop_rotor.oei.design_altitude                      = 20 * Units.feet                  
    prop_rotor.oei.design_thrust                        = 21356/(6-2)  
    prop_rotor.oei.design_freestream_velocity           = np.sqrt(prop_rotor.oei.design_thrust/(2*1.2*np.pi*(prop_rotor.tip_radius**2))) # 
       
              
    # CRUISE                         
    prop_rotor.cruise.design_altitude                   = 2500 * Units.feet                      
    prop_rotor.cruise.design_thrust                     = 1400 
    prop_rotor.cruise.design_freestream_velocity        = 175*Units.mph  
    

    opt_params                                          = prop_rotor.optimization_parameters 
    opt_params.multiobjective_performance_weight        = 1.0  
    opt_params.multiobjective_acoustic_weight           = 1.0  # Do not consider cruise noise 
    opt_params.multiobjective_aeroacoustic_weight       = 1.0  # 1 means only perfomrance optimization 0.5 to weight noise equally

    # DESING ROTOR              
    prop_rotor                                          = prop_rotor_design(prop_rotor)  # Reduced iteration for regression therefore optimal design is NOT reached! 

    # Find the operating conditions
    atmosphere                                          = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions                               =  atmosphere.compute_values(prop_rotor.hover.design_altitude)  
    conditions                                          = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    conditions._size                                    = 1
    conditions.freestream                               = Data()
    conditions.propulsion                               = Data()
    conditions.frames                                   = Data()
    conditions.frames.body                              = Data()
    conditions.frames.inertial                          = Data()
    conditions.freestream.update(atmosphere_conditions) 
    conditions.frames.inertial.velocity_vector          = np.array([[0, 0. ,prop_rotor.hover.design_freestream_velocity]])  
    conditions.propulsion.throttle                      = np.array([[1.0]])
    conditions.frames.body.transform_to_inertial        = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., -1.]]])      

    # Assign rpm
    prop_rotor.inputs.omega              = np.array(prop_rotor.hover.design_angular_velocity,ndmin=2)  
    prop_rotor.orientation_euler_angles  = [0,np.pi/2,0]  

    # rotor with airfoil results  
    F_pr, Q_pr, P_pr, Cp_pr ,output_pr , etap_pr = prop_rotor.spin(conditions)
    plot_results(output_pr, prop_rotor,'blue','-','^') 

    # Truth values for rotor with airfoil geometry defined 
    F_pr_truth = 3559.689350071597

    # Store errors 
    error = Data()  
    error.Thrust_pr = np.max(np.abs(np.linalg.norm(F_pr)-F_pr_truth))/F_pr_truth 

    print('Thrust: ' + str(np.linalg.norm(-F_pr))) 
    print('Error: ' + str(error))  

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
