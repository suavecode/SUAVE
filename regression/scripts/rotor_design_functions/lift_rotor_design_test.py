# lift_rotor_design_test.py
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
from SUAVE.Methods.Propulsion import  lift_rotor_design 

def main():  

    rotor                                               = SUAVE.Components.Energy.Converters.Lift_Rotor() 
    rotor.tag                                           = 'rotor'
    rotor.orientation_euler_angles                      = [0, 90*Units.degrees,0]
    rotor.tip_radius                                    = 1.15
    rotor.hub_radius                                    = 0.15 * rotor.tip_radius  
    rotor.number_of_blades                              = 3    
                 
    rotor.hover.design_altitude                         = 40 * Units.feet  
    rotor.hover.design_thrust                           = 2000 
    rotor.hover.design_freestream_velocity              = np.sqrt(rotor.hover.design_thrust/(2*1.2*np.pi*(rotor.tip_radius**2))) 
                  
    rotor.oei.design_altitude                           = 40 * Units.feet  
    rotor.oei.design_thrust                             = 2000*1.1 
    rotor.oei.design_freestream_velocity                = np.sqrt(rotor.oei.design_thrust/(2*1.2*np.pi*(rotor.tip_radius**2))) 
                 
    rotor.variable_pitch                                = True       
    opt_params                                          = rotor.optimization_parameters 
    opt_params.multiobjective_aeroacoustic_weight       = 0.5 # 1 means only perfomrance optimization 0.5 to weight noise equally  
    rotor                                               = lift_rotor_design(rotor)  # Reduced iteration for regression therefore optimal design is NOT reached!   
    
    # Find the operating conditions
    atmosphere                                          = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions                               = atmosphere.compute_values(rotor.hover.design_altitude)  
    conditions                                          = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    conditions._size                                    = 1
    conditions.freestream                               = Data()
    conditions.propulsion                               = Data()
    conditions.frames                                   = Data()
    conditions.frames.body                              = Data()
    conditions.frames.inertial                          = Data()
    conditions.freestream.update(atmosphere_conditions) 
    conditions.frames.inertial.velocity_vector          = np.array([[0, 0. ,rotor.hover.design_freestream_velocity]])  
    conditions.propulsion.throttle                      = np.array([[1.0]])
    conditions.frames.body.transform_to_inertial        = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., -1.]]])      

    # Assign rpm
    rotor.inputs.omega  = np.array(rotor.hover.design_angular_velocity,ndmin=2) 

    # rotor with airfoil results  
    F_rot, Q_rot, P_rot, Cp_rot ,output_rot , etap_rot = rotor.spin(conditions)
    plot_results(output_rot, rotor,'green','-','^') 

    # Truth values for rotor with airfoil geometry defined  
    F_rot_truth = 1999.7999999998628

    # Store errors 
    error = Data()  
    error.Thrust_rot = np.max(np.abs(np.linalg.norm(F_rot)-F_rot_truth))/F_rot_truth   
    
    print('Thrust: ' + str(np.linalg.norm(-F_rot))) 
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
