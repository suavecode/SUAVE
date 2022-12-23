# rotor_design_test.py
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
from SUAVE.Visualization.Geometry import plot_rotor
import matplotlib.pyplot as plt  
from SUAVE.Core import Data 

import numpy as np
import copy 
from SUAVE.Methods.Propulsion import propeller_design , lift_rotor_design , prop_rotor_design  

def main():

    # This script could fail if either the design or analysis scripts fail,
    # in case of failure check both. The design and analysis powers will 
    # differ because of karman-tsien compressibility corrections in the 
    # analysis scripts 

    propeller_design_test()

    lift_rotor_design_test()

    prop_rotor_design_test()
    return     


def propeller_design_test():

    # This script could fail if either the design or analysis scripts fail,
    # in case of failure check both. The design and analysis powers will 
    # differ because of karman-tsien compressibility corrections in the 
    # analysis scripts

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
    bad_prop                                   = SUAVE.Components.Energy.Converters.Propeller() 
    bad_prop.tag                               = "Prop_W_Aifoil"
    bad_prop.number_of_blades                  = 2
    bad_prop.number_of_engines                 = 1
    bad_prop.tip_radius                        = 0.3
    bad_prop.hub_radius                        = 0.21336 
    bad_prop.cruise.design_freestream_velocity = 1
    bad_prop.cruise.design_tip_mach            = 0.1
    bad_prop.cruise.design_angular_velocity    = gearbox.inputs.speed  
    bad_prop.cruise.design_thrust              = 100000
    bad_prop.cruise.design_Cl                  = 0.7
    bad_prop.cruise.design_altitude            = 1. * Units.km  
    airfoil                                    = SUAVE.Components.Airfoils.Airfoil()   
    airfoil.coordinate_file                    ='4412'   
    airfoil.NACA_4_series_flag                 = True 
    airfoil.polar_files                        =  ['../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_50000.txt',
                                                   '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_100000.txt',
                                                       '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_200000.txt',
                                                       '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_500000.txt',
                                                       '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_1000000.txt']
    bad_prop.append_airfoil(airfoil)          
    bad_prop.airfoil_polar_stations            =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    bad_prop                                   = propeller_design(bad_prop)
    prop_a                                     = SUAVE.Components.Energy.Converters.Propeller()
    prop_a.tag                                 = "Prop_W_Aifoil"
    prop_a.number_of_blades                    = 3
    prop_a.number_of_engines                   = 1
    prop_a.tip_radius                          = 1.0668
    prop_a.hub_radius                          = 0.21336
    prop_a.cruise.design_freestream_velocity   = 49.1744
    prop_a.cruise.design_tip_mach              = 0.65
    prop_a.cruise.design_angular_velocity      = gearbox.inputs.speed # 207.16160479940007
    prop_a.cruise.design_Cl                    = 0.7
    prop_a.cruise.design_altitude              = 1. * Units.km 
    prop_a.cruise.design_thrust                = 3054.4809132125697
  
    # define first airfoil   
    airfoil_1                                  = SUAVE.Components.Airfoils.Airfoil()
    airfoil_1.tag                              = 'NACA_4412' 
    airfoil_1.coordinate_file                  = '../Vehicles/Airfoils/NACA_4412.txt'   # absolute path   
    airfoil_1.polar_files                      = ['../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_50000.txt',
                                                 '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_100000.txt',
                                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_200000.txt',
                                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_500000.txt',
                                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_1000000.txt'] 
    prop_a.append_airfoil(airfoil_1)           # append first airfoil 

    # define  second airfoil 
    airfoil_2                                = SUAVE.Components.Airfoils.Airfoil()  
    airfoil_2.tag                            = 'Clark_Y' 
    airfoil_2.coordinate_file                = '../Vehicles/Airfoils/Clark_y.txt' 
    airfoil_2.polar_files                    = ['../Vehicles/Airfoils/Polars/Clark_y_polar_Re_50000.txt',
                                                '../Vehicles/Airfoils/Polars/Clark_y_polar_Re_100000.txt',
                                                 '../Vehicles/Airfoils/Polars/Clark_y_polar_Re_200000.txt',
                                                 '../Vehicles/Airfoils/Polars/Clark_y_polar_Re_500000.txt',
                                                 '../Vehicles/Airfoils/Polars/Clark_y_polar_Re_1000000.txt'] 
    prop_a.append_airfoil(airfoil_2)          # append second airfoil 

    # define polar stations on rotor 
    prop_a.airfoil_polar_stations           = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1]
    prop_a                                  = propeller_design(prop_a)

    # plot propeller 
    plot_rotor(prop_a)

    # Design the Propeller with airfoil  geometry defined 
    prop                                   = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                               = "Prop_No_Aifoil"
    prop.number_of_blades                  = 3
    prop.number_of_engines                 = 1
    prop.tip_radius                        = 1.0668
    prop.hub_radius                        = 0.21336
    prop.cruise.design_freestream_velocity = 49.1744
    prop.cruise.design_tip_mach            = 0.65
    prop.cruise.design_angular_velocity    = gearbox.inputs.speed
    prop.cruise.design_Cl                  = 0.7
    prop.cruise.design_altitude            = 1. * Units.km
    prop.cruise.design_power               = gearbox.outputs.power 
    prop.origin                            = [[16.*0.3048 , 0. ,2.02*0.3048 ]]
    prop                                   = propeller_design(prop)  

    # Find the operating conditions
    atmosphere                                          = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions                               = atmosphere.compute_values(prop.cruise.design_altitude)  
    conditions                                          = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    conditions._size                                    = 1
    conditions.freestream                               = Data()
    conditions.propulsion                               = Data()
    conditions.frames                                   = Data()
    conditions.frames.body                              = Data()
    conditions.frames.inertial                          = Data()
    conditions.freestream.update(atmosphere_conditions)
    conditions.freestream.dynamic_viscosity             = atmosphere_conditions.dynamic_viscosity
    conditions.frames.inertial.velocity_vector          = np.array([[prop.cruise.design_freestream_velocity,0,0]])
    conditions.propulsion.throttle                      = np.array([[1.0]])
    conditions.frames.body.transform_to_inertial        = np.array([np.eye(3)])

    conditions.frames.inertial.velocity_vector   = np.array([[prop.cruise.design_freestream_velocity,0,0]]) 

    # Create and attach this propeller 
    prop_a.inputs.omega  = np.array(prop.cruise.design_angular_velocity,ndmin=2)
    prop.inputs.omega    = np.array(prop.cruise.design_angular_velocity,ndmin=2) 

    # propeller with airfoil results 
    prop_a.inputs.pitch_command                = 0.0*Units.degree
    F_a, Q_a, P_a, Cplast_a ,output_a , etap_a = prop_a.spin(conditions)  
    plot_results(output_a, prop_a,'blue','-','s')

    # propeller without airfoil results 
    prop.inputs.pitch_command           = 0.0*Units.degree
    F, Q, P, Cplast ,output , etap      = prop.spin(conditions)
    plot_results(output, prop,'red','-','o')

    # Truth values for propeller with airfoil geometry defined 
    F_a_truth       = 3040.8279405853377
    Q_a_truth       = 888.55038334
    P_a_truth       = 184073.52335802
    Cplast_a_truth  = 0.10448797

    # Truth values for propeller without airfoil geometry defined 
    F_truth         = 2423.475147297527 
    Q_truth         = 727.12650888
    P_truth         = 150632.69447246
    Cplast_truth    = 0.08550553

    # Store errors 
    error = Data()
    error.Thrust_a  = np.max(np.abs(np.linalg.norm(F_a) -F_a_truth))
    error.Torque_a  = np.max(np.abs(Q_a -Q_a_truth))    
    error.Power_a   = np.max(np.abs(P_a -P_a_truth))
    error.Cp_a      = np.max(np.abs(Cplast_a -Cplast_a_truth))  
    error.Thrust    = np.max(np.abs(np.linalg.norm(F)-F_truth))
    error.Torque    = np.max(np.abs(Q-Q_truth))    
    error.Power     = np.max(np.abs(P-P_truth))
    error.Cp        = np.max(np.abs(Cplast-Cplast_truth))    

    print('Errors:')
    print(error)

    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)

    return



def lift_rotor_design_test():     

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
    F_rot_truth = 1999.9851182792063

    # Store errors 
    error = Data()  
    error.Thrust_rot = np.max(np.abs(np.linalg.norm(F_rot)-F_rot_truth))/F_rot_truth   
    
    print('Thrust: ' + str(np.linalg.norm(-F_rot))) 
    print('Error: ' + str(error)) 

    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)  

    return


def prop_rotor_design_test():     


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
    F_pr_truth = 3559.688987104335

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
