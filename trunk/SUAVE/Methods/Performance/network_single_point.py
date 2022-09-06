## @ingroup Methods-Performance
# propeller_network_single_point.py
#
# Created:  Aug 2022, R. Erhard
# Modified: 
#           

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import SUAVE

from SUAVE.Core import Data, Units
from SUAVE.Methods.Fluid_Domain.generate_fluid_domain_grid_points import generate_fluid_domain_grid_points
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_wake_induced_velocity import compute_wake_induced_velocity

from DCode.Common.Visualization_Tools.box_contour_field_vtk import box_contour_field_vtk
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------
#   Propeller Single Point
# ------------------------------------------------------------------------------

## @ingroup Methods-Performance
def network_single_point(net,
                           pitch,
                           omega,
                           altitude,
                           delta_isa,
                           speed,
                           analyses=None,
                           plots=False,
                           print_results=False):
    """network_single_point(net,
                            pitch,
                            omega,
                            altitude,
                            delta_isa,
                            speed,
                            analyses=None,
                            plots=False,
                            print_results=False):

        Uses SUAVE's propeller spin function to evaluate propeller network performance at a
        single altitude, pitch command, and angular velocity. Can be used indep-
        endently, or as part of creation of a propller maps or flight envelopes.

        Sources:
        N/A

        Assumptions:

        Assumes use of Battery Propeller Energy Network.

        Inputs:

            net                 SUAVE Propeller Network Data Structure
            pitch                Propeller Pitch/Collective                    [User Set]
            omega                Test Angular Velocity                         [User Set]
            altitude             Test Altitude                                 [User Set]
            delta_isa            Atmosphere Temp Offset                        [K]
            speed                Propeller Intake Speed                        [User Set]
            HFW                  Flag for use of helical fixed wake for rotor  [Boolean]
            analyses             SUAVE Analyses Structure
                .atmosphere      SUAVE Atmosphere Analysis Object
            plots                Flag for Plot Generation                      [Boolean]
            print_results        Flag for Terminal Output                      [Boolean]

        Outputs:

            results                             SUAVE Data Object
                .thrust                         BEVW Thrust Prediction      [N]
                .torque                         BEVW Torque Prediction      [N-m]
                .power                          BEVW Power Prediction       [W]
                .power_coefficient              BEVW Cp Prediction          [Unitless]
                .efficiency                     BEVW Efficiency Prediction  [Unitless]
                .induced_axial_velocity         BEVW Ind. V_a Prediction    [m/s]
                .induced_tangential_velocity    BEVW Ind. V_tPrediction     [m/s]
                .radial_distribution            BEVW Radial Stations        [m]
                .thrust_distribution            BEVW T Dist. Prediction     [N/m]
                .torque_distribution            BEVW Q Dist. Prediction     [(N-m)/m]
                .tangential_velocity            BEVW V_t Prediction         [m/s]
                .axial_velocity                 BEVW V_a Prediction         [m/s]
    """
    # Set atmosphere
    if analyses==None:
        # setup standard US 1976 atmosphere
        analyses   = SUAVE.Analyses.Vehicle()
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        analyses.append(atmosphere)           
        
    # Unpack Inputs
    ctrl_pts = 1

    atmo_data           = analyses.atmosphere.compute_values(altitude, delta_isa)
    T                   = atmo_data.temperature
    a                   = atmo_data.speed_of_sound
    density             = atmo_data.density
    dynamic_viscosity   = atmo_data.dynamic_viscosity

    # Setup Pseudo-Mission for Prop Evaluation
    conditions                                      = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
    conditions.freestream                           = Data()
    conditions.propulsion                           = Data()
    conditions.noise                                = Data()
    conditions.noise.sources                        = Data()
    conditions.noise.sources.propellers             = Data()
    conditions.frames                               = Data()
    conditions.frames.inertial                      = Data()
    conditions.frames.body                          = Data()    
    conditions.freestream.density                   = np.ones((ctrl_pts, 1)) * density
    conditions.freestream.dynamic_viscosity         = np.ones((ctrl_pts, 1)) * dynamic_viscosity
    conditions.freestream.speed_of_sound            = np.ones((ctrl_pts, 1)) * a
    conditions.freestream.temperature               = np.ones((ctrl_pts, 1)) * T
    conditions.freestream.mach_number               = speed / a
    conditions.freestream.velocity                  = speed
    velocity_vector                                 = np.array([[speed, 0., 0.]])
    
    conditions.propulsion.throttle                  = np.ones((ctrl_pts, 1)) * 1.
    conditions.frames.inertial.velocity_vector      = np.tile(velocity_vector, (ctrl_pts, 1))
    conditions.frames.body.transform_to_inertial    = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])

    # ----------------------------------------------------------------------------
    # Run Propeller Analysis for all props, including interactions if specified
    # ----------------------------------------------------------------------------
    propellers = net.propellers

    # Generate grids for fluid domain
    for prop in propellers:
        prop.inputs.pitch_command   = pitch
        prop.inputs.omega           = np.ones((ctrl_pts, 1)) * omega
    
        wake_fidelity = prop.Wake.wake_method_fidelity
        
        if wake_fidelity == 2:
            # Initialize wake with F1 wake
            prop.Wake.initialize(prop, conditions)
            
            # Generate a grid domain for this rotor
            Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = prop.Wake.get_fluid_domain_boundaries(prop)
            grid_points = generate_fluid_domain_grid_points(Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, prop.tip_radius)            
            prop.Wake.fluid_domain_grid_points = grid_points
    if wake_fidelity == 2:
        
        # Generate induced velocity field at each propeller's computational grid from all propeller wakes  
        for prop in propellers:
            total_V_ind = np.zeros((1,len(prop.Wake.fluid_domain_grid_points.XC),3))
            ComponentContributions = Data()
            # Compute the induced velocity at this prop grid points from all propellers
            for p in propellers:
                p_contribution = compute_wake_induced_velocity(p.Wake.vortex_distribution, prop.Wake.fluid_domain_grid_points, cpts=1)  
                ComponentContributions[p.tag] = p_contribution
                total_V_ind += p_contribution      
                
            # Store induced velocity fluid domain for this propeller's computational grid
        
            Vind = np.zeros(np.append(grid_points.original_shape, 3))
            Vind[:,:,:,0] = np.reshape(total_V_ind[0,:,0], grid_points.original_shape)
            Vind[:,:,:,1] = np.reshape(total_V_ind[0,:,1], grid_points.original_shape)
            Vind[:,:,:,2] = np.reshape(total_V_ind[0,:,2], grid_points.original_shape)
            
            interpolatedBoxData = Data()
            interpolatedBoxData.N_width  = len(prop.Wake.fluid_domain_grid_points.Xp[:,0,0]) # x-direction (rotor frame)
            interpolatedBoxData.N_depth  = len(prop.Wake.fluid_domain_grid_points.Xp[0,:,0]) # y-direction (rotor frame)
            interpolatedBoxData.N_height = len(prop.Wake.fluid_domain_grid_points.Xp[0,0,:]) # z-direction (rotor frame)
            interpolatedBoxData.Position = np.transpose(np.array([prop.Wake.fluid_domain_grid_points.Xp, prop.Wake.fluid_domain_grid_points.Yp, prop.Wake.fluid_domain_grid_points.Zp]), (1,2,3,0))
            interpolatedBoxData.Velocity = Vind
            interpolatedBoxData.ComponentContributions = ComponentContributions
        
            # generate contour box vtk visual
            stateData = Data()
            stateData.vFreestream = speed
            stateData.alphaDeg = prop.orientation_euler_angles[1] / Units.deg    
            box_contour_field_vtk(interpolatedBoxData, stateData, iteration=0, filename="/Users/rerha/Desktop/sbs_test_vtks/"+prop.tag+"ContourBox.vtk")     
            for p in propellers: #tagContribution in interpolatedBoxData.ComponentContributions:
                tempData = Data()
                tempData.N_width  = interpolatedBoxData.N_width 
                tempData.N_depth  = interpolatedBoxData.N_depth 
                tempData.N_height = interpolatedBoxData.N_height
                tempData.Position = interpolatedBoxData.Position
                
                pV = interpolatedBoxData.ComponentContributions[p.tag]
                pVind = np.zeros(np.append(grid_points.original_shape, 3))
                pVind[:,:,:,0] = np.reshape(pV[0,:,0], grid_points.original_shape)
                pVind[:,:,:,1] = np.reshape(pV[0,:,1], grid_points.original_shape)
                pVind[:,:,:,2] = np.reshape(pV[0,:,2], grid_points.original_shape)
                tempData.Velocity = pVind
                box_contour_field_vtk(tempData, stateData, iteration=0, filename="/Users/rerha/Desktop/sbs_test_vtks/"+prop.tag+"ContourBox_"+p.tag+"_contribution.vtk")    
                
            
            #--------------------------------------------------------------------------------------------    
            # Step 1d: Generate function for induced velocity, Vind = f(x,y,z)
            #--------------------------------------------------------------------------------------------
            #Vinf = conditions.frames.inertial.velocity_vector
            #rot_mat = prop.body_to_prop_vel()
            #vVec = np.matmul(Vinf, rot_mat)
            
            #V_induced = Vind + V_ind_ext + vVec[0]
            
            fun_V_induced = RegularGridInterpolator((prop.Wake.fluid_domain_grid_points.Xouter,prop.Wake.fluid_domain_grid_points.Youter,prop.Wake.fluid_domain_grid_points.Zouter), Vind)        
            prop.Wake.fluid_domain_velocity_field = fun_V_induced
            prop.Wake.fluid_domain_interpolated_box_data = interpolatedBoxData
    
    # Analyze all propellers in presence of fluid domain
    for prop in propellers:
        F, Q, P, Cp, outputs, etap = prop.spin(conditions)
        
    va_ind_BEVW         = outputs.disc_axial_induced_velocity[0, :, 0]
    vt_ind_BEVW         = outputs.disc_tangential_induced_velocity[0, :, 0]
    r_BEVW              = outputs.disc_radial_distribution[0, :, 0]
    T_distribution_BEVW = outputs.disc_thrust_distribution[0, :, 0]
    vt_BEVW             = outputs.disc_tangential_velocity[0, :, 0]
    va_BEVW             = outputs.disc_axial_velocity[0, :, 0]
    Q_distribution_BEVW = outputs.disc_torque_distribution[0, :, 0]

    if print_results:
        print('Total Thrust:    {} N'.format(F[0][0]))
        print('Total Torque:    {} N-m'.format(Q[0][0]))
        print('Total Power:     {} W'.format(P[0][0]))
        print('Prop Efficiency: {}'.format(etap[0][0]))

    # ----------------------------------------------------------------------------
    # 2D - Plots  Plots
    # ----------------------------------------------------------------------------

    if plots:
        plt.figure(1)
        plt.plot(r_BEVW, va_BEVW, 'ro-', label='axial BEVW')
        plt.plot(r_BEVW, vt_BEVW, 'bo-', label='tangential BEVW')
        plt.xlabel('Radial Location')
        plt.ylabel('Velocity')
        plt.legend(loc='lower right')

        plt.figure(2)
        plt.plot(r_BEVW, T_distribution_BEVW, 'ro-')
        plt.xlabel('Radial Location')
        plt.ylabel('Thrust, N')

        plt.figure(3)
        plt.plot(r_BEVW, Q_distribution_BEVW, 'ro-')
        plt.xlabel('Radial Location')
        plt.ylabel('Torque, N-m')

        plt.figure(4)
        plt.plot(r_BEVW, va_ind_BEVW, 'ro-', label='Axial')
        plt.plot(r_BEVW, vt_ind_BEVW, 'bo-', label='Tangential')
        plt.xlabel('Radial Location')
        plt.ylabel('Induced Velocity') 

    # Pack Results

    results                             = Data()
    results.thrust                      = F[0][0]
    results.torque                      = Q[0][0]
    results.power                       = P[0][0]
    results.power_coefficient           = Cp[0][0]
    results.efficiency                  = etap[0][0]
    results.induced_axial_velocity      = va_ind_BEVW
    results.induced_tangential_velocity = vt_ind_BEVW
    results.radial_distribution         = r_BEVW
    results.thrust_distribution         = T_distribution_BEVW
    results.torque_distribution         = Q_distribution_BEVW
    results.tangential_velocity         = vt_BEVW
    results.axial_velocity              = va_BEVW
    results.outputs                     = outputs

    return net, results, conditions