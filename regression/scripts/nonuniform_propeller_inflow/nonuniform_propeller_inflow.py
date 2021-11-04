# nonuniform_propeller_inflow.py
#
# Created:   Mar 2021, R. Erhard
# Modified:

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Plots.Performance.Propeller_Plots import *
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_wake import compute_wing_wake
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_propeller_nonuniform_freestream import compute_propeller_nonuniform_freestream


import numpy as np
import pylab as plt



def main():
    '''
    This example shows a propeller operating in three cases of nonuniform freestream flow:
    First, a propeller operates at a nonzero thrust angle relative to the freestream.
    Second, a propeller operates with an arbitrary upstream disturbance.
    Third, a propeller operates in the wake of an upstream wing

    '''
    # setup a simple vehicle
    vehicle = vehicle_setup(Na=24, Nr=101)

    # setup the atmospheric conditions
    conditions = test_conditions()

    case_1(vehicle, conditions)
    case_2(vehicle, conditions)
    case_3(vehicle, conditions)


    return

def case_1(vehicle, conditions):
    #-------------------------------------------------------------
    # test propeller at inclined thrust angle
    #-------------------------------------------------------------
    # set operating conditions for propeller test
    prop = vehicle.networks.prop_net.propeller
    prop.inputs.omega = np.ones_like(conditions.aerodynamics.angle_of_attack)*prop.angular_velocity
    prop.orientation_euler_angles  = [0.,20.*Units.degrees,0]
    prop.use_2d_analysis           = True
    
    # spin propeller in nonuniform flow
    thrust, torque, power, Cp, outputs , etap = prop.spin(conditions)

    # plot velocities at propeller plane and resulting performance
    plot_propeller_disc_performance(prop,outputs,title='Case 1: Operating at Thrust Angle')

    thrust   = np.linalg.norm(thrust)
    thrust_r = 845.7318746871123
    torque_r = 445.93087432
    power_r  = 60707.10354738
    Cp_r     = 0.27953017
    etap_r   = 1.02414969
    print('\nCase 1 Errors: \n')
    print('Thrust difference = ', np.abs(thrust - thrust_r) / thrust_r )
    print('Torque difference = ', np.abs(torque - torque_r) / torque_r )
    print('Power difference = ', np.abs(power - power_r) / power_r )
    print('Cp difference = ', np.abs(Cp - Cp_r) / Cp_r )
    print('Etap difference = ', np.abs(etap - etap_r) / etap_r )
    assert (np.abs(thrust - thrust_r) / thrust_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Thrust Test"
    assert (np.abs(torque - torque_r) / torque_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Torque Test"
    assert (np.abs(power - power_r) / power_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Power Test"
    assert (np.abs(Cp - Cp_r) / Cp_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Power Coefficient Test"
    assert (np.abs(etap - etap_r) / etap_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Efficiency Test"

    return

def case_2(vehicle,conditions, Na=24, Nr=101):
    #-------------------------------------------------------------
    # test propeller in arbitrary nonuniform freestream disturbance
    #-------------------------------------------------------------
    prop = vehicle.networks.prop_net.propeller
    prop.nonuniform_freestream  = True
    prop.orientation_euler_angles  = [0,0,0]
    ctrl_pts                    = len(conditions.aerodynamics.angle_of_attack)

    # azimuthal distribution
    psi            = np.linspace(0,2*np.pi,Na+1)[:-1]
    psi_2d         = np.tile(np.atleast_2d(psi),(Nr,1))
    psi_2d         = np.repeat(psi_2d[None,:,:], ctrl_pts, axis=0)

    # set an arbitrary nonuniform freestream disturbance
    va = (1+psi_2d) * 1.1
    vt = (1+psi_2d) * 2.0
    vr = (1+psi_2d) * 0.9

    prop.tangential_velocities_2d = vt
    prop.axial_velocities_2d      = va
    prop.radial_velocities_2d     = vr

    # spin propeller in nonuniform flow
    thrust, torque, power, Cp, outputs , etap = prop.spin(conditions)

    # plot velocities at propeller plane and resulting performance
    plot_propeller_disc_performance(prop,outputs,title='Case 2: Arbitrary Freestream')

    # expected results
    thrust   = np.linalg.norm(thrust)
    thrust_r = 77.97739689728701
    torque_r = 60.25485125
    power_r  = 8202.83524862
    Cp_r     = 0.03777054
    etap_r   = 0.74368527
    print('\nCase 2 Errors: \n')
    print('Thrust difference = ', np.abs(thrust - thrust_r) / thrust_r )
    print('Torque difference = ', np.abs(torque - torque_r) / torque_r )
    print('Power difference = ', np.abs(power - power_r) / power_r )
    print('Cp difference = ', np.abs(Cp - Cp_r) / Cp_r )
    print('Etap difference = ', np.abs(etap - etap_r) / etap_r )
    assert (np.abs(thrust - thrust_r) / thrust_r < 1e-6), "Nonuniform Propeller Inflow Regression Failed at Thrust Test"
    assert (np.abs(torque - torque_r) / torque_r < 1e-6), "Nonuniform Propeller Inflow Regression Failed at Torque Test"
    assert (np.abs(power - power_r) / power_r < 1e-6), "Nonuniform Propeller Inflow Regression Failed at Power Test"
    assert (np.abs(Cp - Cp_r) / Cp_r < 1e-6), "Nonuniform Propeller Inflow Regression Failed at Power Coefficient Test"
    assert (np.abs(etap - etap_r) / etap_r < 1e-6), "Nonuniform Propeller Inflow Regression Failed at Efficiency Test"



    return

def case_3(vehicle,conditions):
    '''
    This example shows a propeller operating in a nonuniform freestream flow.
    A wing in front of the propeller produces a wake, which is accounted for in the propeller analysis.
    '''
    # set plot flag
    plot_flag = True


    # grid and VLM settings
    grid_settings, VLM_settings = simulation_settings(vehicle)

    #--------------------------------------------------------------------------------------
    # Part 1. Compute the velocities induced by the wing at the propeller plane downstream
    #--------------------------------------------------------------------------------------
    prop_loc      = vehicle.networks.prop_net.propeller.origin
    prop_x_center = np.array([vehicle.wings.main_wing.origin[0][0] + prop_loc[0][0]])
    wing_wake, _  = compute_wing_wake(vehicle,conditions,prop_x_center[0], grid_settings, VLM_settings, plot_grid=plot_flag, plot_wake=plot_flag)


    #--------------------------------------------------------------------------------------
    # Part 2. Compute and run the propeller in this nonuniform flow
    #--------------------------------------------------------------------------------------
    prop = compute_propeller_nonuniform_freestream(vehicle.networks.prop_net.propeller, wing_wake,conditions)
    prop.nonuniform_freestream = True
    thrust, torque, power, Cp, outputs , etap = prop.spin(conditions)

    thrust   = np.linalg.norm(thrust)
    thrust_r, torque_r, power_r, Cp_r, etap_r = 619.066989094471, 389.67529537, 53048.71195988, 0.24426656, 0.91295051
    print('\nCase 3 Errors: \n')
    print('Thrust difference = ', np.abs(thrust - thrust_r) / thrust_r )
    print('Torque difference = ', np.abs(torque - torque_r) / torque_r )
    print('Power difference = ', np.abs(power - power_r) / power_r )
    print('Cp difference = ', np.abs(Cp - Cp_r) / Cp_r )
    print('Etap difference = ', np.abs(etap - etap_r) / etap_r )
    assert (np.abs(thrust - thrust_r) / thrust_r < 1e-5), "Nonuniform Propeller Inflow Regression Failed at Thrust Test"
    assert (np.abs(torque - torque_r) / torque_r < 1e-5), "Nonuniform Propeller Inflow Regression Failed at Torque Test"
    assert (np.abs(power - power_r) / power_r < 1e-5), "Nonuniform Propeller Inflow Regression Failed at Power Test"
    assert (np.abs(Cp - Cp_r) / Cp_r < 1e-5), "Nonuniform Propeller Inflow Regression Failed at Power Coefficient Test"
    assert (np.abs(etap - etap_r) / etap_r < 1e-5), "Nonuniform Propeller Inflow Regression Failed at Efficiency Test"

    # Plot results
    if plot_flag:
        plot_propeller_disc_performance(prop,outputs, title='Case 3: Pusher Propeller')

    return

def test_conditions():
    # --------------------------------------------------------------------------------------------------
    # Atmosphere Conditions:
    # --------------------------------------------------------------------------------------------------
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmosphere.compute_values(altitude=14000 * Units.ft)
    rho = atmo_data.density
    mu = atmo_data.dynamic_viscosity
    T = atmo_data.temperature
    a = atmo_data.speed_of_sound


    # aerodynamics analyzed for a fixed angle of attack
    aoa   = np.array([[ 3 * Units.deg  ]])
    Vv    = np.array([[ 175 * Units.mph]])
    ones  = np.ones_like(aoa)

    mach  = Vv/a

    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    conditions.freestream.density           = rho* ones
    conditions.freestream.dynamic_viscosity = mu* ones
    conditions.freestream.speed_of_sound    = a* ones
    conditions.freestream.temperature       = T* ones
    conditions.freestream.mach_number       = mach* ones
    conditions.freestream.velocity          = Vv * ones
    conditions.aerodynamics.angle_of_attack = aoa
    conditions.frames.body.transform_to_inertial = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )

    velocity_vector = np.zeros([len(aoa), 3])
    velocity_vector[:, 0] = Vv
    conditions.frames.inertial.velocity_vector = velocity_vector
    conditions.propulsion.throttle = np.ones_like(velocity_vector)

    return conditions



def vehicle_setup(Na, Nr):
    #-----------------------------------------------------------------
    #   Vehicle Initialization:
    #-----------------------------------------------------------------
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'simple_vehicle'

    # ------------------------------------------------------------------
    #   Wing Properties
    # ------------------------------------------------------------------
    wing = basic_wing()
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #   Propulsion Properties
    # ------------------------------------------------------------------
    net                          = SUAVE.Components.Energy.Networks.Battery_Propeller()
    net.tag                      = 'prop_net'
    net.number_of_engines        = 2

    prop = SUAVE.Components.Energy.Converters.Propeller()
    prop = basic_prop()

    # adjust propeller location and rotation:
    prop.rotation = [-1]
    prop.origin  = np.array([[(0.7+0.2), -2., 0.],
                             [(0.7+0.2),  2., 0.]])
    
    net.propeller = prop
    vehicle.append_component(net)

    return vehicle

def basic_prop(Na=24, Nr=101):

    # Design the Propeller
    prop = SUAVE.Components.Energy.Converters.Propeller()

    prop.number_of_blades          = 2
    prop.freestream_velocity       = 135.   * Units['mph']
    prop.angular_velocity          = 1300.  * Units.rpm
    prop.tip_radius                = 38.    * Units.inches
    prop.hub_radius                = 8.     * Units.inches
    prop.design_Cl                 = 0.8
    prop.design_altitude           = 12000. * Units.feet
    prop.design_thrust             = 1200.
    prop.origin                    = [[0.,0.,0.]]
    prop.number_azimuthal_stations = Na
    prop.rotation                  = [1]
    prop.symmetry                  = True

    prop.airfoil_geometry          =  ['../Vehicles/Airfoils/NACA_4412.txt']
    prop.airfoil_polars            = [['../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
                                       '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
                                       '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
                                       '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
                                       '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ]]

    prop.airfoil_polar_stations    = list(np.zeros(Nr).astype(int))
    prop                           = propeller_design(prop,Nr)

    return prop

def basic_wing():
    #-------------------------------------------------------------------------
    #          Variables:
    #-------------------------------------------------------------------------
    span         = 9.6
    croot        = 0.7
    ctip         = 0.3
    sweep_le     = 0.0
    sref         = span*(croot+ctip)/2
    twist_root   = 0.0 * Units.degrees
    twist_tip    = 0.0 * Units.degrees
    dihedral     = 0.0 * Units.degrees
    AR           = span**2/sref
    # ------------------------------------------------------------------
    # Initialize the Main Wing
    # ------------------------------------------------------------------
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.aspect_ratio            = AR
    wing.spans.projected         = span
    wing.chords.root             = croot
    wing.chords.tip              = ctip
    wing.areas.reference         = sref
    wing.twists.root             = twist_root
    wing.twists.tip              = twist_tip
    wing.sweeps.leading_edge     = sweep_le #45. * Units.degrees
    wing.dihedral                = dihedral #0. * Units.degrees
    wing.span_efficiency         = 0.8
    wing.origin                  = np.array([[0.,0.,0.]])
    wing.vertical                = False
    wing.symmetric               = True


    return wing


def simulation_settings(vehicle):

    grid_settings = Data()
    grid_settings.height = 12*vehicle.networks.prop_net.propeller.tip_radius / vehicle.wings.main_wing.spans.projected
    grid_settings.length = 1.2
    grid_settings.height_fine = 0.2

    VLM_settings        = SUAVE.Analyses.Aerodynamics.Vortex_Lattice().settings
    VLM_settings.number_spanwise_vortices        = 16
    VLM_settings.number_chordwise_vortices       = 4
    VLM_settings.use_surrogate                   = True
    VLM_settings.propeller_wake_model            = False
    VLM_settings.use_bemt_wake_model             = False
    VLM_settings.model_fuselage                  = False
    VLM_settings.model_nacelle                   = False
    VLM_settings.spanwise_cosine_spacing         = True
    VLM_settings.number_of_wake_timesteps        = 0.
    VLM_settings.leading_edge_suction_multiplier = 1.
    VLM_settings.initial_timestep_offset         = 0.
    VLM_settings.wake_development_time           = 0.

    return grid_settings, VLM_settings

if __name__ == '__main__':
    main()
    plt.show()
