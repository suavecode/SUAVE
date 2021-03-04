# nonuniform_propeller_inflow_regression.py
#
# Created: Mar 2021, R. Erhard
# Modified:

import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Propulsion import propeller_design 
from SUAVE.Plots.Propeller_Plots import plot_propeller_performance

import numpy as np
import pylab as plt



def main():
    '''
    This example shows a propeller operating in a nonuniform freestream flow caused by thrust angle.
    The thrust angle contributes to disturbances in the tangential and radial velocities at the blades.
    
    '''
    # setup basic propeller
    Nr = 101
    Na = 24
    prop    = basic_prop(Na, Nr)
    
    # setup the atmospheric conditions
    conditions = test_conditions()
    
    # set operating conditions for propeller test
    prop.inputs.omega = np.ones_like(conditions.aerodynamics.angle_of_attack)*prop.angular_velocity
    prop.thrust_angle = 20. * Units.deg
    
    # spin propeller in nonuniform flow
    thrust, torque, power, Cp, outputs , etap = prop.spin(conditions)
    
    
    # expected results
    thrust_r = 847.83217329
    torque_r = 446.59326314
    power_r  = 60797.27830077
    Cp_r     = 0.27994538
    etap_r   = 1.02517027
    
    assert (np.abs(thrust - thrust_r) / thrust_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Thrust Test"
    assert (np.abs(torque - torque_r) / torque_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Torque Test"
    assert (np.abs(power - power_r) / power_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Power Test"
    assert (np.abs(Cp - Cp_r) / Cp_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Power Coefficient Test"
    assert (np.abs(etap - etap_r) / etap_r < 1e-6), "Nonuniform Propeller Thrust Angle Regression Failed at Efficiency Test"    
    
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

def basic_prop(Na, Nr):
 
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
    prop.symmetry                  = True
    prop.airfoil_geometry          =  ['../Vehicles/NACA_4412.txt'] 
    prop.airfoil_polars            = [['../Vehicles/NACA_4412_polar_Re_50000.txt' ,
                                       '../Vehicles/NACA_4412_polar_Re_100000.txt' ,
                                       '../Vehicles/NACA_4412_polar_Re_200000.txt' ,
                                       '../Vehicles/NACA_4412_polar_Re_500000.txt' ,
                                       '../Vehicles/NACA_4412_polar_Re_1000000.txt' ]]
    
    prop.airfoil_polar_stations    = list(np.zeros(Nr).astype(int))
    prop                           = propeller_design(prop,Nr)
    prop.rotation = 1
    
    return prop



if __name__ == '__main__':
    main()
    plt.show()
