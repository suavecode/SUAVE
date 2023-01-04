# propeller_interactions.py
# 
# Created:  April 2021, R. Erhard
# Modified: Feb 2022, R. Erhard

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data

from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_One import Rotor_Wake_Fidelity_One
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_wake_induced_velocity import compute_wake_induced_velocity
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_propeller_nonuniform_freestream import compute_propeller_nonuniform_freestream
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_propeller_grid import generate_propeller_grid
from SUAVE.Visualization.Performance.Aerodynamics.Rotor import *

import numpy as np
import pylab as plt
import copy
import sys

sys.path.append('../Vehicles/Propellers') 
from APC_10x7_thin_electric import propeller_geometry

def main():
    '''
    This example shows the influence of one propeller wake on another nearby propeller.
    A propeller produces a wake, which is accounted for in the analysis of another propeller.
    '''
    #--------------------------------------------------------------------
    #    SETUP
    #--------------------------------------------------------------------
    # flag for plotting results
    plot_flag = True
    
    # set the basic propeller geometry
    vehicle = vehicle_setup()
    prop    = vehicle.networks.prop_net.propeller
    
    # set the atmospheric conditions
    conditions = simulation_conditions(prop)
    
    # set the grid and VLM settings
    grid_settings, VLM_settings = simulation_settings(vehicle)
    
    # generate the grid points at the downstream propeller:
    grid_points = generate_propeller_grid(prop, grid_settings, plot_grid=plot_flag)
    
    #--------------------------------------------------------------------
    #    ANALYSIS
    #--------------------------------------------------------------------    
    
    # run the BEVW for upstream isolated propeller
    T_iso, Q_iso, P_iso, Cp_iso, outputs_iso , etap_iso = prop.spin(conditions)
    
    conditions.noise.sources.propellers[prop.tag] = outputs_iso
    
    # compute the induced velocities from upstream propeller at the grid points on the downstream propeller
    propeller_wake = compute_propeller_wake_velocities(prop, grid_settings, grid_points, conditions, plot_velocities=plot_flag)
    
    # run the downstream propeller in the presence of this nonuniform flow
    T, Q, P, Cp, outputs , etap = run_downstream_propeller(prop, propeller_wake, conditions, plot_performance=plot_flag)
    
    # compare regression results:
    T_iso_true, Q_iso_true, P_iso_true, Cp_iso_true, etap_iso_true = 3.229745783054582, 0.07191487, 48.95089701,0.04572016, 0.5899077
    
    assert(abs(np.linalg.norm(T_iso)-T_iso_true)<1e-6)
    assert(abs(Q_iso-Q_iso_true)<1e-6)
    assert(abs(P_iso-P_iso_true)<1e-6)
    assert(abs(Cp_iso-Cp_iso_true)<1e-6)
    assert(abs(etap_iso-etap_iso_true)<1e-6)
    
    T_true, Q_true, P_true, Cp_true, etap_true = 3.449059963023848,0.07231234, 49.22145028, 0.04597286, 0.62650237

    assert(abs(np.linalg.norm(T)-T_true)<1e-6)
    assert(abs(Q-Q_true)<1e-6)
    assert(abs(P-P_true)<1e-6)
    assert(abs(Cp-Cp_true)<1e-6)
    assert(abs(etap-etap_true)<1e-6)    
    
    # Display plots:
    if plot_flag:
        plt.show()
    
    return

def run_downstream_propeller(prop, propeller_wake, conditions, plot_performance=False):
    # assess the inflow at the propeller
    prop_copy = copy.deepcopy(prop)
    prop_copy.nonuniform_freestream = True
    prop_copy.origin = np.array([prop.origin[1] ])# only concerned with the impact the upstream prop has on this one
    prop_copy.rotation = prop.rotation
    prop = compute_propeller_nonuniform_freestream(prop_copy, propeller_wake, conditions)
    
    # run the propeller in this nonuniform flow
    T, Q, P, Cp, outputs , etap = prop.spin(conditions)
    
    if plot_performance:
        plot_rotor_disc_performance(prop,outputs)
        
    return T, Q, P, Cp, outputs , etap

def compute_propeller_wake_velocities(prop,grid_settings,grid_points, conditions, plot_velocities=True):
    
    x_plane = prop.origin[1,0] #second propeller, x-location
    
    # generate the propeller wake distribution for the upstream propeller
    prop_copy                = copy.deepcopy(prop)
    cpts                     = 1 # only testing one condition
    
    props = SUAVE.Core.Container()
    props.append(prop_copy)
    
    WD = prop.Wake.vortex_distribution
    prop.start_angle = prop_copy.start_angle
    
    # compute the wake induced velocities:
    VD      = Data()
    VD.YC   = grid_points.ymesh
    VD.ZC   = grid_points.zmesh
    VD.XC   = x_plane*np.ones_like(VD.YC)
    VD.n_cp = np.size(VD.YC)
    
    V_ind   = compute_wake_induced_velocity(WD, VD, cpts)
    u       = V_ind[0,:,0]
    v       = V_ind[0,:,1]
    w       = V_ind[0,:,2]
    
    propeller_wake = Data()
    propeller_wake.u_velocities  = u
    propeller_wake.v_velocities  = v
    propeller_wake.w_velocities  = w
    propeller_wake.VD            = VD
    
    if plot_velocities:
        # plot the velocities input to downstream propeller
        plot_rotor_disc_inflow(prop,propeller_wake,grid_points)
        
    
    return propeller_wake


    
    
def simulation_conditions(prop):
    # --------------------------------------------------------------------------------------------------
    # Atmosphere Conditions:
    # --------------------------------------------------------------------------------------------------
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data  = atmosphere.compute_values(altitude=14000 * Units.ft)
    rho        = atmo_data.density
    mu         = atmo_data.dynamic_viscosity
    T          = atmo_data.temperature
    a          = atmo_data.speed_of_sound
    
    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    conditions.freestream.density           = rho
    conditions.freestream.dynamic_viscosity = mu
    conditions.freestream.speed_of_sound    = a
    conditions.freestream.temperature       = T
    
    # Set freestream operating conditions
    Vv    = np.array([[ 20 * Units.mph]])
    mach  = Vv/a

    conditions.freestream.mach_number = mach
    conditions.freestream.velocity    = Vv
    conditions.frames.body.transform_to_inertial = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    conditions.frames.inertial.velocity_vector = np.array([[Vv[0][0],0,0]])
    conditions.propulsion.throttle             = np.array([[1]])
        
    # Set propeller operating conditions
    prop.inputs.omega = np.array([[6500 * Units.rpm]]) 
    prop.rotation     = -1
    prop.origin       = np.array([[0., 0., 0.],
                                  [0., 0.01+2*prop.tip_radius, 0.]])     
    
    return conditions    

def simulation_settings(vehicle):
    
    # grid conditions for downstream propeller
    grid_settings            = Data()
    grid_settings.radius     = vehicle.networks.prop_net.propeller.tip_radius
    grid_settings.hub_radius = vehicle.networks.prop_net.propeller.hub_radius
    grid_settings.Nr         = 40
    grid_settings.Na         = 40
    
    # cartesian grid specs
    grid_settings.Ny         = 50
    grid_settings.Nz         = 50
    grid_settings.grid_mode  = 'cartesian'
    
    VLM_settings        = Data()
    VLM_settings.number_spanwise_vortices        = 10
    VLM_settings.number_chordwise_vortices       = 4
    VLM_settings.use_surrogate                   = True
    VLM_settings.propeller_wake_model            = False
    VLM_settings.model_fuselage                  = False
    VLM_settings.spanwise_cosine_spacing         = True
    VLM_settings.leading_edge_suction_multiplier = 1.
    
    return grid_settings, VLM_settings

def vehicle_setup():
    
    # Vehicle Initialization:
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'simple_vehicle'    
    
    # Propulsion Properties:
    net                         = SUAVE.Components.Energy.Networks.Battery_Rotor()
    net.tag                     = 'prop_net'
    net.number_of_rotor_engines = 2

    prop = SUAVE.Components.Energy.Converters.Propeller()
    prop = propeller_geometry() 
    
    # assign wake fidelity
    prop.Wake = Rotor_Wake_Fidelity_One()
    prop.Wake.number_rotor_rotations = 1
    
    net.propeller = prop
    vehicle.append_component(net)
    
    return vehicle



if __name__ == '__main__':
    main()
    plt.show()
