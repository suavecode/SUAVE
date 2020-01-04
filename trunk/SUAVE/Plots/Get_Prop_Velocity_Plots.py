

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import scipy.optimize
import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt
import math

from SUAVE.Core import Data
from SUAVE.Methods.Propulsion import propeller_design
#from compute_induced_velocity_at_prop import compute_induced_velocity_matrix
#from compute_vortex_distr_prop import compute_vortex_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import compute_induced_velocity_matrix
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import compute_mach_cone_matrix
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_vortex_distribution import compute_vortex_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_RHS_matrix              import compute_RHS_matrix
from SUAVE.Plots.Vehicle_Plots import plot_vehicle_vlm_panelization

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main(span,croot,sweep,radius,RPM,nprops,prop_location):
    #Sref                     = 174. * Units.feet**2 
    vortices                 = 1
    drag                     = 2050.0

    # --------------------------------------------------------------------------------
    #     Vehicle Inputs:
    # --------------------------------------------------------------------------------    
    wing_parameters          = Data()
    wing_parameters.span     = span * Units.feet #+ 1. * Units.inches#:D
    wing_parameters.croot    = croot * Units.inches#:D
    #ctip                     = wing_parameters.croot #((2*166.5 )/wing_parameters.span)*12.0 - wing_parameters.croot
    wing_parameters.ctip     = wing_parameters.croot
    wing_parameters.sweep    = sweep * Units.deg 
    wing_parameters.drag     = drag  
    
    # --------------------------------------------------------------------------------
    #   Propeller Inputs:
    # --------------------------------------------------------------------------------
    
    prop_parameters          = Data()
    prop_parameters.radius   = radius * Units.inches#:D
    prop_parameters.diameter = 2*prop_parameters.radius #:D
    prop_parameters.n_props  = nprops #:D # Number of props PER WING, change the prop.origin
    prop_parameters.RPM      = RPM * Units.rpm #1250.  * Units.rpm
    prop_parameters.location = prop_location # Location relative to TE of wing
    
    
    # unpack settings
    geometry, prop                   = vehicle_setup(wing_parameters,prop_parameters)
    settings                         = Data()
    settings.number_panels_spanwise  = vortices **2
    settings.number_panels_chordwise = vortices     
    
    n_sw       = settings.number_panels_spanwise    
    n_cw       = settings.number_panels_chordwise   
    Sref       = geometry.reference_area

    # --------------------------------------------------------------------------------    
    # Cruise conditions  
    # --------------------------------------------------------------------------------
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()  
    aoa = np.array([[2.0 * Units.deg]])
    mach  = np.array([0.0])
    rho = np.array([[0.365184]])   
    mu  = np.array([[0.0000143326]])   
    T   = np.array([[216.827]])
    P   = 22729.3    
    a   = 295.190
    pressure = np.array([[22729.3]]) 
    re       = rho*a*mach/mu    
    state.conditions.freestream.mach_number = mach
    state.conditions.freestream.density = rho
    state.conditions.freestream.dynamic_viscosity = mu
    state.conditions.freestream.temperature = T
    state.conditions.freestream.pressure = pressure
    state.conditions.freestream.reynolds_number = re
    state.conditions.aerodynamics.angle_of_attack =  aoa   
    state.conditions.frames.body.transform_to_inertial= np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])
    

    # --------------------------------------------------------------------------------
    #    Spin Propeller:
    # --------------------------------------------------------------------------------    
    F, Q, P, Cp ,  outputs  , etap  = prop.spin(state.conditions)
    geometry.propulsors.propulsor.propeller.outputs = outputs
    
    ones = np.atleast_2d(np.ones_like(aoa)) 

    # generate vortex distribution
    VD = compute_vortex_distribution(geometry,settings)
    #plot_vehicle_vlm_panelization(VD, save_figure = False, save_filename = "VLM_Panelization")
    #plt.show()
    C_mn, DW_mn = compute_induced_velocity_matrix(VD,n_sw,n_cw,aoa,mach)
    
    #--------------------------------------------------------------------------
    #     Getting Vortex Strength:
    #--------------------------------------------------------------------------
    # Compute flow tangency conditions   
    inv_root_beta = np.zeros_like(mach)
    inv_root_beta[mach<1] = 1/np.sqrt(1-mach[mach<1]**2)     
    inv_root_beta[mach>1] = 1/np.sqrt(mach[mach>1]**2-1) 
    if np.any(mach==1):
        raise('Mach of 1 cannot be used in building compressibiliy corrections.')
    inv_root_beta = np.atleast_2d(inv_root_beta)    
    
    phi   = np.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC))*ones
    delta = np.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH)*inv_root_beta))    
    
    # Build Aerodynamic Influence Coefficient Matrix
    A = C_mn[:,:,:,2] - np.multiply(C_mn[:,:,:,0],np.atleast_3d(np.tan(delta)))- np.multiply(C_mn[:,:,:,1],np.atleast_3d(np.tan(phi)))  # EDIT
    
    # Build the vector
    RHS = compute_RHS_matrix(VD,n_sw,n_cw,delta,state.conditions,geometry)
    
    # Compute vortex strength  
    gamma = np.linalg.solve(A,RHS)  
    
    # Compute induced velocities     
    u = np.dot(C_mn[:,:,:,0],gamma[:,:].T)[:,:,0]
    v = np.dot(C_mn[:,:,:,1],gamma[:,:].T)[:,:,0]
    w = np.sum(np.dot(C_mn[:,:,:,2],gamma[:,:].T)*tile_eye,axis=2)    
    
    #----------------------------------------------------------------------------------------------------------------------
    #      Updating Vortex Distribution Matrix to Solve for Arbitrary Control Point (x,y,z):  VD.X: inverse of xc hats, etc
    #----------------------------------------------------------------------------------------------------------------------
    XC_hat = prop_location[0]
    ZC_hat = prop_location[2]
    Cpoint = np.array([XC_hat,ZC_hat])
    
    A_mat = np.array([[np.cos(aoa), np.sin(aoa)], [-np.sin(aoa), np.cos(aoa)]])
    
    xz_mat = np.dot(np.linalg.inv(A_mat.reshape(2,2)), Cpoint.reshape(2,1))
    VD.XC = xz_mat[0]
    VD.YC = np.array([prop_location[1]])
    VD.ZC = xz_mat[1]
    
    
    # Build new induced velocity matrix, C_mn
    C_mn, DW_mn = compute_induced_velocity_matrix(VD,n_sw,n_cw,aoa,mach)
    MCM = VD.MCM
    
    
    # Do some matrix magic
    len_aoa = len(aoa)
    len_cps = VD.n_cp
    eye = np.eye(len_aoa)
    tile_eye = np.broadcast_to(eye,(len_cps,len_aoa,len_aoa))
    tile_eye =  np.transpose(tile_eye,axes=[1,0,2])
    
    # Compute induced velocities     
    u = np.dot(C_mn[:,:,:,0]*MCM[:,:,:,0],gamma[:,:].T)[:,:,0]
    v = np.dot(C_mn[:,:,:,1]*MCM[:,:,:,1],gamma[:,:].T)[:,:,0]
    w = np.sum(np.dot(C_mn[:,:,:,2]*MCM[:,:,:,2],gamma[:,:].T)*tile_eye,axis=2)
    
    return C_mn, u, v, w

    #spanX = np.linspace(0, x1[1]/2.0, num=len(Cl_dist))
    #plt.plot(spanX,Cl_dist)
    #plt.xlabel('Spanwise Location (m)')
    #plt.ylabel('Cl')
    #plt.title('Cl Distribution over Wing Half-Span')
    #plt.show()
    

    

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup(wing_parameters,prop_parameters):

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Cessna_172_SP'    


    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff   = 2550. * Units.pounds
    vehicle.mass_properties.takeoff       = 2550. * Units.pounds
    vehicle.mass_properties.max_zero_fuel = 2550. * Units.pounds
    vehicle.mass_properties.cargo         = 0. 

    # envelope properties
    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load    = 3.8

    # basic parameters
    vehicle.reference_area         = 174. * Units.feet**2 # :D 
    vehicle.passengers             = 4

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.sweeps.quarter_chord    = wing_parameters.sweep # 0.0* Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.9
    wing.areas.reference         = 174. * Units.feet**2 #:D 
    wing.spans.projected         = wing_parameters.span #:D 

    wing.chords.root             = wing_parameters.croot #:D 
    wing.chords.tip              = wing_parameters.ctip #:D 
    wing.chords.mean_aerodynamic = wing.chords.root-(2*(wing.chords.root-wing.chords.tip)*(0.5*wing.chords.root+wing.chords.tip) / (3*(wing.chords.root+wing.chords.tip))) #:D 
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [0.0* Units.inches,0,0] 
    wing.aerodynamic_center      = [0.25*wing.chords.mean_aerodynamic,0,0]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True

    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Piston Propeller Network
    # ------------------------------------------------------------------    
    
    # build network
    net = SUAVE.Components.Energy.Networks.Battery_Propeller()
    net.number_of_engines = prop_parameters.n_props #:D
    net.nacelle_diameter  = 42 * Units.inches
    net.engine_length     = 0.01 * Units.inches
    net.areas             = Data()
    net.rated_speed       = 2700. * Units.rpm
    net.areas.wetted      = 0.01

    # Component 1 the engine
    net.engine = SUAVE.Components.Energy.Converters.Internal_Combustion_Engine()
    net.engine.sea_level_power    = 180. * Units.horsepower
    net.engine.flat_rate_altitude = 0.0
    net.engine.speed              = 2700. * Units.rpm
    net.engine.BSFC               = 0.52


    # Design the Propeller    
    prop  = SUAVE.Components.Energy.Converters.Propeller()
    prop.number_blades       = 2.
    prop.freestream_velocity = 135.*Units['mph']    
    prop.angular_velocity    = prop_parameters.RPM #1250.  * Units.rpm
    prop.inputs.omega = prop.angular_velocity
    prop.tip_radius          = prop_parameters.radius #:D 
    prop.hub_radius          = prop.tip_radius * 0.239
    prop.design_Cl           = 0.8
    prop.design_altitude     = 12000. * Units.feet
    prop.design_thrust       = wing_parameters.drag/(2.0*prop_parameters.n_props) #0.0 # Want to design for thrust to match the drag from the wing?
    prop.design_power        = 0.0#180. * Units.horsepower #0.32 * 180. * Units.horsepower
    prop               = propeller_design(prop) #add in trouble later
    
    prop.origin = []
    for x in range(prop_parameters.n_props):
        xloc = (x+1)*(wing.spans.projected/2.0)/(prop_parameters.n_props+1) # half span, split between nprops
        prop.origin.append([2,xloc,0])
        
    net.propeller            = prop
    

    # add the network to the vehicle
    vehicle.append_component(net)      

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle, prop #add in trouble later

if __name__ == '__main__':    
    span = 8/Units.ft #64.0
    croot = 2/Units.inches #50.0
    sweep = 0.0
    radius = 50.0
    RPM = 1250.0
    nprops = 1
    #x_prop = 
    #y_prop = 
    #z_prop = np.zeros()
    
    # Check where these values should be referenced from
    
    prop_x = [1.2*radius*Units.inches +80*Units.inches +croot*Units.inches]
    prop_y = [0]#[span*Units.ft/6]
    #prop_x = np.linspace((radius)*Units.inches, 6*radius*Units.inches, 16) + 80.0*Units.inches + croot*Units.inches #np.array([radius*Units.inches, 2*radius*Units.inches, 3*radius*Units.inches]) + 80.* Units.inches
    #prop_y = np.linspace(0, (span/2)*Units.ft, 16) #np.array([span*Units.ft/2, span*Units.ft/2, span*Units.feet/2])
    print(prop_x)
    print(prop_y)
    prop_z = np.zeros(20)
    X, Y = np.meshgrid(prop_x,prop_y)
    

    
    u_pts = [[0 for i in range(len(prop_x)) ] for j in range(len(prop_y))]
    v_pts = [[0 for i in range(len(prop_x)) ] for j in range(len(prop_y))]
    w_pts = [[0 for i in range(len(prop_x)) ] for j in range(len(prop_y))]
    
    for xl in range(len(prop_x)):
        for yl in range(len(prop_y)):
            prop_loc = [prop_x[xl], prop_y[yl], prop_z[0]]
            C_mn, u,v,w = main(span, croot, sweep, radius, RPM, nprops, prop_loc)
            print("XL and YL:")
            print(xl)
            print(yl)
        
            u_pts[yl][xl] = u[0,0]
            v_pts[yl][xl] = v[0,0]
            w_pts[yl][xl] = w[0,0]

    #print("u-velocity (spanwise): ", u[0,0])
    #print("v-velocity (chordwise): ", v)
    #print("w-velocity (vertical): ", w[0,0])
    
    #fig = plt.figure()
    #axes = fig.add_subplot(1,3,1)
    #axes.plot(prop_x, u_pts, 'ro-', label="u-velocity")
    #axes.set_ylabel("u-velocity")
    #axes.set_xlabel("Location in x-direction (chordwise)")
    #axes = fig.add_subplot(1,3,2)
    #axes.plot(prop_x, v_pts, 'bo-', label="v-velocity")    
    #axes.legend(loc='lower right')
    #axes.set_ylabel("v-velocity")
    #axes.set_xlabel("Location in x-direction (chordwise)")    
    #axes = fig.add_subplot(1,3,3)
    #axes.plot(prop_x, w_pts, 'mo-', label="w-velocity")    
    #axes.legend(loc='lower right')
    #axes.set_ylabel("w-velocity")
    #axes.set_xlabel("Location in x-direction (chordwise)")       
    #plt.show()
    
    fig = plt.figure()
    axes = fig.add_subplot(3,1,1)
    a = axes.contourf(prop_x, prop_y, u_pts)#, 'ro-', label="u-velocity")
    axes.set_ylabel("Spanwise Location (y)")
    axes.set_title("u-velocities")
    plt.colorbar(a)
    
    axes = fig.add_subplot(3,1,2)
    b = axes.contourf(prop_x, prop_y, v_pts)#, 'ro-', label="u-velocity")
    axes.set_ylabel("Spanwise Location (y)")
    axes.set_title("v-velocities")
    plt.colorbar(b)
    
    axes = fig.add_subplot(3,1,3)
    c = axes.contourf(prop_x, prop_y, w_pts)#, 'ro-', label="u-velocity")
    axes.set_ylabel("Spanwise Location (y)")
    axes.set_xlabel("Chordwise Location(x)")   
    axes.set_title("w-velocities")
    plt.colorbar(c)
    
    plt.show()
    print(u_pts)