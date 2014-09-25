# test_aerodynamics
#
# Created:  Tim MacDonald - 09/09/14
# Modified: Tim MacDonald - 09/10/14

import SUAVE
from SUAVE.Attributes import Units
from SUAVE.Structure import Data
#from SUAVE.Methods.Aerodynamics.Lift import compute_aircraft_lift
#from SUAVE.Methods.Aerodynamics.Drag import compute_aircraft_drag

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_aircraft_lift
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag import compute_aircraft_drag

import numpy as np
import pylab as plt

import copy, time
from copy import deepcopy
import random

def main():
    
    vehicle = vehicle_setup() # Create the vehicle for testing
    
    test_num = 11 # Length of arrays used in this test
    
    # --------------------------------------------------------------------
    # Test Lift Surrogate
    # --------------------------------------------------------------------    
    
    AoA = np.linspace(-.174,.174,test_num) # +- 10 degrees
    
    lift_model = vehicle.configs.cruise.aerodynamics_model.configuration.surrogate_models.lift_coefficient
    
    wing_lift = lift_model(AoA)
    
    wing_lift_r = np.array([-0.79420805, -0.56732369, -0.34043933, -0.11355497,  0.11332939,
                            0.34021374,  0.5670981 ,  0.79398246,  1.02086682,  1.24775117,
                            1.47463553])
    
    surg_test = np.abs((wing_lift-wing_lift_r)/wing_lift)
    
    print 'Surrogate Test Results \n'
    print surg_test
    
    assert(np.max(surg_test)<1e-4), 'Aero regression failed at surrogate test'

    
    # --------------------------------------------------------------------
    # Initialize variables needed for CL and CD calculations
    # Use a seeded random order for values
    # --------------------------------------------------------------------
    
    random.seed(1)
    Mc = np.linspace(0.05,0.9,test_num)
    random.shuffle(Mc)
    rho = np.linspace(0.3,1.3,test_num)
    random.shuffle(rho)
    mu = np.linspace(5*10**-6,20*10**-6,test_num)
    random.shuffle(mu)
    T = np.linspace(200,300,test_num)
    random.shuffle(T)
    pressure = np.linspace(10**5,10**6,test_num)

    
    conditions = Data()
    
    conditions.freestream = Data()
    conditions.freestream.mach_number = Mc
    conditions.freestream.density = rho
    conditions.freestream.viscosity = mu
    conditions.freestream.temperature = T
    conditions.freestream.pressure = pressure
    
    conditions.aerodynamics = Data()
    conditions.aerodynamics.angle_of_attack = AoA
    conditions.aerodynamics.lift_breakdown = Data()
    
    configuration = vehicle.configs.cruise.aerodynamics_model.configuration
    
    conditions.aerodynamics.drag_breakdown = Data()

    geometry = Data()
    for k in ['fuselages','wings','propulsors']:
        geometry[k] = deepcopy(vehicle[k])    
    geometry.reference_area = vehicle.reference_area  
    #geometry.wings[0] = Data()
    #geometry.wings[0].vortex_lift = False
    
    # --------------------------------------------------------------------
    # Test compute Lift
    # --------------------------------------------------------------------
    
    compute_aircraft_lift(conditions, configuration, geometry) 
    
    lift = conditions.aerodynamics.lift_breakdown.total
    lift_r = np.array([-2.07712357, -0.73495391, -0.38858687, -0.1405849 ,  0.22295808,
                       0.5075275 ,  0.67883681,  0.92787301,  1.40470556,  2.08126751,
                       1.69661601])
    
    lift_test = np.abs((lift-lift_r)/lift)
    
    print '\nCompute Lift Test Results\n'
    print lift_test
        
    assert(np.max(lift_test)<1e-4), 'Aero regression failed at compute lift test'    
    
    
    # --------------------------------------------------------------------
    # Test compute drag 
    # --------------------------------------------------------------------
    
    compute_aircraft_drag(conditions, configuration, geometry)
    
    # Pull calculated values
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    # Only one wing is evaluated since they rely on the same function
    cd_c           = drag_breakdown.compressible['Main Wing'].compressibility_drag
    cd_i           = drag_breakdown.induced.total
    cd_m           = drag_breakdown.miscellaneous.total
    cd_m_fuse_base = drag_breakdown.miscellaneous.fuselage_base
    cd_m_fuse_up   = drag_breakdown.miscellaneous.fuselage_upsweep
    cd_m_nac_base  = drag_breakdown.miscellaneous.nacelle_base['Turbo Fan']
    cd_m_ctrl      = drag_breakdown.miscellaneous.control_gaps
    cd_p_fuse      = drag_breakdown.parasite.Fuselage.parasite_drag_coefficient
    cd_p_wing      = drag_breakdown.parasite['Main Wing'].parasite_drag_coefficient
    cd_tot         = drag_breakdown.total
    
    (cd_c_r, cd_i_r, cd_m_r, cd_m_fuse_base_r, cd_m_fuse_up_r, cd_m_nac_base_r, cd_m_ctrl_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r) = reg_values()
    
    drag_tests = Data()
    drag_tests.cd_c = np.abs((cd_c-cd_c_r)/cd_c)
    drag_tests.cd_i = np.abs((cd_i-cd_i_r)/cd_i)
    drag_tests.cd_m = np.abs((cd_m-cd_m_r)/cd_m)
    # Line below is not normalized since regression values are 0, insert commented line if this changes
    drag_tests.cd_m_fuse_base = np.abs((cd_m_fuse_base-cd_m_fuse_base_r)) # np.abs((cd_m_fuse_base-cd_m_fuse_base_r)/cd_m_fuse_base)
    drag_tests.cd_m_fuse_up   = np.abs((cd_m_fuse_up - cd_m_fuse_up_r)/cd_m_fuse_up)
    drag_tests.cd_m_ctrl      = np.abs((cd_m_ctrl - cd_m_ctrl_r)/cd_m_ctrl)
    drag_tests.cd_p_fuse      = np.abs((cd_p_fuse - cd_p_fuse_r)/cd_p_fuse)
    drag_tests.cd_p_wing      = np.abs((cd_p_wing - cd_p_wing_r)/cd_p_wing)
    drag_tests.cd_tot         = np.abs((cd_tot - cd_tot_r)/cd_tot)
    
    print '\nCompute Drag Test Results\n'
    print drag_tests
    
    for i, tests in drag_tests.items():
        assert(np.max(tests)<1e-4),'Aero regression test failed at ' + i
    
    return conditions, configuration, geometry, test_num
    

def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Boeing 737-800'    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 79015.8   # kg
    vehicle.mass_properties.operating_empty           = 62746.4   # kg
    vehicle.mass_properties.takeoff                   = 79015.8   # kg
    vehicle.mass_properties.max_zero_fuel             = 0.9 * vehicle.mass_properties.max_takeoff 
    vehicle.mass_properties.cargo                     = 10000.  * Units.kilogram   
    
    vehicle.mass_properties.center_of_gravity         = [60 * Units.feet, 0, 0]  # Not correct
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct
    
    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area        = 124.862       
    vehicle.passengers = 170
    vehicle.systems.control  = "fully powered" 
    vehicle.systems.accessories = "medium range"
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    
    wing.areas.reference = 124.862    #
    wing.aspect_ratio    = 10.18       #
    wing.spans.projected = 35.66      #
    wing.sweep           = 25 * Units.deg
    wing.symmetric       = True
    wing.thickness_to_chord = 0.1
    wing.taper           = 0.16
    
    
    # size the wing planform ----------------------------------
    # These can be determined by the wing sizing function
    # Note that the wing sizing function will overwrite span
    wing.chords.root  = 6.81
    wing.chords.tip   = 1.09
    wing.areas.wetted = wing.areas.reference*2.0 
    # The span that would normally be overwritten here doesn't match
    # ---------------------------------------------------------
    
    wing.chords.mean_aerodynamic = 12.5
    wing.areas.exposed = 0.8*wing.areas.wetted
    wing.areas.affected = 0.6*wing.areas.wetted
    wing.span_efficiency = 0.9
    wing.twists.root = 3.0*Units.degrees
    wing.twists.tip  = 3.0*Units.degrees
    wing.origin          = [20,0,0]
    wing.aerodynamic_center = [3,0,0] 
    wing.vertical   = False
    wing.eta         = 1.0
    
    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'
    
    wing.areas.reference = 32.488    #
    wing.aspect_ratio    = 6.16      #
    wing.spans.projected = 14.146      #
    wing.sweep           = 30 * Units.deg
    wing.symmetric       = True
    wing.thickness_to_chord = 0.08
    wing.taper           = 0.4
    
    # size the wing planform ----------------------------------
    # These can be determined by the wing sizing function
    # Note that the wing sizing function will overwrite span
    wing.chords.root  = 3.28
    wing.chords.tip   = 1.31
    wing.areas.wetted = wing.areas.reference*2.0 
    # ---------------------------------------------------------
    
    wing.chords.mean_aerodynamic = 8.0
    wing.areas.exposed = 0.8*wing.areas.wetted
    wing.areas.affected = 0.6*wing.areas.wetted
    wing.span_efficiency = 0.9
    wing.twists.root = 3.0*Units.degrees
    wing.twists.tip  = 3.0*Units.degrees  
    wing.origin          = [50,0,0]
    wing.aerodynamic_center = [2,0,0]
    wing.vertical   = False 
    wing.eta         = 0.9  
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'    
    
    wing.areas.reference = 32.488    #
    wing.aspect_ratio    = 1.91      #
    wing.spans.projected = 7.877      #
    wing.sweep           = 25 * Units.deg
    wing.symmetric       = False
    wing.thickness_to_chord = 0.08
    wing.taper           = 0.25
    
    # size the wing planform ----------------------------------
    # These can be determined by the wing sizing function
    # Note that the wing sizing function will overwrite span
    wing.chords.root  = 6.60
    wing.chords.tip   = 1.65
    wing.areas.wetted = wing.areas.reference*2.0 
    # ---------------------------------------------------------
    
    wing.chords.mean_aerodynamic = 8.0
    wing.areas.exposed = 0.8*wing.areas.wetted
    wing.areas.affected = 0.6*wing.areas.wetted
    wing.span_efficiency = 0.9
    wing.twists.root = 0.0*Units.degrees
    wing.twists.tip  = 0.0*Units.degrees  
    wing.origin          = [50,0,0]
    wing.aerodynamic_center = [2,0,0]    
    wing.vertical   = True 
    wing.t_tail     = False
    wing.eta         = 1.0
        
    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    
    fuselage.number_coach_seats = 200
    fuselage.seats_abreast = 6
    fuselage.seat_pitch = 1
    fuselage.fineness.nose = 1.6
    fuselage.fineness.tail = 2.
    fuselage.lengths.fore_space = 6.
    fuselage.lengths.aft_space  = 5.
    fuselage.width = 4.
    fuselage.heights.maximum          = 4.    #
    fuselage.areas.side_projected       = 4.* 59.8 #  Not correct
    fuselage.heights.at_quarter_length = 4. # Not correct
    fuselage.heights.at_three_quarters_length = 4. # Not correct
    fuselage.heights.at_wing_root_quarter_chord = 4. # Not correct
    fuselage.differential_pressure = 10**5   * Units.pascal    # Maximum differential pressure
    
    # size fuselage planform
    # A function exists to do this
    fuselage.lengths.nose  = 6.4
    fuselage.lengths.tail  = 8.0
    fuselage.lengths.cabin = 44.0
    fuselage.lengths.total = 58.4
    fuselage.areas.wetted  = 688.64
    fuselage.areas.front_projected = 12.57
    fuselage.effective_diameter        = 4.0
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    # ------------------------------------------------------------------
    #  Turbofan
    # ------------------------------------------------------------------    
    
    turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    turbofan.tag = 'Turbo Fan'
    
    turbofan.propellant = SUAVE.Attributes.Propellants.Jet_A()
    
    #turbofan.analysis_type                 = '1D'     #
    turbofan.diffuser_pressure_ratio       = 0.98     #
    turbofan.fan_pressure_ratio            = 1.7      #
    turbofan.fan_nozzle_pressure_ratio     = 0.99     #
    turbofan.lpc_pressure_ratio            = 1.14     #
    turbofan.hpc_pressure_ratio            = 13.415   #
    turbofan.burner_pressure_ratio         = 0.95     #
    turbofan.turbine_nozzle_pressure_ratio = 0.99     #
    turbofan.Tt4                           = 1450.0   #
    turbofan.bypass_ratio                  = 5.4      #
    turbofan.thrust.design                 = 25000.0  #
    turbofan.number_of_engines             = 2.0      #
    
    # size the turbofan
    turbofan.A2          =   1.753
    turbofan.df          =   1.580
    turbofan.nacelle_dia =   1.580
    turbofan.A2_5        =   0.553
    turbofan.dhc         =   0.857
    turbofan.A7          =   0.801
    turbofan.A5          =   0.191
    turbofan.Ao          =   1.506
    turbofan.mdt         =   9.51
    turbofan.mlt         =  22.29
    turbofan.mdf         = 355.4
    turbofan.mdlc        =  55.53
    turbofan.D           =   1.494
    turbofan.mdhc        =  49.73  
    
    # add to vehicle
    vehicle.append_component(turbofan)    
    
    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero()
    aerodynamics.initialize(vehicle)
    
    # build stability model
    stability = SUAVE.Attributes.Flight_Dynamics.Fidelity_Zero()
    stability.initialize(vehicle)
    aerodynamics.stability = stability
    vehicle.aerodynamics_model = aerodynamics
    
    # ------------------------------------------------------------------
    #   Simple Propulsion Model
    # ------------------------------------------------------------------     
    
    vehicle.propulsion_model = vehicle.propulsors

    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------

    # --- Takeoff Configuration ---
    config = vehicle.new_configuration("takeoff")
    # this configuration is derived from the baseline vehicle

    # --- Cruise Configuration ---
    config = vehicle.new_configuration("cruise")
    # this configuration is derived from vehicle.configs.takeoff

    # --- Takeoff Configuration ---
    takeoff_config = vehicle.configs.takeoff
    takeoff_config.wings['Main Wing'].flaps_angle =  20. * Units.deg
    takeoff_config.wings['Main Wing'].slats_angle  = 25. * Units.deg
    # V2_V2_ratio may be informed by user. If not, use default value (1.2)
    takeoff_config.V2_VS_ratio = 1.21
    # CLmax for a given configuration may be informed by user. If not, is calculated using correlations
    takeoff_config.maximum_lift_coefficient = 2.
    #takeoff_config.max_lift_coefficient_factor = 1.0

    # --- Landing Configuration ---
    landing_config = vehicle.new_configuration("landing")
    landing_config.wings['Main Wing'].flaps_angle =  30. * Units.deg
    landing_config.wings['Main Wing'].slats_angle  = 25. * Units.deg
    # Vref_V2_ratio may be informed by user. If not, use default value (1.23)
    landing_config.Vref_VS_ratio = 1.23
    # CLmax for a given configuration may be informed by user
    landing_config.maximum_lift_coefficient = 2.
    #landing_config.max_lift_coefficient_factor = 1.0
    landing_config.mass_properties.landing = 0.85 * vehicle.mass_properties.takeoff
    

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return vehicle    

def reg_values():
    cd_c_r = np.array([  1.41429794e-08,   2.96579619e-09,   1.03047740e-22,   4.50771390e-09,
                         1.27784183e-03,   1.31214322e-04,   3.98984222e-09,   6.19742191e-11,
                         8.21182714e-05,   1.20217216e-03,   5.63926215e-14])
    
    cd_i_r = np.array([ 0.17295472,  0.02165349,  0.00605319,  0.00079229,  0.00199276,
                        0.01032588,  0.01847305,  0.03451317,  0.07910034,  0.17364551,
                        0.11539178])
    cd_m_r = np.array([ 0.00047933,  0.00047933,  0.00047933,  0.00047933,  0.00047933,
                        0.00047933,  0.00047933,  0.00047933,  0.00047933,  0.00047933,
                        0.00047933])
    
    cd_m_fuse_base_r = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
    
    cd_m_fuse_up_r   = np.array([  4.80530506e-05,   4.80530506e-05,   4.80530506e-05,
                                   4.80530506e-05,   4.80530506e-05,   4.80530506e-05,
                                   4.80530506e-05,   4.80530506e-05,   4.80530506e-05,
                                   4.80530506e-05,   4.80530506e-05])
    
    cd_m_nac_base_r = np.array([ 0.00033128,  0.00033128,  0.00033128,  0.00033128,  0.00033128,
                                0.00033128,  0.00033128,  0.00033128,  0.00033128,  0.00033128,
                                0.00033128])
    
    cd_m_ctrl_r     = np.array([ 0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,
                                 0.0001,  0.0001,  0.0001,  0.0001])
    
    cd_p_fuse_r     = np.array([  0.00861449,  0.01003034,  0.01543476,  0.00983168,  0.01004746,
                                  0.00840775,  0.01029339,  0.01273788,  0.01002575,  0.00900746,
                                  0.01043446])
    
    cd_p_wing_r     = np.array([ 0.00398269,  0.00401536,  0.00619387,  0.00388993,  0.00442375,
                                 0.00343623,  0.00405385,  0.00506457,  0.00406928,  0.00379353,
                                 0.00407611])
    
    cd_tot_r        = np.array([ 0.19368287,  0.03905116,  0.03209541,  0.01737741,  0.0213476 ,
                                 0.02507019,  0.03614299,  0.05658934,  0.09780619,  0.19398041,
                                 0.13518241])
    
    return cd_c_r, cd_i_r, cd_m_r, cd_m_fuse_base_r, cd_m_fuse_up_r, cd_m_nac_base_r, cd_m_ctrl_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r

if __name__ == '__main__':
    (conditions, configuration, geometry, test_num) = main()
    
    print 'Aero regression test passed!'
    
    # --------------------------------------------------------------------
    # Drag Polar
    # --------------------------------------------------------------------  
    
    # Cruise conditions (except Mach number)
    conditions.freestream.mach_number = np.array([0.2]*test_num)
    conditions.freestream.density = np.array([0.3804534]*test_num)
    conditions.freestream.viscosity = np.array([1.43408227e-05]*test_num)
    conditions.freestream.temperature = np.array([218.92391647]*test_num)
    conditions.freestream.pressure = np.array([23908.73408391]*test_num)
    
    compute_aircraft_lift(conditions, configuration, geometry) # geometry is third variable, not used
    CL = conditions.aerodynamics.lift_breakdown.total    
    
    compute_aircraft_drag(conditions, configuration, geometry)
    CD = conditions.aerodynamics.drag_breakdown.total
    
    plt.figure("Drag Polar")
    axes = plt.gca()     
    axes.plot(CD,CL,'bo-')
    axes.set_xlabel('$C_D$')
    axes.set_ylabel('$C_L$')
    
    
    plt.show(block=True) # here so as to not block the regression test