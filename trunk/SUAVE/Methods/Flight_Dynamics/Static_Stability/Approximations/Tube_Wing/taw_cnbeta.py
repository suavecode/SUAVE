# taw_cnbeta.py
#
# Created:  Tim Momose, March 2014
# Modified: Andrew Wendorff, July 2014
# 
# TO DO:
#    - Add capability for multiple vertical tails
#    - Smooth out k_v factor (line 143)
#    - Add effect of propellers

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
import copy
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area
from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def taw_cnbeta(geometry,conditions,configuration):
    """ CnBeta = SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cnbeta(configuration,conditions)
        This method computes the static directional stability derivative for a
        standard Tube-and-Wing aircraft configuration.        
        
        CAUTION: The correlations used in this method do not account for the
        destabilizing moments due to propellers. This can lead to higher-than-
        expected values of CnBeta, particularly for smaller prop-driven aircraft
        
        Inputs:
            geometry - aircraft geometrical features: a data dictionary with the fields:
                wings['Main Wing'] - the aircraft's main wing
                    areas.reference - wing reference area [meters**2]
                    spans.projected - span of the wing [meters]
                    sweep - sweep of the wing leading edge [radians]
                    aspect_ratio - wing aspect ratio [dimensionless]
                    origin - the position of the wing root in the aircraft body frame [meters]
                wings['Vertical Stabilizer']
                    spans.projected - projected span (height for a vertical tail) of
                     the exposed surface [meters]
                    areas.reference - area of the reference vertical tail [meters**2]
                    sweep - leading edge sweep of the aerodynamic surface [radians]
                    chords.root - chord length at the junction between the tail and 
                     the fuselage [meters]
                    chords.tip - chord length at the tip of the aerodynamic surface
                    [meters]
                    symmetric - Is the wing symmetric across the fuselage centerline?
                    origin - the position of the vertical tail root in the aircraft body frame [meters]
                    exposed_root_chord_offset - the displacement from the fuselage
                     centerline to the exposed area's physical root chordline [meters]
                     
                     

    x_v    = vert.origin[0]
    b_v    = vert.spans.projected
    ac_vLE = vert.aerodynamic_center[0]
    
                fuselages.Fuselage - a data dictionary with the fields:
                    areas.side_projected - fuselage body side area [meters**2]
                    lengths.total - length of the fuselage [meters]
                    heights.maximum - maximum height of the fuselage [meters]
                    width - maximum width of the fuselage [meters]
                    heights.at_quarter_length - fuselage height at 1/4 of the fuselage length [meters]
                    heights.at_three_quarters_length - fuselage height at 3/4 of fuselage 
                     length [meters]
                    heights.at_vertical_root_quarter_chord - fuselage height at the quarter 
                     chord of the vertical tail root [meters]
                vertical - a data dictionary with the fields below:
                NOTE: This vertical tail geometry will be used to define a reference
                 vertical tail that extends to the fuselage centerline.
                    
                    x_ac_LE - the x-coordinate of the vertical tail aerodynamic 
                    center measured relative to the tail root leading edge (root
                    of reference tail area - at fuselage centerline)
                    leading edge, relative to the nose [meters]
                    sweep_le - leading edge sweep of the vertical tail [radians]
                    span - height of the vertical tail [meters]
                    taper - vertical tail taper ratio [dimensionless]
                    aspect_ratio - vertical tail AR: bv/(Sv)^2 [dimensionless]
                    effective_aspect_ratio - effective aspect ratio considering
                    the effects of fuselage and horizontal tail [dimensionless]
                    symmetric - indicates whether the vertical panel is symmetric
                    about the fuselage centerline [Boolean]
                other_bodies - an list of data dictionaries containing bodies 
                such as nacelles if these are large enough to strongly influence
                stability. Each body data dictionary contains the same fields as
                the fuselage data dictionary (described above), except no value 
                is needed for 'height_at_vroot_quarter_chord'. CAN BE EMPTY LIST
                    x_front - This is the only new field needed: the x-coordinate 
                    of the nose of the body relative to the fuselage nose
                    
            conditions - a data dictionary with the fields:
                v_inf - true airspeed [meters/second]
                M - flight Mach number
                rho - air density [kg/meters**3]
                mew - air dynamic viscosity [kg/meter/second]
                
            configuration - a data dictionary with the fields:
                mass_properties - a data dictionary with the field:
                    center_of_gravity - A vector in 3-space indicating CG position [meters]
                other - a dictionary of aerodynamic bodies, other than the fuselage,
                whose effect on directional stability is to be included in the analysis
    
        Outputs:
            CnBeta - a single float value: The static directional stability 
            derivative
                
        Assumptions:
            -Assumes a tube-and-wing configuration with a single centered 
            vertical tail
            -Uses vertical tail effective aspect ratio, currently calculated by
            hand, using methods from USAF Stability and Control DATCOM
            -The validity of correlations for KN is questionable for sqrt(h1/h2)
            greater than about 4 or h_max/w_max outside [0.3,2].
            -This method assumes a small angle of attack, so the vertical tail AC
            z-position does not affect the sideslip derivative.
        
        Correlations:
            -Correlations are taken from Roskam's Airplane Design, Part VI.
    """         

    try:
        configuration.other
    except AttributeError:
        configuration.other = 0
    CnBeta_other = []

    # Unpack inputs
    S      = geometry.wings['Main Wing'].areas.reference
    b      = geometry.wings['Main Wing'].spans.projected
    sweep  = geometry.wings['Main Wing'].sweep
    AR     = geometry.wings['Main Wing'].aspect_ratio
    z_w    = geometry.wings['Main Wing'].origin[2]
    S_bs   = geometry.fuselages.Fuselage.areas.side_projected
    l_f    = geometry.fuselages.Fuselage.lengths.total
    h_max  = geometry.fuselages.Fuselage.heights.maximum
    w_max  = geometry.fuselages.Fuselage.width
    h1     = geometry.fuselages.Fuselage.heights.at_quarter_length
    h2     = geometry.fuselages.Fuselage.heights.at_three_quarters_length
    d_i    = geometry.fuselages.Fuselage.heights.at_vertical_root_quarter_chord
    other  = configuration.other
    vert   = extend_to_ref_area(geometry.wings['Vertical Stabilizer'])
    S_v    = vert.areas.reference
    x_v    = vert.origin[0]
    b_v    = vert.spans.projected
    ac_vLE = vert.aerodynamic_center[0]
    x_cg   = configuration.mass_properties.center_of_gravity[0]
    v_inf  = conditions.freestream.velocity
    mu     = conditions.freestream.viscosity
    rho    = conditions.freestream.density
    M      = conditions.freestream.mach_number
    
    #Compute wing contribution to Cn_beta
    CnBeta_w = 0.0    #The wing contribution is assumed to be zero except at very
                      #high angles of attack. 
    
    #Compute fuselage contribution to Cn_beta
    Re_fuse  = rho*v_inf*l_f/mu
    x1       = x_cg/l_f
    x2       = l_f**2.0/S_bs
    x3       = np.sqrt(h1/h2)
    x4       = h_max/w_max
    kN_1     = 3.2413*x1 - 0.663345 + 6.1086*np.exp(-0.22*x2)
    kN_2     = (-0.2023 + 1.3422*x3 - 0.1454*x3**2)*kN_1
    kN_3     = (0.7870 + 0.1038*x4 + 0.1834*x4**2 - 2.811*np.exp(-4.0*x4))
    K_N      = (-0.47899 + kN_3*kN_2)*0.001
    K_Rel    = 1.0+0.8*np.log(Re_fuse/1.0E6)/np.log(50.)  
        #K_Rel: Correction for fuselage Reynolds number. Roskam VI, page 400.
    CnBeta_f = -57.3*K_N*K_Rel*S_bs*l_f/S/b
    
    #Compute contributions of other bodies on CnBeta
    if other > 0:
        for body in other:
            #Unpack inputs
            S_bs   = body.areas.side_projected
            x_le   = body.origin[0]
            l_b    = body.lengths.total
            h_max  = body.heights.maximum
            w_max  = body.width
            h1     = body.heights.at_quarter_length
            h2     = body.heights.at_three_quarters_length 
            #Compute body contribution to Cn_beta
            x_cg_on_body = (x_cg-x_le)/l_b
            Re_body  = rho*v_inf*l_b/mew
            x1       = x_cg_on_body/l_b
            x2       = l_b**2.0/S_bs
            x3       = np.sqrt(h1/h2)
            x4       = h_max/w_max
            kN_1     = 3.2413*x1 - 0.663345 + 6.1086*np.exp(-0.22*x2)
            kN_2     = (-0.2023 + 1.3422*x3 - 0.1454*x3**2)*kN_1
            kN_3     = (0.7870 + 0.1038*x4 + 0.1834*x4**2 - 2.811*np.exp(-4.0*x4))
            K_N      = (-0.47899 + kN_3*kN_2)*0.001
            #K_Rel: Correction for fuselage Reynolds number. Roskam VI, page 400.
            K_Rel    = 1.0+0.8*np.log(Re_body/1.0E6)/np.log(50.)
            CnBeta_b = -57.3*K_N*K_Rel*S_bs*l_b/S/b
            CnBeta_other.append(CnBeta_b)
    
    #Compute vertical tail contribution
    l_v    = x_v + ac_vLE - x_cg
    #try:
    #    CLa_v  = geometry.wings['Vertical Stabilizer'].CL_alpha
    #except AttributeError:
    #    CLa_v  = datcom(geometry.wings['Vertical Stabilizer'], [M])
    try:
        iter(M)
    except TypeError:
        M = [M]
    CLa_v = datcom(vert,M)
    #k_v correlated from Roskam Fig. 10.12. NOT SMOOTH.
    bf     = b_v/d_i
    if bf < 2.0:
        k_v = 0.76
    elif bf < 3.5:
        k_v = 0.76 + 0.24*(bf-2.0)/1.5
    else:
        k_v = 1.0
    quarter_chord_sweep = convert_sweep(geometry.wings['Main Wing'])
    k_sweep  = (1.0+np.cos(quarter_chord_sweep))
    dsdb_e   = 0.724 + 3.06*((S_v/S)/k_sweep) + 0.4*z_w/h_max + 0.009*AR
    Cy_bv    = -k_v*CLa_v*dsdb_e*(S_v/S)  #ASSUMING SINGLE VERTICAL TAIL
    
    CnBeta_v = -Cy_bv*l_v/b
    
    CnBeta   = CnBeta_w + CnBeta_f + CnBeta_v + sum(CnBeta_other)
    
    ##print "Wing: {}  Fuse: {}   Vert: {}   Othr: {}".format(CnBeta_w,CnBeta_f,CnBeta_v,sum(CnBeta_other))
    
    return CnBeta


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line
if __name__ == '__main__':
    from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area
    from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x
    ##Parameters Required
    ##Using values for a Boeing 747-200
    #wing                = SUAVE.Components.Wings.Wing()
    #wing.area           = 5500.0 * Units.feet**2
    #wing.span           = 196.0  * Units.feet
    #wing.sweep_le          = 42.0   * Units.deg
    #wing.z_position     = 3.6    * Units.feet
    #wing.taper          = 14.7/54.5
    #wing.aspect_ratio   = wing.span**2/wing.area
    
    #fuselage            = SUAVE.Components.Fuselages.Fuselage()
    #fuselage.side_area  = 4696.16 * Units.feet**2
    #fuselage.length     = 229.7   * Units.feet
    #fuselage.h_max      = 26.9    * Units.feet
    #fuselage.w_max      = 20.9    * Units.feet
    #fuselage.height_at_vroot_quarter_chord   = 15.8 * Units.feet
    #fuselage.height_at_quarter_length        = 26   * Units.feet
    #fuselage.height_at_three_quarters_length = 19.7 * Units.feet
    
    #vertical              = SUAVE.Components.Wings.Wing()
    #vertical.span         = 32.4   * Units.feet
    #vertical.root_chord   = 38.7   * Units.feet
    #vertical.tip_chord    = 13.4   * Units.feet
    #vertical.sweep_le     = 50.0   * Units.deg
    #vertical.x_root_LE1   = 181.0  * Units.feet
    #dz_centerline         = 13.5   * Units.feet
    #ref_vertical          = extend_to_ref_area(vertical,dz_centerline)
    #vertical.span         = ref_vertical.ref_span
    #vertical.area         = ref_vertical.ref_area
    #vertical.aspect_ratio = ref_vertical.ref_aspect_ratio
    #vertical.x_root_LE    = vertical.x_root_LE1 + ref_vertical.root_LE_change
    #vertical.taper        = vertical.tip_chord/ref_vertical.ref_root_chord
    #vertical.effective_aspect_ratio = 2.25
    #vertical_symm         = copy.deepcopy(vertical)
    #vertical_symm.span    = 2.0*vertical.span
    #vertical_symm.area    = 2.0*vertical.area
    #vertical.x_ac_LE      = trapezoid_ac_x(vertical_symm)
    
    #aircraft            = SUAVE.Vehicle()
    #aircraft.wing       = wing
    #aircraft.fuselage   = fuselage
    #aircraft.vertical   = vertical
    #aircraft.mass_properties.center_of_gravity[0] = 112.2 * Units.feet
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Embraer_E190'

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------

    # mass properties
    vehicle.mass_properties.max_takeoff               = 51800.0   # kg
    vehicle.mass_properties.operating_empty           = 29100.0   # kg
    vehicle.mass_properties.takeoff                   = 51800.0   # kg
    vehicle.mass_properties.max_zero_fuel             = 45600.0   # kg
    vehicle.mass_properties.cargo                     = 0.0 * Units.kilogram
    vehicle.mass_properties.max_payload               = 11786. * Units.kilogram
    vehicle.mass_properties.max_fuel                  = 12970.

    vehicle.mass_properties.center_of_gravity         = [112.2 * Units.feet, 0, 0]
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct

    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 5500.0 * Units.feet**2
    vehicle.passengers             = 0
    vehicle.systems.control        = "fully powered"
    vehicle.systems.accessories    = "medium range"

    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'

    wing.aspect_ratio            = (196.0**2.)  * (Units.feet**2.) / vehicle.reference_area
    wing.sweep                   = 42.0 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 14.7/54.5
    wing.span_efficiency         = 1.0

    wing.spans.projected         = 196.0  * Units.feet

    wing.chords.root             = 54.5 * Units.feet
    wing.chords.tip              = 14.7 * Units.feet

    wing.areas.reference         = 5500.0 * Units.feet**2
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted

    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [65.*Units.feet,0.,-3.6*Units.feet]
    wing.aerodynamic_center      = [trapezoid_ac_x(wing),0,0]

    wing.vertical                = False
    wing.symmetric               = True

    wing.dynamic_pressure_ratio = 1.0

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'

    wing.aspect_ratio            = 5.5
    wing.sweep                   = 34.5 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.11
    wing.span_efficiency         = 0.9

    wing.spans.projected         = 11.958

    wing.chords.root             = 3.9175
    wing.chords.tip              = 0.4309
    wing.chords.mean_aerodynamic = 2.6401

    wing.areas.reference         = 26.0
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted

    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 2.0 * Units.degrees

    wing.origin                  = [50,0,0]
    wing.aerodynamic_center      = [2,0,0]

    wing.vertical                = False
    wing.symmetric               = True

    wing.dynamic_pressure_ratio = 0.9

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'
    wing.effective_aspect_ratio  = 2.25
    wing.sweep                   = 50. * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.taper                   = 13.4 / 38.7
    wing.span_efficiency         = 0.9

    wing.spans.projected         = 32.4 * Units.feet
    wing.spans.exposed           = 32.4 * Units.feet

    wing.chords.root             = 38.7 * Units.feet
    wing.chords.tip              = 13.4 * Units.feet
    wing.chords.mean_aerodynamic = 8.
    wing.chords.fuselage_intersect = wing.chords.root

    wing.areas.reference         = 16.0    #
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted
    
    wing.exposed_root_chord_offset = 13.5 * Units.feet

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [181.*Units.feet,0,0]
    wing.aerodynamic_center      = [trapezoid_ac_x(wing),0,.4*wing.spans.exposed]
    print wing.aerodynamic_center

    wing.vertical                = True
    wing.symmetric               = False

    wing.dynamic_pressure_ratio = 1.0  

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'

    fuselage.number_coach_seats    = vehicle.passengers
    fuselage.seats_abreast         = 4
    fuselage.seat_pitch            = 0.7455

    fuselage.fineness.nose         = 2.0
    fuselage.fineness.tail         = 3.0

    fuselage.lengths.nose          = 6.0
    fuselage.lengths.tail          = 9.0
    fuselage.lengths.cabin         = 21.24
    fuselage.lengths.total         = 229.7 * Units.feet
    fuselage.lengths.fore_space    = 0.
    fuselage.lengths.aft_space     = 0.

    fuselage.width                 = 20.9 * Units.feet

    fuselage.heights.maximum       = 26.9 * Units.feet
    fuselage.heights.at_quarter_length          = 26   * Units.feet
    fuselage.heights.at_three_quarters_length   = 19.7 * Units.feet
    fuselage.heights.at_vertical_root_quarter_chord = 15.8 * Units.feet

    fuselage.areas.side_projected  = 4696.16 * Units.feet**2

    fuselage.effective_diameter    = 23.7 * Units.feet

    fuselage.differential_pressure = 10**5 * Units.pascal    # Maximum differential pressure

    # add to vehicle
    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    #  Turbofan
    # ------------------------------------------------------------------

    turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    turbofan.tag = 'Turbo Fan'

    turbofan.propellant = SUAVE.Attributes.Propellants.Jet_A()

    turbofan.analysis_type                 = '1D'     #
    turbofan.diffuser_pressure_ratio       = 0.99     #
    turbofan.fan_pressure_ratio            = 1.7      #
    turbofan.fan_nozzle_pressure_ratio     = 0.98     #
    turbofan.lpc_pressure_ratio            = 1.9      #
    turbofan.hpc_pressure_ratio            = 10.0     #
    turbofan.burner_pressure_ratio         = 0.95     #
    turbofan.turbine_nozzle_pressure_ratio = 0.99     #
    turbofan.Tt4                           = 1500.0   #
    turbofan.bypass_ratio                  = 5.4      #
    turbofan.thrust.design                 = 20300.0  #
    turbofan.number_of_engines                 = 2.0      #
    turbofan.engine_length                     = 3.0

    # turbofan sizing conditions
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()

    sizing_segment.M   = 0.78          #
    sizing_segment.alt = 10.668         #
    sizing_segment.T   = 223.0        #
    sizing_segment.p   = 0.265*10**5  #

    # size the turbofan
    turbofan.engine_sizing_1d(sizing_segment)

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

    takeoff_config.wings['Main Wing'].flaps_angle = 20. * Units.deg
    takeoff_config.wings['Main Wing'].slats_angle = 25. * Units.deg

    takeoff_config.V2_VS_ratio = 1.21
    takeoff_config.maximum_lift_coefficient = 2.
    #takeoff_config.max_lift_coefficient_factor = 1.0

    # --- Landing Configuration ---
    landing_config = vehicle.new_configuration("landing")

    landing_config.wings['Main Wing'].flaps_angle = 30. * Units.deg
    landing_config.wings['Main Wing'].slats_angle = 25. * Units.deg

    landing_config.Vref_VS_ratio = 1.23
    landing_config.maximum_lift_coefficient = 2.
    #landing_config.max_lift_coefficient_factor = 1.0

    landing_config.mass_properties.landing = 0.85 * vehicle.mass_properties.takeoff

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------    
    segment            = SUAVE.Attributes.Missions.Segments.Aerodynamic_Segment()
    M                  = 0.198
    segment.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    altitude           = 0.0 * Units.feet
    p, T, rho, a, mew = segment.atmosphere.compute_values(altitude)
    segment.conditions.freestream.velocity = M * a
    segment.conditions.freestream.viscosity = mew
    segment.conditions.freestream.density = rho
    segment.conditions.freestream.mach_number = M
    
    #Method Test
    print '<<Test run of the taw_cnbeta() method>>'
    print 'Boeing 747 at M = {0} and h = {1} meters'.format(M, altitude)
    
    cn_b = taw_cnbeta(vehicle,segment.conditions,vehicle.configs.cruise)[0]
    
    expected = 0.184
    print 'Cn_beta =        {0:.4f}'.format(cn_b)
    print 'Expected value = {}'.format(expected)
    print 'Percent Error =  {0:.2f}%'.format(100.0*(cn_b-expected)/expected)
    print ' '    