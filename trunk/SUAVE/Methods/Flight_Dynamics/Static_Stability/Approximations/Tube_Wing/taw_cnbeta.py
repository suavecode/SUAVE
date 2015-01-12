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
        destabilizing moments due to propellers. This can lead to higher than
        expected values of CnBeta, particularly for smaller prop-driven aircraft
        
        Inputs:
            configuration - a data dictionary with the fields:
                Mass_Props - a data dictionary with the field:
                    pos_cg - A vector in 3-space indicating CG position
                wing - a data dictionary with the fields:
                    area - wing reference area [meters**2]
                    span - span of the wing [meters]
                    sweep_le - sweep of the wing leading edge [radians]
                    z_position - distance of wing root quarter chord point
                    below fuselage centerline [meters]
                    taper - wing taper ratio [dimensionless]
                    aspect_ratio - wing aspect ratio [dimensionless]
                fuselage - a data dictionary with the fields:
                    side_area - fuselage body side area [meters**2]
                    length - length of the fuselage [meters]
                    h_max - maximum height of the fuselage [meters]
                    w_max - maximum width of the fuselage [meters]
                    height_at_quarter_length - fuselage height at 1/4 of the 
                    fuselage length [meters]
                    height_at_three_quarters_length - fuselage height at 3/4 of
                    the fuselage length [meters]
                    height_at_vroot_quarter_chord - fuselage height at the 
                    aerodynamic center of the vertical tail [meters]
                vertical - a data dictionary with the fields below:
                NOTE: Reference vertical tail extends to the fuselage centerline
                    area - area of the reference vertical tail [meters**2]
                    x_root_LE - x-position of the vertical reference root chord 
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
    S      = geometry.wings['main_wing'].areas.reference
    b      = geometry.wings['main_wing'].spans.projected
    sweep  = geometry.wings['main_wing'].sweep
    AR     = geometry.wings['main_wing'].aspect_ratio
    z_w    = geometry.wings['main_wing'].origin[2]
    S_bs   = geometry.fuselages['fuselage'].areas.side_projected
    l_f    = geometry.fuselages['fuselage'].lengths.total
    h_max  = geometry.fuselages['fuselage'].heights.maximum
    w_max  = geometry.fuselages['fuselage'].width
    h1     = geometry.fuselages['fuselage'].heights.at_quarter_length
    h2     = geometry.fuselages['fuselage'].heights.at_three_quarters_length
    d_i    = geometry.fuselages['fuselage'].heights.at_wing_root_quarter_chord
    other  = configuration.other
    S_v    = geometry.wings['vertical_stabilizer'].areas.reference
    x_v    = geometry.wings['vertical_stabilizer'].origin[0]
    b_v    = geometry.wings['vertical_stabilizer'].spans.projected
    ac_vLE = geometry.wings['vertical_stabilizer'].aerodynamic_center[0]
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
    CLa_v  = geometry.wings['vertical_stabilizer'].CL_alpha
    #k_v correlated from Roskam Fig. 10.12. NOT SMOOTH.
    bf     = b_v/d_i
    if bf < 2.0:
        k_v = 0.76
    elif bf < 3.5:
        k_v = 0.76 + 0.24*(bf-2.0)/1.5
    else:
        k_v = 1.0
    quarter_chord_sweep = convert_sweep(geometry.wings['main_wing'])
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
    #Parameters Required
    #Using values for a Boeing 747-200
    wing                = SUAVE.Components.Wings.Wing()
    wing.area           = 5500.0 * Units.feet**2
    wing.span           = 196.0  * Units.feet
    wing.sweep_le          = 42.0   * Units.deg
    wing.z_position     = 3.6    * Units.feet
    wing.taper          = 14.7/54.5
    wing.aspect_ratio   = wing.span**2/wing.area
    
    fuselage            = SUAVE.Components.Fuselages.Fuselage()
    fuselage.side_area  = 4696.16 * Units.feet**2
    fuselage.length     = 229.7   * Units.feet
    fuselage.h_max      = 26.9    * Units.feet
    fuselage.w_max      = 20.9    * Units.feet
    fuselage.height_at_vroot_quarter_chord   = 15.8 * Units.feet
    fuselage.height_at_quarter_length        = 26   * Units.feet
    fuselage.height_at_three_quarters_length = 19.7 * Units.feet
    
    vertical              = SUAVE.Components.Wings.Wing()
    vertical.span         = 32.4   * Units.feet
    vertical.root_chord   = 38.7   * Units.feet
    vertical.tip_chord    = 13.4   * Units.feet
    vertical.sweep_le     = 50.0   * Units.deg
    vertical.x_root_LE1   = 181.0  * Units.feet
    dz_centerline         = 13.5   * Units.feet
    ref_vertical          = extend_to_ref_area(vertical,dz_centerline)
    vertical.span         = ref_vertical.ref_span
    vertical.area         = ref_vertical.ref_area
    vertical.aspect_ratio = ref_vertical.ref_aspect_ratio
    vertical.x_root_LE    = vertical.x_root_LE1 + ref_vertical.root_LE_change
    vertical.taper        = vertical.tip_chord/ref_vertical.ref_root_chord
    vertical.effective_aspect_ratio = 2.25
    vertical_symm         = copy.deepcopy(vertical)
    vertical_symm.span    = 2.0*vertical.span
    vertical_symm.area    = 2.0*vertical.area
    vertical.x_ac_LE      = trapezoid_ac_x(vertical_symm)
    
    aircraft            = SUAVE.Vehicle()
    aircraft.wing       = wing
    aircraft.fuselage   = fuselage
    aircraft.vertical   = vertical
    aircraft.mass_properties.center_of_gravity[0] = 112.2 * Units.feet
    
    segment            = SUAVE.Attributes.Missions.Segments.Segment()
    segment.M          = 0.198
    segment.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    altitude           = 0.0 * Units.feet
    segment.compute_atmosphere(altitude / Units.km)
    segment.v_inf      = segment.M * segment.a
    
    #Method Test
    print '<<Test run of the taw_cnbeta() method>>'
    print 'Boeing 747 at M = {0} and h = {1} meters'.format(segment.M, altitude)
    
    cn_b = taw_cnbeta(aircraft,segment)
    
    expected = 0.184
    print 'Cn_beta =        {0:.4f}'.format(cn_b)
    print 'Expected value = {}'.format(expected)
    print 'Percent Error =  {0:.2f}%'.format(100.0*(cn_b-expected)/expected)
    print ' '    