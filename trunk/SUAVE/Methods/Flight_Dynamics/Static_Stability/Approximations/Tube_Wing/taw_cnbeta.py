## @ingroup Methods-Flight_Dynamics-Static_Stability-Approximations-Tube_Wing
# taw_cnbeta.py
#
# Created:  Mar 2014, T. Momose
# Modified: Jul 2014, A. Wendorff
#           Jan 2016, E. Botero
#           May 2021, E. Botero

# TO DO:
#    - Add capability for multiple vertical tails
#    - Smooth out k_v factor (line 143)
#    - Add effect of propellers

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import copy
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------
## @ingroup Methods-Flight_Dynamics-Static_Stability-Approximations-Tube_Wing
def taw_cnbeta(geometry,conditions,configuration):
    """ This method computes the static directional stability derivative for a
    standard Tube-and-Wing aircraft configuration.        

    CAUTION: The correlations used in this method do not account for the
    destabilizing moments due to propellers. This can lead to higher-than-
    expected values of CnBeta, particularly for smaller prop-driven aircraft

    Assumptions:
        -Assumes a tube-and-wing configuration with a single centered 
        vertical tail
        -Uses vertical tail effective aspect ratio, currently calculated by
        hand, using methods from USAF Stability and Control DATCOM
        -The validity of correlations for KN is questionable for sqrt(h1/h2)
        greater than about 4 or h_max/w_max outside [0.3,2].
        -This method assumes a small angle of attack, so the vertical tail AC
        z-position does not affect the sideslip derivative.
    
    Source:
        Correlations are taken from Roskam's Airplane Design, Part VI.
    
    Inputs:
        geometry - aircraft geometrical features: a data dictionary with the fields:
            wings                                                                         ['Main Wing'] - the aircraft's main wing
                areas.reference - wing reference area                                     [meters**2]
                spans.projected - span of the wing                                        [meters]
                sweep - sweep of the wing leading edge                                    [radians]
                aspect_ratio - wing aspect ratio                                          [dimensionless]
                origin - the position of the wing root in the aircraft body frame         [meters]
            wings                                                                         ['Vertical Stabilizer']
                spans.projected - projected span (height for a vertical tail) of
                 the exposed surface                                                      [meters]
                areas.reference - area of the reference vertical tail                     [meters**2]
                sweep - leading edge sweep of the aerodynamic surface                     [radians]
                chords.root - chord length at the junction between the tail and 
                 the fuselage                                                             [meters]
                chords.tip - chord length at the tip of the aerodynamic surface           [meters]
                symmetric - Is the wing symmetric across the fuselage centerline?
                origin - the position of the vertical tail root in the aircraft body frame[meters]
                exposed_root_chord_offset - the displacement from the fuselage
                 centerline to the exposed area's physical root chordline                 [meters]

            fuselages.Fuselage - a data dictionary with the fields:
                areas.side_projected - fuselage body side area                            [meters**2]
                lengths.total - length of the fuselage                                    [meters]
                heights.maximum - maximum height of the fuselage                          [meters]
                width - maximum width of the fuselage                                     [meters]
                heights.at_quarter_length - fuselage height at 1/4 of the fuselage length [meters]
                heights.at_three_quarters_length - fuselage height at 3/4 of fuselage 
                 length                                                                   [meters]
                heights.at_vertical_root_quarter_chord - fuselage height at the quarter 
                 chord of the vertical tail root                                          [meters]
            vertical - a data dictionary with the fields below:
            NOTE: This vertical tail geometry will be used to define a reference
             vertical tail that extends to the fuselage centerline.

                x_ac_LE - the x-coordinate of the vertical tail aerodynamic 
                center measured relative to the tail root leading edge (root
                of reference tail area - at fuselage centerline)
                leading edge, relative to the nose                                        [meters]
                sweep_le - leading edge sweep of the vertical tail                        [radians]
                span - height of the vertical tail                                        [meters]
                taper - vertical tail taper ratio                                         [dimensionless]
                aspect_ratio - vertical tail AR: bv/(Sv)^2                                [dimensionless]
                effective_aspect_ratio - effective aspect ratio considering
                the effects of fuselage and horizontal tail                               [dimensionless]
                symmetric - indicates whether the vertical panel is symmetric
                about the fuselage centerline                                             [Boolean]
            other_bodies - an list of data dictionaries containing bodies 
            such as nacelles if these are large enough to strongly influence
            stability. Each body data dictionary contains the same fields as
            the fuselage data dictionary (described above), except no value 
            is needed for 'height_at_vroot_quarter_chord'. CAN BE EMPTY LIST
                x_front - This is the only new field needed: the x-coordinate 
                of the nose of the body relative to the fuselage nose

        conditions - a data dictionary with the fields:
            v_inf - true airspeed                                                         [meters/second]
            M - flight Mach number
            rho - air density                                                             [kg/meters**3]
            mu  - air dynamic dynamic_viscosity                                           [kg/meter/second]

        configuration - a data dictionary with the fields:
            mass_properties - a data dictionary with the field:
                center_of_gravity - A vector in 3-space indicating CG position            [meters]
            other - a dictionary of aerodynamic bodies, other than the fuselage,
            whose effect on directional stability is to be included in the analysis

    Outputs:
        CnBeta - a single float value: The static directional stability 
        derivative

    Properties Used:
        N/A  
    """         

    # Unpack inputs
    S      = geometry.reference_area
    b      = geometry.wings['main_wing'].spans.projected
    AR     = geometry.wings['main_wing'].aspect_ratio
    z_w    = geometry.wings['main_wing'].origin[0][2]
    vert   = extend_to_ref_area(geometry.wings['vertical_stabilizer'])
    S_v    = vert.extended.areas.reference
    x_v    = vert.extended.origin[0][0]
    b_v    = vert.extended.spans.projected
    ac_vLE = vert.aerodynamic_center[0]
    x_cg   = configuration.mass_properties.center_of_gravity[0][0]
    v_inf  = conditions.freestream.velocity
    mu     = conditions.freestream.dynamic_viscosity
    rho    = conditions.freestream.density
    M      = conditions.freestream.mach_number
    
    #Compute wing contribution to Cn_beta
    CnBeta_w = 0.0    #The wing contribution is assumed to be zero except at very
                      #high angles of attack. 
    fuse_cnb = 0.0
                      
    for fuse in geometry.fuselages:
    
        S_bs   = fuse.areas.side_projected
        l_f    = fuse.lengths.total
        h_max  = fuse.heights.maximum
        w_max  = fuse.width
        h1     = fuse.heights.at_quarter_length
        h2     = fuse.heights.at_three_quarters_length
        d_i    = fuse.heights.at_wing_root_quarter_chord    
    
        #Compute fuselage contribution to Cn_beta
        Re_fuse  = rho*v_inf*l_f/mu
        x1       = x_cg/l_f
        x2       = l_f*l_f/S_bs
        x3       = np.sqrt(h1/h2)
        x4       = h_max/w_max
        kN_1     = 3.2413*x1 - 0.663345 + 6.1086*np.exp(-0.22*x2)
        kN_2     = (-0.2023 + 1.3422*x3 - 0.1454*x3*x3)*kN_1
        kN_3     = (0.7870 + 0.1038*x4 + 0.1834*x4*x4 - 2.811*np.exp(-4.0*x4))
        K_N      = (-0.47899 + kN_3*kN_2)*0.001
        K_Rel    = 1.0+0.8*np.log(Re_fuse/1.0E6)/np.log(50.) 
            #K_Rel: Correction for fuselage Reynolds number. Roskam VI, page 400.
        fuse_cnb = fuse_cnb -57.3*K_N*K_Rel*S_bs*l_f/S/b
    
    
    #Compute vertical tail contribution
    l_v    = x_v + ac_vLE - x_cg

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
        
    if geometry.wings.main_wing.sweeps.quarter_chord is not None:
        quarter_chord_sweep  = geometry.wings.main_wing.sweeps.quarter_chord
    else:
        quarter_chord_sweep = convert_sweep(geometry.wings['main_wing'])
    
    k_sweep  = (1.0+np.cos(quarter_chord_sweep))
    dsdb_e   = 0.724 + 3.06*((S_v/S)/k_sweep) + 0.4*z_w/h_max + 0.009*AR
    Cy_bv    = -k_v*CLa_v*dsdb_e*(S_v/S)  #ASSUMING SINGLE VERTICAL TAIL
    
    CnBeta_v = -Cy_bv*l_v/b
    
    CnBeta   = CnBeta_w + CnBeta_v + fuse_cnb
    
    return CnBeta
