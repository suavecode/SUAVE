# v_tail_sizing.py
#
# Created:  Sep 2016, D. Bianchi
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#   V-tail sizing
# ----------------------------------------------------------------------

def v_tail_sizing(wing, v_tail, c_ht, c_vt, i_opt_dihedral = 0):
    """ Sizes the v-tail based on horizontal and vertical tail volume coefficients and the
        equivalent horizontal and vertical tails projections on each plane

        Inputs:
            wing: main wing
                areas.reference                 main wing reference area
                spans.projected                 main wing span
                chords.mean_aerodynamic         main wing mean aerodynamic chord
                sweeps.quarter_chord            main wing quarter chord sweep angle
                taper                           main wing taper ratio
                origin                          [x,y,z] of main wing appex
            v_tail: structured data containing the geometrical description of the v-tail with attributes:
                areas.reference (optional)      reference area at the inclined plane - first guess
                aspect_ratio                    aspect-ratio (b**2/S, with b and S both measured in the inclined plane)
                sweeps.quarter_chord            quarter chord sweep angle
                taper                           taper-ratio (tip_chord/root_chord)
                thickness_to_chord              average thickness to chord ratio
                dihedral                        dihedral angle
                origin                          [x,y,z] of vtail appex
            c_ht: equivalent horizontal tail volume coefficient (design)
            c_vt: equivalent vertical tail volume coefficient (design)
            i_opt_dihedral: flag for using optimal dihedral (=1: use; =0: uses user's defined)
                            optimal dihedral:: the one that minimizes tail wetted area by
                            simultaneously attaining to both design tail volume coefficients

        Outputs:
            v_tail:     sized v-tail
            c_ht_out:   resulted equivalent horizontal tail volume coefficient
            c_vt_out:   resulted equivalent vertical tail volume coefficient

        Assumptions:
            The projections of the v-tail in the horizontal and vertical planes are representative
            of the equivalent tails.
            Reference: NACA Report No.823
    """
    # configuring convergence criteria
    eps = 1e-18
    i = 0 # iteration counter

    # check if initial tail area has been provided
    if not(v_tail.areas.reference):
        v_tail.areas.reference = 0.25 * wing.areas.reference # first guess
    # calling initial sizing
    v_tail, c_ht_out, c_vt_out = sizing(wing,v_tail,c_ht,c_vt,i_opt_dihedral)

    # calculating residual
    res = np.min([(c_ht_out - c_ht)**2., (c_vt_out - c_vt)**2.])

    # convergence loop
    while res > eps:
        i += 1
        v_tail, c_ht_out, c_vt_out = sizing(wing,v_tail,c_ht,c_vt,i_opt_dihedral)
        res = np.min([(c_ht_out - c_ht)**2., (c_vt_out - c_vt)**2.])
##        print i, res, v_tail.areas.reference, v_tail.spans.projected, v_tail.chords.mean_aerodynamic, v_tail.chords.root, v_tail.chords.tip

    return v_tail, c_ht_out, c_vt_out

def sizing(wing,v_tail,c_ht,c_vt,i_opt_dihedral=0):
    # unpack inputs
    S_w = wing.areas.reference
    b_w = wing.spans.projected
    c_w = wing.chords.mean_aerodynamic
    sweep_w = wing.sweeps.quarter_chord
    taper_w = wing.taper
    x_w = wing.origin[0]
    S_t = v_tail.areas.reference
    ar_t = v_tail.aspect_ratio
    sweep_t = v_tail.sweeps.quarter_chord
    taper_t = v_tail.taper
    tc_t = v_tail.thickness_to_chord
    dihedral_t = v_tail.dihedral
    x_t = v_tail.origin[0]

    # calculate the main wing aspect-ratio
    ar_w = b_w**2. / S_w

    # calculate main wing leading edge sweep
    le_sweep_w = np.arctan( np.tan(sweep_w) - (4./ar_w)*(0.-0.25)*(1.-taper_w)/(1.+taper_w) )

    # estimating main wing aerodynamic center coordinates (reference on the wing)
    y_coord_w = b_w / 6. * (( 1. + 2. * taper_w ) / (1. + taper_w))
    x_coord_w = c_w * 0.25 + y_coord_w * np.tan(le_sweep_w)

    # translating coordenates to the airframe reference
    x_coord_w += x_w

    vtail_planform(v_tail)
    x_coord_t = v_tail.aerodynamic_center[0]

    # translating coordenates to the airframe reference
    x_coord_t += x_t

    # calculating tail arm
    l_t = np.abs(x_coord_t - x_coord_w)

    # estimating equivalent horizontal and vertical tail required areas to meet
    # the desired tail volume coefficients (i.e. c_ht, c_vt)
    S_ht = S_w * c_w * c_ht / l_t
    S_vt = S_w * b_w * c_vt / l_t

    # sizing the v-tail area to meet the most demanding criteria
    S_t = np.max([S_ht / (np.cos(dihedral_t))**2., S_vt / (np.sin(dihedral_t))**2.])

    if i_opt_dihedral:
        # estimating the suggested dihedral so that both tail volume coefficients are simultaneously met
        # by the same v-tail area
        l_ht = l_t
        l_vt = l_t
        optimal_dihedral = np.arctan((c_vt * l_ht * b_w / (c_ht * l_vt * c_w))**0.5)
        v_tail.dihedral = optimal_dihedral
        sizing(wing, v_tail, c_ht, c_vt)

        # sizing the v-tail area to meet the most demanding criteria
        S_t = np.max([S_ht / (np.cos(optimal_dihedral))**2., S_vt / (np.sin(optimal_dihedral))**2.])

    # updating tail span and reference area
    b_t = (ar_t * S_t)**0.5
    b_h = b_t * np.cos(dihedral_t)
    b_v = b_t * np.sin(dihedral_t)
    v_tail.spans.projected = b_h
    v_tail.areas.reference = S_t


    # re-calculating tail geometry (after sizing)
    vtail_planform(v_tail)
    x_coord_t = v_tail.aerodynamic_center[0]
    S_t = v_tail.areas.reference

    # updating coordinates
    x_coord_t += x_t

    # updating tail arm
    l_t = np.abs(x_coord_t - x_coord_w)

    # updating tail volume coefficients (after sizing)
    S_ht = S_t * (np.cos(dihedral_t))**2.
    S_vt = S_t * (np.sin(dihedral_t))**2.
    c_ht = S_ht * l_t / (S_w * c_w)
    c_vt = S_vt * l_t / (S_w * b_w)

    return v_tail, c_ht, c_vt

def vtail_planform(v_tail):

    # unpack inputs
    S_t = v_tail.areas.reference
    ar_t = v_tail.aspect_ratio
    sweep_t = v_tail.sweeps.quarter_chord
    taper_t = v_tail.taper
    tc_t = v_tail.thickness_to_chord
    dihedral_t = v_tail.dihedral
    x_t = v_tail.origin[0]

    # calculating the vee-tail spans
    b_t = (ar_t * S_t)**0.5         # actual span in the inclined plan
    b_h = b_t * np.cos(dihedral_t)  # span projected in the horizontal plane
    b_v = b_t * np.sin(dihedral_t)  # span projected in the vertical plane

    # calculate v-tail leading edge sweep
    le_sweep_t = np.arctan( np.tan(sweep_t) - (4./ar_t)*(0.-0.25)*(1.-taper_t)/(1.+taper_t) )

    # calculating the tail chords
    c_r_t = 2 * S_t / b_t / ( 1 + taper_t)
    c_t_t  = taper_t * c_r_t
    c_t = 2./3. * ( c_r_t + c_t_t - c_r_t * c_t_t / (c_r_t + c_t_t) )

    # estimating v-tail aerodynamic center coordinates (reference on the tail)
    y_coord_t = b_h / 6. * (( 1. + 2. * taper_t ) / (1. + taper_t))
    x_coord_t = c_t * 0.25 + y_coord_t * np.tan(le_sweep_t)
    z_coord_t = y_coord_t * np.tan(dihedral_t)

    y_coord_t = 0 # t-tail is symmetric

    # packing outputs
    v_tail.sweeps.leading_edge              = le_sweep_t
    v_tail.chords.root                      = c_r_t
    v_tail.chords.tip                       = c_t_t
    v_tail.chords.mean_aerodynamic          = c_t
    v_tail.spans.projected                  = b_h
    v_tail.aerodynamic_center               = [x_coord_t,y_coord_t,z_coord_t]

    return v_tail

# ----------------------------------------------------------------------
#                       test unit script
# ----------------------------------------------------------------------
if __name__ == '__main__':

    import SUAVE
    from SUAVE.Core import Data,Units
    from SUAVE.Components.Wings import Wing

    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'

    wing.areas.reference         = 16.0
    wing.aspect_ratio            = 6.43
    wing.sweeps.quarter_chord    = 9.34 * Units.deg
    wing.thickness_to_chord      = 0.108
    wing.taper                   = 0.50
    wing.origin                  = [5.05,0,0]
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)

    # ------------------------------------------------------------------
    #  V-tail
    # ------------------------------------------------------------------
    v_tail = SUAVE.Components.Wings.Wing()
    v_tail.tag = 'v_tail'

    v_tail.areas.reference         = 7.21 #6.30 # at inclined plane
    v_tail.aspect_ratio            = 6.624**2./7.21
    v_tail.sweeps.quarter_chord    = 42.22 * Units.deg
    v_tail.taper                   = 1.00/1.90
    v_tail.thickness_to_chord      = 0.11
    v_tail.dihedral                = 47.0 * Units.deg
    v_tail.origin                  = [8.912,0,0]

    # Desired tail volume coefficients
    c_ht = 0.60
    c_vt = 0.06

    # First size considering the given baseline dihedral
    v_tail, c_ht, c_vt = v_tail_sizing(wing,v_tail,c_ht,c_vt)
    print v_tail.areas.reference
    print 'Final c_ht', c_ht
    print 'Final c_vt', c_vt
    print '.................'

    # Desired tail volume coefficients
    c_ht = 0.60
    c_vt = 0.06

    # Then size considering the optimal dihedral
    import time
    tinit = time.time()
    v_tail, c_ht, c_vt = v_tail_sizing(wing,v_tail,c_ht,c_vt,1)
    tfinal = time.time()
    total_time = tfinal - tinit
    print v_tail.areas.reference
    print 'Final c_ht', c_ht
    print 'Final c_vt', c_vt
    print 'Optimal dihedral', v_tail.dihedral/Units.deg

##    x_w = wing.origin[0] + wing.aerodynamic_center[0]
##    sweep_t = v_tail.sweeps.leading_edge
##    taper_t = v_tail.taper
##    ar_t = v_tail.aspect_ratio
##    x_t = v_tail.origin[0]
##    dihedral_t = v_tail.dihedral
##    S_t = v_tail.areas.reference
##
##    l_t = - x_w + x_t + np.tan(sweep_t) * ((S_t * ar_t)**0.5 * np.cos(dihedral_t) / 6.) * (1 + 2*taper_t) / (1 + taper_t) \
##                      + (1./3.) * (S_t / ar_t)**0.5 / (1 + taper_t) * (taper_t**2. + taper_t + 1) / (taper_t + 1)
##
##
##    l_t_x = v_tail.aerodynamic_center[0] + v_tail.origin[0] - x_w
##
##    print l_t, l_t_x