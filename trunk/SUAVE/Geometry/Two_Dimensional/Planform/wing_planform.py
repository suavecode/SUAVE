# Geoemtry.py
#

""" SUAVE Methods for Geometry Generation
"""

# TODO:
# object placement, wing location
# tail: placed at end of fuselage, or pull from volume
# engines: number of engines, position by 757

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Geometry.Two_Dimensional.Planform.TrapezoidalPlanform import TrapezoidalPlanform


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def wing_planform(wing):
    """ err = SUAVE.Geometry.wing_planform(Wing)
    
        basic wing planform calculation
        
        Assumptions:
            trapezoidal wing
            no leading/trailing edge extensions
            
        Inputs:
            Wing.sref
            Wing.ar
            Wing.taper
            Wing.sweep
            
        Outputs:
            Wing.chord_root
            Wing.chord_tip
            Wing.chord_mac
            Wing.area_wetted
            Wing.span
        
    """
    
    # unpack
    sref = wing.areas.reference
    taper = wing.taper
    sweep = wing.sweep
    ar = wing.aspect_ratio
    thickness_to_chord = wing.thickness_to_chord
    span_ratio_fuselage = wing.span_ratios.fuselage

    # compute wing planform geometry
    wpt = TrapezoidalPlanform(sref, ar, sweep, taper,
                              span_ratio_fuselage)

    # set the wing origin
    wpt.set_origin(wing.origin)

    # compute
    wpt.update()

    # update
    wing.chords.root = wpt.chord_root
    wing.chords.tip = wpt.chord_tip
    wing.chords.mean_aerodynamic = wpt.mean_aerodynamic_chord
    wing.chords.mean_aerodynamic_exposed = wpt.mean_aerodynamic_chord_exposed
    wing.chords.mean_geometric = wpt.mean_geometric_chord

    wing.aerodynamic_center = [wpt.x_aerodynamic_center, 0, 0]

    wing.areas.wetted = wpt.calc_area_wetted(thickness_to_chord)
    wing.areas.gross = wpt.area_gross
    wing.areas.exposed = wpt.area_exposed

    wing.spans.projected = wpt.span

    return wing
