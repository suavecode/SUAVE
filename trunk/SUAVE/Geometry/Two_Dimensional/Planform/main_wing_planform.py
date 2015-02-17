# main_wing_planform.py
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

from SUAVE.Geometry.Two_Dimensional.Planform.Cranked_Planform import Cranked_Planform
# import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def main_wing_planform(wing):
    """ err = SUAVE.Geometry.main_wing_planform(Wing)

        main wing planform

        Assumptions:
            cranked wing with leading and trailing edge extensions

        Inputs:
            wing.sref
            wing.ar
            wing.taper
            wing.sweep
            wing.span
            wing.lex
            wing.tex
            wing.span_ratios.fuselage
            wing.span_ratios.break_point

        Outputs:
            wing.chord_root
            wing.chord_tip
            wing.chord_mid
            wing.chord_mac
            wing.area_wetted
            wing.span

    """
    
    # unpack
    sref = wing.areas.reference
    taper = wing.taper
    sweep = wing.sweep
    ar = wing.aspect_ratio
    thickness_to_chord = wing.thickness_to_chord
    span_ratio_fuselage = wing.span_ratios.fuselage
    span_ratio_break = wing.span_ratios.break_point
    lex_ratio = wing.lex
    tex_ratio = wing.tex

    # compute wing planform geometry
    wpc = Cranked_Planform(sref, ar, sweep, taper, span_ratio_fuselage,
                          span_ratio_break, lex_ratio, tex_ratio)

    # set the wing origin
    wpc.set_origin(wing.origin)

    # compute
    wpc.update()

    # compute flapped area if wing is flapped
    if wing.flaps.type is not None:
        wpc.add_flap(wing.flaps.span_start, wing.flaps.span_end)

    # update
    wing.chords.root = wpc.chord_root
    wing.chords.tip = wpc.chord_tip
    wing.chords.break_point = wpc.chord_break
    wing.chords.mean_aerodynamic = wpc.mean_aerodynamic_chord
    wing.chords.mean_aerodynamic_exposed = wpc.mean_aerodynamic_chord_exposed
    wing.chords.mean_geometric = wpc.mean_geometric_chord

    # plot the wing to check things
    # x_wing, y_wing = wpc.get_wing_coordinates()
    # plt.plot(x_wing, y_wing, "k-")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('equal')
    # plt.show()

    wing.aerodynamic_center = [wpc.x_aerodynamic_center, 0, 0]
    wing.areas.wetted = wpc.calc_area_wetted(thickness_to_chord)
    wing.areas.gross = wpc.area_gross
    wing.areas.exposed = wpc.area_exposed
    wing.spans.projected = wpc.span
    wing.areas.flapped = wpc.area_flapped

    # for backward compatibility
    wing.areas.affected = wing.areas.flapped
    wing.chords.mid = wing.chords.break_point

    return wing
