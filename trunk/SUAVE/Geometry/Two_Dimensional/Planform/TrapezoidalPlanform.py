import numpy as np

from SemiPlanform import SemiPlanform
from Planform import Planform


"""
Geometry calculations for a trapzoidal wing
    - Computes the gross, exposed and wetted areas
    - Computes the mean aerodynamic chord of the cranked and exposed planform
    - Estimates the aerodynamic center of the cranked planform
    - Includes option to compute the flapped wing area
"""


class TrapezoidalPlanform(Planform):
    def __init__(self, sref, ar, sweep_qc, taper,
                 thickness_to_chord=0.12, span_ratio_fuselage=0.1):
        # call superclass constructor with 0 lex and tex
        super(TrapezoidalPlanform, self).__init__(sref, ar, sweep_qc, taper,
                                                  thickness_to_chord, span_ratio_fuselage, 0, 0)

    def update(self):
        """
        Update the wing geometry
        :return:
        """

        # span of control points: [root, break, tip]
        self.y_node = np.array([0, 1]) * self._semi_span

        # trapezoidal wing definition chords
        c_node_trap = np.array([self.chord_root_trap, self.chord_tip])

        # root chord
        self.chord_root = self.chord_root_trap * (1 + self.lex_ratio + self.tex_ratio)

        # extended wing definition chords
        self.c_node = np.array([self.chord_root, self.chord_tip])

        # compute the extended wing semi-planform geometry
        semi_planform = SemiPlanform(self.c_node, self.y_node)
        semi_planform.update()

        # sort the chords with y; guard against case where fuselage is wider than break point
        semi_planform.sort_chord_by_y()

        # build a wing chord interpolant that can be reused
        self.chord_from_y = semi_planform.get_chord_interpolator()

        # get the fuselage-wing intersection chord
        chord_fuse_intersection = self.chord_from_y(self.span_ratio_fuselage * self.__semi_span)

        # compute the exposed semi-planform properties
        c_node_exposed = np.array([chord_fuse_intersection, self.chord_tip])
        y_node_exposed = np.array([self.span_ratio_fuselage, 1]) * self.__semi_span
        exposed_semi_planform = SemiPlanform(c_node_exposed, y_node_exposed)
        exposed_semi_planform.sort_chord_by_y()
        exposed_semi_planform.update()

        # compute the trapzoidal x quarter chord location of all definition sections
        self.__x_le_node = self.y_node * np.tan(self.sweep_qc) - c_node_trap / 4.

        # move root section to account for lex
        self.__x_le_node[0] -= self.lex_ratio * c_node_trap[0]

        # transform coordinate to LE
        self.__x_le_node += (self.lex_ratio + 0.25) * c_node_trap[0]

        # compute the aerodynamic center in the local coordinate system
        x_ac_local = semi_planform.get_aerodynamic_center(self.__x_le_node)

        # update
        self.area_gross = 2.0 * semi_planform.area
        self.area_exposed = 2.0 * exposed_semi_planform.area
        self.area_wetted = 2.0 * (1 + 0.2 * self.thickness_to_chord) * self.area_exposed
        self.mean_aerodynamic_chord = semi_planform.mean_aerodynamic_chord
        self.mean_aerodynamic_chord_exposed = exposed_semi_planform.mean_aerodynamic_chord
        self.mean_geometric_chord = semi_planform.mean_geometric_chord
        self.aerodynamic_center = self.__wing_origin + np.array([x_ac_local, 0, 0])

    def get_flapped_area(self, span_ratio_inner, span_ratio_outer):
        """
        Compute the wing flapped (flap affected) area given the span ratios of the flaps
        :param span_ratio_inner:
        :param span_ratio_outer:
        :return:
        """

        # get the corresponding chords of the span ratios
        chord_flap_inner, chord_flap_outer = \
            self.chord_from_y(np.array([span_ratio_inner, span_ratio_outer])*self.__semi_span)

        c_node_flapped = np.array([chord_flap_inner, chord_flap_outer])
        y_node_flapped = np.array([span_ratio_inner, span_ratio_outer])*self.__semi_span

        flapped_semi_planform = SemiPlanform(c_node_flapped, y_node_flapped)
        flapped_semi_planform.update()

        return 2.0*flapped_semi_planform.area