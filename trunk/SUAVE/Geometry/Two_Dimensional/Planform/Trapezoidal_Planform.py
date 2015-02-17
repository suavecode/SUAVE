import numpy as np

from Semi_Planform import Semi_Planform
from Planform import Planform


"""
Geometry calculations for a trapzoidal wing
    - Computes the gross, exposed and wetted areas
    - Computes the mean aerodynamic chord of the cranked and exposed planform
    - Estimates the aerodynamic center of the cranked planform
    - Includes option to compute the flapped wing area
"""


class Trapezoidal_Planform(Planform):
    def __init__(self, sref, ar, sweep_qc, taper,
                 span_ratio_fuselage=0.1):
        # call superclass constructor with 0 lex and tex
        super(Trapezoidal_Planform, self).__init__(sref, ar, sweep_qc, taper,
                                                  span_ratio_fuselage, 0, 0)

    def update(self):
        """
        Update the wing geometry
        :return:
        """

        # span of control points: [root, tip]
        y_node = np.array([0, 1]) * self.semi_span

        # trapezoidal wing definition chords
        self.c_trap = np.array([self.chord_root_trap, self.chord_tip])

        # root chord
        self.chord_root = self.chord_root_trap * (1 + self.lex_ratio + self.tex_ratio)

        # extended wing definition chords
        c_node = np.array([self.chord_root, self.chord_tip])

        # compute the extended wing semi-planform geometry
        self.semi_planform = Semi_Planform(c_node, y_node)
        # sort the chords with y; guard against case where fuselage is wider than break point
        self.semi_planform.sort_chord_by_y()
        self.semi_planform.update()

        # build a wing chord interpolant that can be reused
        self.chord_from_y = self.semi_planform.chord_from_y

        # get the fuselage-wing intersection chord
        chord_fuse_intersection = self.chord_from_y(self.span_ratio_fuselage * self.semi_span)

        # compute the exposed semi-planform properties
        c_node_exposed = np.array([chord_fuse_intersection, self.chord_tip])
        y_node_exposed = np.array([self.span_ratio_fuselage, 1]) * self.semi_span
        self.semi_planform_exposed = Semi_Planform(c_node_exposed, y_node_exposed)
        self.semi_planform_exposed.sort_chord_by_y()
        self.semi_planform_exposed.update()

        self.x_aerodynamic_center = self.wing_origin[0] + self.calc_x_ac(self.semi_planform, self.c_trap,
                                                                         self.sweep_qc, self.lex_ratio)

    def add_flap(self, span_ratio_inner, span_ratio_outer):
        """
        Compute the wing flapped (flap affected) area given the span ratios of the flaps
        :param span_ratio_inner:
        :param span_ratio_outer:
        :return:
        """

        # get the corresponding chords of the span ratios
        chord_flap_inner = self.chord_from_y(span_ratio_inner*self.semi_span)
        chord_flap_outer = self.chord_from_y(span_ratio_outer*self.semi_span)

        c_node_flapped = np.array([chord_flap_inner, chord_flap_outer])
        y_node_flapped = np.array([span_ratio_inner, span_ratio_outer])*self.semi_span

        self.semi_planform_flapped = Semi_Planform(c_node_flapped, y_node_flapped)
        self.semi_planform_flapped.update()
