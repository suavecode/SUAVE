import math as math

import numpy as np

from SemiPlanform import SemiPlanform
from Planform import Planform


"""
Geometry calculations for a singlely crank wing (with lex and/or tex)
    - Computes the gross, exposed and wetted areas
    - Computes the mean aerodynamic chord of the cranked and exposed planform
    - Estimates the aerodynamic center of the cranked planform
    - Includes option to compute the flapped wing area
"""


class CrankedPlanform(Planform):
    def __init__(self, sref, ar, sweep_qc, taper,
                 thickness_to_chord=0.12, span_ratio_fuselage=0.1,
                 span_ratio_break=0.3, lex_ratio=0, tex_ratio=0):

        # call superclass constructor with lex and tex
        super(CrankedPlanform, self).__init__(sref, ar, sweep_qc, taper,
                                              thickness_to_chord, span_ratio_fuselage,
                                              lex_ratio, tex_ratio)

        # input parameters
        self.span_ratio_break = span_ratio_break

        # computed parameters
        self.chord_break = None

    def update(self):
        """
        Update the wing geometry
        :return:
        """

        # span of control points: [root, break, tip]
        y_node = np.array([0, self.span_ratio_break, 1]) * self.semi_span

        # compute the break chord
        self.chord_break = self.chord_root_trap + self.span_ratio_break * (self.chord_tip - self.chord_root_trap)

        # trapezoidal wing definition chords
        self.c_trap = np.array([self.chord_root_trap, self.chord_break, self.chord_tip])

        # root chord
        self.chord_root = self.chord_root_trap * (1 + self.lex_ratio + self.tex_ratio)

        # extended wing definition chords
        c_node = np.array([self.chord_root, self.chord_break, self.chord_tip])

        # compute the extended wing semi-planform geometry
        semi_planform = SemiPlanform(c_node, y_node)
        # sort the chords with y; guard against case where fuselage is wider than break point
        semi_planform.sort_chord_by_y()
        semi_planform.update()

        # build a wing chord interpolant that can be reused
        self.chord_from_y = semi_planform.get_chord_interpolant()

        # get the fuselage-wing intersection chord
        chord_fuse_intersection = self.chord_from_y(self.span_ratio_fuselage * self.semi_span)

        # compute the exposed semi-planform properties
        c_node_exposed = np.array([chord_fuse_intersection, self.chord_break, self.chord_tip])
        y_node_exposed = np.array([self.span_ratio_fuselage, self.span_ratio_break, 1]) * self.semi_span
        exposed_semi_planform = SemiPlanform(c_node_exposed, y_node_exposed)
        exposed_semi_planform.sort_chord_by_y()
        exposed_semi_planform.update()

        # update
        self._semi_planform = semi_planform
        self._exposed_semi_planform = exposed_semi_planform

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

        # add the break section definition if it is within the flapped area
        if span_ratio_outer > self.span_ratio_break > span_ratio_inner:
            y_node_flapped = np.insert(y_node_flapped, 1, self.span_ratio_break*self.semi_span)
            c_node_flapped = np.insert(c_node_flapped, 1, self.chord_break)

        flapped_semi_planform = SemiPlanform(c_node_flapped, y_node_flapped)
        flapped_semi_planform.update()

        # update
        self._flapped_semi_planform = flapped_semi_planform

