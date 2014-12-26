import numpy as np
from SemiPlanform import SemiPlanform

"""
Geometry calculations for a trapzoidal wing
    - Computes the gross, exposed and wetted areas
    - Computes the mean aerodynamic chord of the cranked and exposed planform
    - Estimates the aerodynamic center of the cranked planform
    - Includes option to compute the flapped wing area
"""


class Planform(object):

    def __init__(self, sref, ar, sweep_qc, taper,
                 thickness_to_chord=0.12,
                 span_ratio_fuselage=0.1,
                 lex_ratio=0, tex_ratio=0):

        # input parameters
        self.sref = sref
        self.ar = ar
        self.taper = taper
        self.thickness_to_chord = thickness_to_chord
        self.lex_ratio = lex_ratio
        self.tex_ratio = tex_ratio
        self.sweep_qc = sweep_qc
        self.span_ratio_fuselage = span_ratio_fuselage

        # computed parameters
        self.span = None
        self.chord_from_y = None
        self.mean_aerodynamic_chord = None
        self.mean_geometric_chord = None
        self.mean_aerodynamic_chord_exposed = None
        self.area_gross = None
        self.area_exposed = None
        self.area_wetted = None
        self.aerodynamic_center = None
        self.chord_root = None
        self.chord_tip = None
        self.chord_root_trap = None

        # instance parameters
        self.semi_span = None
        self.wing_origin = np.array([0, 0, 0])
        self.x_le_node = None
        self.c_node = None
        self.y_node = None

        # span
        self.span = np.sqrt(self.ar * self.sref)

        # semi-span
        self.semi_span = self.span / 2.

        # trapezoidal wing root chord
        self.chord_root_trap = 2 * self.sref / self.span / (1 + self.taper)

        # tip chord, defined in terms of the trapezoidal wing
        self.chord_tip = self.taper * self.chord_root_trap

    def get_wing_coordinates(self):
        """
        # write out the planform for plotting
        :return:
        """

        x_te_node = self.x_le_node+self.c_node
        x = np.append(self.x_le_node, x_te_node[::-1], self.x_le_node[0])
        y = np.append(self.y_node, self.y_node[::-1], self.y_node[0])
        return x, y

    def set_wing_origin(self, value):
        """
        Set the wing origin coordinates
        """
        self.wing_origin = value

