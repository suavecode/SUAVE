import numpy as np

"""
Geometry calculations for a trapzoidal wing
    - Computes the gross, exposed and wetted areas
    - Computes the mean aerodynamic chord of the cranked and exposed planform
    - Estimates the aerodynamic center of the cranked planform
    - Includes option to compute the flapped wing area
"""


class Planform(object):
    def __init__(self, sref, ar, sweep_qc, taper,
                 span_ratio_fuselage=0.1,
                 lex_ratio=0, tex_ratio=0):
        # input parameters
        self.sref = sref
        self.ar = ar
        self.taper = taper
        self.lex_ratio = lex_ratio
        self.tex_ratio = tex_ratio
        self.sweep_qc = sweep_qc
        self.span_ratio_fuselage = span_ratio_fuselage
        self.wing_origin = np.array([0, 0, 0])

        # computed parameters
        self.span = None
        self.chord_from_y = None
        self.chord_root = None
        self.chord_tip = None
        self.chord_root_trap = None
        self.semi_span = None
        self.c_trap = None
        self.x_aerodynamic_center = None

        self.semi_planform = None
        self.semi_planform_exposed = None
        self.semi_planform_flapped = None

        # span
        self.span = np.sqrt(self.ar * self.sref)

        # semi-span
        self.semi_span = self.span / 2.

        # trapezoidal wing root chord
        self.chord_root_trap = 2 * self.sref / self.span / (1 + self.taper)

        # tip chord, defined in terms of the trapezoidal wing
        self.chord_tip = self.taper * self.chord_root_trap

    def set_wing_origin(self, value):
        """
        Set the wing origin coordinates
        """
        self.wing_origin = value

    @staticmethod
    def calc_x_ac(semi_planform, c_trap, sweep_qc, lex_ratio):
        """

        :param semi_planform:
        :param c_trap:
        :param sweep_qc:
        :param lex_ratio:
        :return:
        """

        # trapezoidal wing definition chords
        x_le_node = semi_planform.y * np.tan(sweep_qc) - c_trap / 4.

        # move root section to account for lex
        x_le_node[0] -= lex_ratio * c_trap[0]

        # transform coordinate to be referenced from root leading edge
        x_le_node += (lex_ratio + 0.25) * c_trap[0]

        # compute local aerodynamic center
        return semi_planform.get_aerodynamic_center(x_le_node)

    def calc_area_wetted(self, thickness_to_chord=0.12):
        """
        Compute the wetted area using the thickness to chord ratio
        :param thickness_to_chord:
        :return:
        """
        return 2 * (1 + 0.2 * thickness_to_chord) * self.area_exposed

    @property
    def mean_geometric_chord(self):
        return self.semi_planform.mean_geometric_chord

    @property
    def mean_aerodynamic_chord(self):
        return self.semi_planform.mean_aerodynamic_chord

    @property
    def mean_aerodynamic_chord_exposed(self):
        return self.semi_planform_exposed.mean_aerodynamic_chord

    @property
    def area_gross(self):
        return 2*self.semi_planform.area

    @property
    def area_exposed(self):
        return 2*self.semi_planform_exposed.area

    @property
    def area_flapped(self):
        return 2*self.semi_planform_flapped.area

