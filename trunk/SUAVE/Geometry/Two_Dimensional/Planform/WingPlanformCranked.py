import numpy as np
import math as math
from AeroSurfacePlanform import AeroSurfacePlanform


class WingPlanformCranked:

    def __init__(self, sref, ar, sweep, taper,
                 thickness_to_chord=0.12, span_ratio_fuselage=0.1,
                 span_ratio_break=0.3, lex_ratio=0, tex_ratio=0):

        self.sref = sref
        self.ar = ar
        self.taper = taper
        self.thickness_to_chord = thickness_to_chord
        self.lex_ratio = lex_ratio
        self.tex_ratio = tex_ratio
        self.sweep = sweep
        self.span_ratio_break = span_ratio_break
        self.span_ratio_fuselage = span_ratio_fuselage

        self.span = None
        self.chord_from_eta = None
        self.mean_aerodynamic_chord = None
        self.mean_geometric_chord = None
        self.area_gross = None
        self.area_exposed = None
        self.area_wetted = None
        self.aerodynamic_center = None
        self.chord_root = None
        self.chord_break = None
        self.chord_tip = None

        self.__semi_span = None
        self.__wing_origin = np.array([0, 0, 0])

    def update(self):

        # span
        self.span = math.sqrt(self.ar*self.sref)
        self.__semi_span = self.span/2.

        # trapzoidal wing root chord
        chord_root_trap = 2*self.sref/self.span/(1+self.taper)

        # tip chord
        self.chord_tip = self.taper * chord_root_trap

        # span of control points: [root, break, tip]
        y_node = np.array([0, self.span_ratio_break, 1])*self.__semi_span

        # break chord
        self.chord_break = chord_root_trap + self.span_ratio_break*(self.chord_tip-chord_root_trap)

        # trapezoidal wing definition chords
        c_node_trap = np.array([chord_root_trap, self.chord_break, self.chord_tip])

        # root chord
        self.chord_root = chord_root_trap*(1+self.lex_ratio+self.tex_ratio)

        # extended wing definition chords
        c_node = np.array([self.chord_root, self.chord_break, self.chord_tip])

        # compute the extended wing semi-planform geometry
        extended_wing = AeroSurfacePlanform(c_node, y_node)
        extended_wing.update()

        # build a wing chord interpolant that can be reused
        self.chord_from_eta = extended_wing.make_chord_interpolator()

        # get the fuselage-wing intersection chord
        chord_fuse_intersection = self.chord_from_eta(self.span_ratio_fuselage)

        # compute the exposed semi-planform properties
        c_node_exposed = np.array([chord_fuse_intersection, self.chord_break, self.chord_tip])
        y_node_exposed = np.array([self.span_ratio_fuselage, self.span_ratio_break, 1])*self.__semi_span
        planform_exposed = AeroSurfacePlanform(c_node_exposed, y_node_exposed)
        planform_exposed.sort_sections()
        planform_exposed.update()

        # compute aerodynamic center
        x_ac_local = self.__calc_aerodynamic_center(self.sweep, self.lex_ratio, y_node, c_node_trap, extended_wing)

        # update
        self.area_gross = 2.0*extended_wing.area
        self.area_exposed = 2.0*planform_exposed.area
        self.area_wetted = 2.0*(1+0.2*self.thickness_to_chord)*self.area_exposed
        self.mean_aerodynamic_chord = extended_wing.mean_aerodynamic_chord
        self.mean_geometric_chord = extended_wing.mean_geometric_chord
        self.aerodynamic_center = self.__wing_origin + np.array([x_ac_local, 0, 0])

        return 0

    def wing_origin(self, value):
        """
        set the wing origin location
        """
        self.__wing_origin = value


    @staticmethod
    def __calc_aerodynamic_center(sweep, lex_ratio, y_node, c_node_trap, extended_wing):
        # re-factor this
        # move to the main function. Always compute the AC?
        # compute x leading edge in local coordinates
        # TODO: check and optimize this part
        x_qc_trap_node = c_node_trap[0]/4.+y_node*np.tan(np.radians(sweep))
        x_le_node = x_qc_trap_node - c_node_trap/4.
        x_le_node[0] -= lex_ratio*x_le_node[0]

        # compute the aerodynamic center in the local coordinate system
        return extended_wing.calc_aerodynamic_center(x_le_node)

    def calc_flapped_area(self, span_ratio_flap_inner, span_ratio_flap_outer):
        """

        :param span_ratio_flap_inner:
        :param span_ratio_flap_outer:
        :return:
        """

        # TODO: these are limitations, should be addressed in the future using a sort and clip operation
        assert(span_ratio_flap_outer > self.span_ratio_break)  # flap terminates outboard of wing break
        chord_flap_outer = self.chord_from_eta(span_ratio_flap_outer)
        chord_flap_inner = self.chord_from_eta(span_ratio_flap_inner)
        c_node_flapped = np.array([chord_flap_inner, self.chord_break, chord_flap_outer])
        y_node_flapped = np.array([span_ratio_flap_inner, self.span_ratio_break, span_ratio_flap_outer])*self.__semi_span
        planform_flapped = AeroSurfacePlanform(c_node_flapped, y_node_flapped)
        planform_flapped.update()
        return 2.0*planform_flapped.area