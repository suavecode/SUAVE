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

        # computed quantities
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
        self.chord_break = None
        self.chord_tip = None

        self.__semi_span = None
        self.__wing_origin = np.array([0, 0, 0])

    def update(self):
        """
        Update the wing geometry
        :return:
        """

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
        cranked_planform = AeroSurfacePlanform(c_node, y_node)
        cranked_planform.update()
        # cranked_planform.sort_chord_by_y()

        # build a wing chord interpolant that can be reused
        self.chord_from_y = cranked_planform.get_chord_interpolator()

        # get the fuselage-wing intersection chord
        chord_fuse_intersection = self.chord_from_y(self.span_ratio_fuselage*self.__semi_span)

        # compute the exposed semi-planform properties
        c_node_exposed = np.array([chord_fuse_intersection, self.chord_break, self.chord_tip])
        y_node_exposed = np.array([self.span_ratio_fuselage, self.span_ratio_break, 1])*self.__semi_span
        exposed_planform = AeroSurfacePlanform(c_node_exposed, y_node_exposed)
        exposed_planform.sort_chord_by_y()
        exposed_planform.update()

        # compute the trapzoidal x quater chord location of all definition sections
        x_le_node = y_node*np.tan(np.radians(self.sweep)) - c_node_trap/4.

        # include the effect of the lex
        x_le_node[0] -= self.lex_ratio*c_node_trap[0]

        # compute the aerodynamic center in the local coordinate system
        x_ac_local = cranked_planform.get_aerodynamic_center(x_le_node)

        # update
        self.area_gross = 2.0*cranked_planform.area
        self.area_exposed = 2.0*exposed_planform.area
        self.area_wetted = 2.0*(1+0.2*self.thickness_to_chord)*self.area_exposed
        self.mean_aerodynamic_chord = cranked_planform.mean_aerodynamic_chord
        self.mean_aerodynamic_chord_exposed = exposed_planform.mean_aerodynamic_chord
        self.mean_geometric_chord = cranked_planform.mean_geometric_chord
        self.aerodynamic_center = self.__wing_origin + np.array([x_ac_local, 0, 0])

    def wing_origin(self, value):
        """
        Set the wing origin coordinates
        """
        self.__wing_origin = value

    def get_flapped_area(self, span_ratio_inner, span_ratio_outer):
        """
        Compute the wing flapped (flap affected) area given the span ratios of the flaps
        :param span_ratio_inner:
        :param span_ratio_outer:
        :return:
        """

        # TODO: these limitation should be addressed in the future using a sort and clip operation
        # assert(span_ratio_outer > self.span_ratio_break)
        # assert(span_ratio_inner < self.span_ratio_break)

        # get the corresponding chords of the span ratios
        chord_flap_inner = self.chord_from_y(span_ratio_inner*self.__semi_span)
        chord_flap_outer = self.chord_from_y(span_ratio_outer*self.__semi_span)
        c_node_flapped = np.array([chord_flap_inner, chord_flap_outer])
        y_node_flapped = np.array([span_ratio_inner, span_ratio_outer])*self.__semi_span

        # add in the break section if we need to
        if span_ratio_outer > self.span_ratio_break > span_ratio_inner:
            y_node_flapped = np.insert(y_node_flapped, 1, self.span_ratio_break*self.__semi_span)
            c_node_flapped = np.insert(c_node_flapped, 1, self.chord_break)
            
        planform_flapped = AeroSurfacePlanform(c_node_flapped, y_node_flapped)
        planform_flapped.update()
        return 2.0*planform_flapped.area