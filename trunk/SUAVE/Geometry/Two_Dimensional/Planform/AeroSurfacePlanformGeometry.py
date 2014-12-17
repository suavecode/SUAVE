import numpy as np
from scipy import interpolate


class AeroSurfacePlanformGeometry:

    def __init__(self, c, y):
        self.c = c
        self.y = y
        self.span = y[-1]-y[0]
        self.c_bar = None
        self.dy = None
        self.a_panel = None
        self.area = None
        self.mean_aerodynamic_chord = None

    def update(self):

        # panel lengths of the semi span
        self.dy = np.diff(self.y)

        # average chords for panels of the semi span
        self.c_bar = (self.c[:-1]+self.c[1:])/2.

        # panel areas of the semi span
        self.a_panel = self.c_bar*self.dy

        # surface semi-area
        self.area = sum(self.a_panel)

        # mean aerodynamic chord
        self.mean_aerodynamic_chord = self.__calc_mac()

    def get_chord_interpolator(self):
        return interpolate.interp1d(self.y, self.c, kind='linear')

    def __calc_mac(self):
        """
        Panel-area weight mean aerodynamic chord of general planform
        :return:
        """
        c_inner = self.c[:-1]
        c_outer = self.c[1:]
        return np.dot(2/3.*(c_inner+c_outer-c_inner*c_outer/(c_inner+c_outer)), self.a_panel)/self.area

    def calc_aerodynamic_center(self, x_le):
        """
        Compute the aerodynamic chord based on a general wing
        :param c: array of chords at definition sections
        :param x_le: array of wing x le at definition sections
        :param areas: array of areas for panels in between definition sections
        :param span: span
        :param sgross: gross wing area
        :return:
        """
        x_qc = self.get_x_local(0.25, x_le)  # compute the quarter chord of the definition sections
        xac = 0
        for i, area in enumerate(self.a_panel):
            xac += self.__get_ac_panel(x_qc[i], x_qc[i+1], self.c[i], self.c[i+1], self.a_panel)
        return xac*self.span/6/self.area

    def get_x_local(self, x_ratio, x_le):
        """
        Get one local x coordinate from another set
        :param x_ratio: desired ratio, 0.25 for quarter chord
        :param x_le: leading edge x at definition sections
        :return:
        """
        return x_le+x_ratio*self.c

    @staticmethod
    def __get_ac_panel(x1, x2, c1, c2, dy):
        """

        :param x1:
        :param x2:
        :param c1:
        :param c2:
        :param dy:
        :return:
        """
        # TODO: this may not be correct - need to go by the definitions
        return (2*x1*c1+x1*c2+x2*c1+2*x2*c2)*dy
