import numpy as np
from scipy import interpolate


class AeroSurfacePlanform:
    """
    This representation is only for a semispan, and is therefore also
    suitable for unsymmetric surfaces
    """
    def __init__(self, c, y):
        """
        Constructor
        :param c: numpy array of definition chords
        :param y: numpy array of y position of definition chords
        :return:
        """
        self.c = c
        self.y = y
        self.length = None
        self.c_bar_panel = None
        self.dy = None
        self.panel_area = None
        self.area = None
        self.mean_aerodynamic_chord = None
        self.mean_geometric_chord = None

    def update(self):
        """
        Update all geometric parameters
        :return:
        """

        # length (semi-span)
        self.length = self.y[-1]-self.y[0]

        # panel lengths of the semi span
        self.dy = np.diff(self.y)

        # average chords for panels of the semi span
        self.c_bar_panel = (self.c[:-1]+self.c[1:])/2.

        # panel areas of the semi wing
        self.panel_area = self.c_bar_panel*self.dy

        # area of the semi wing
        self.area = sum(self.panel_area)

        # mean geometric chord
        self.mean_geometric_chord = np.dot(self.c_bar_panel, self.panel_area)/self.area

        # mean aerodynamic chord
        self.mean_aerodynamic_chord = self.__calc_mac()

    def sort_chord_by_y(self):
        """
        Sort the section chords by y
        :return:
        """
        sort_index = self.y.argsort()
        self.y = self.y[sort_index]
        self.c = self.c[sort_index]

    def get_chord_interpolator(self):
        """
        Create a chord interpolator
        :return:
        """
        return interpolate.interp1d(self.y, self.c, kind='linear')

    def __calc_mac(self):
        """
        Panel-area weight mean aerodynamic chord of general planform
        :return:
        """
        c_inner = self.c[:-1]
        c_outer = self.c[1:]
        return 2/3.*np.dot((c_inner+c_outer-c_inner*c_outer/(c_inner+c_outer)), self.panel_area)/self.area

    def get_aerodynamic_center(self, x_le):
        """
        Estimate the aerodynamic center
        :param x_le:
        :return:
        """
        x_qc = self.get_x_local(0.25, x_le)  # compute the quarter chord of the definition sections
        xac = 0
        for i, area in enumerate(self.panel_area):
            xac += self.__estimate_ac_panel(x_qc[i], x_qc[i+1], self.c[i], self.c[i+1], self.dy[i])

        # local aerodynamic center
        return xac*self.length/6/self.area

    def get_x_local(self, x_ratio, x_le):
        """
        get one local x coordinate based on LE location
        :param x_ratio: desired ratio, 0.25 for quarter chord
        :param x_le: leading edge x at definition sections
        :return:
        """
        return x_le+x_ratio*self.c

    @staticmethod
    def __estimate_ac_panel(x_qc1, x_qc2, c1, c2, dy):
        """
        Estimate the aerodynamic center of a single panel
        :param x_qc1: x quarter chord of first bounding section
        :param x_qc2: x quarter chord of second bounding section
        :param c1: chord of first bounding section
        :param c2: chord of second bounding section
        :param dy: span of panel
        :return:
        """
        # this is only an estimate
        return (2*x_qc1*c1+x_qc1*c2+x_qc2*c1+2*x_qc2*c2)*dy
