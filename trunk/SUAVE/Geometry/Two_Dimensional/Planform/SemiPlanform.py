import numpy as np
from scipy import interpolate

"""
Geometry calculations for a general semi-planform defined by arrays of chords and y-stations
    - Computes the semi-area
    - Computes the mean aerodynamic chord
    - Has the option to estimate the aerodynamic center
    - Aerodynamic center expressed in the local coordinate system relative to the leading edge of the wing root
    - Can represent unsymmetric surfaces like the vertical tail
"""


class SemiPlanform:
    def __init__(self, c, y):
        """
        Constructor
        :param c: numpy array of definitional chords
        :param y: numpy array of y-station of chords
        :return:
        """

        # input parameters
        self.c = c
        self.y = y

        # computed parameters
        self.length = None
        self.area = None
        self.mean_aerodynamic_chord = None
        self.mean_geometric_chord = None

        # instance parameters
        self.__area_panel = None
        self.__c_bar_panel = None
        self.__dy = None

    def update(self):
        """
        Update all geometric parameters
        :return:
        """

        # length (semi-span)
        self.length = self.y[-1] - self.y[0]

        # panel lengths of the semi span
        self.__dy = np.diff(self.y)

        # average chords for panels of the semi span
        self.__c_bar_panel = (self.c[:-1] + self.c[1:]) / 2.

        # panel areas of the semi wing
        self.__area_panel = self.__c_bar_panel * self.__dy

        # area of the semi wing
        self.area = sum(self.__area_panel)

        # mean geometric chord
        self.mean_geometric_chord = np.dot(self.__c_bar_panel, self.__area_panel) / self.area

        # mean aerodynamic chord
        self.mean_aerodynamic_chord = self.__calc_mac()

    def sort_chord_by_y(self):
        """
        Sort the section chords by y-stations
        :return:
        """
        sort_index = self.y.argsort()
        self.y = self.y[sort_index]
        self.c = self.c[sort_index]

    def get_chord_interpolant(self, interp_type='linear'):
        """
        Create a chord interpolator
        :return: scipy interpolator
        """
        return interpolate.interp1d(self.y, self.c, kind=interp_type)

    def __calc_mac(self):
        """
        Panel-area weight mean aerodynamic chord of general planform
        :return:
        """
        c_inner = self.c[:-1]
        c_outer = self.c[1:]
        return 2 / 3. * np.dot((c_inner + c_outer - c_inner * c_outer / (c_inner + c_outer)),
                               self.__area_panel) / self.area

    def get_aerodynamic_center(self, x_le, ac_chord_ratio=0.25):
        """
        Estimate the aerodynamic center (area weighted location of quarter chord)
        :param x_le:
        :return: local aerodynamic center
        """

        # TODO: this is a rather crude estimate. Alternatively one could use the 1/4 chord of the mean aerodynamic chord
        # TODO: May have to relocate this to the analysis since ac position is Mach-dependent
        x_qc = self.get_x_local(ac_chord_ratio, x_le)  # compute the quarter chord of the definition sections
        return np.trapz(x_qc, self.y) / self.area

    def get_x_local(self, x_ratio, x_le):
        """
        get one local x coordinate based on LE location
        :param x_ratio: desired ratio, 0.25 for quarter chord
        :param x_le: leading edge x at definition sections
        :return:
        """

        return x_le + x_ratio * self.c

    def get_wing_coordinates(self, x_le_node):
        """
        # write out the connected planform coordinates for plotting
        :return:
        """
        x_te_node = x_le_node + self.c
        x = np.append(x_le_node, x_te_node[::-1], x_le_node[0])
        y = np.append(self.y, self.y[::-1], self.y[0])
        return x, y