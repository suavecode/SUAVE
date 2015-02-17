import numpy as np
from scipy import integrate
from scipy import interpolate

"""
Geometry calculations for a general semi-planform defined by arrays of chords and y-stations
    - Computes the semi-area
    - Computes the mean aerodynamic chord
    - Has the option to estimate the aerodynamic center
    - Aerodynamic center expressed in the local coordinate system relative to the leading edge of the wing root
    - Can represent unsymmetric surfaces like the vertical tail
"""


class Semi_Planform(object):
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
        self.semi_span = None
        self.area = None
        self.mean_aerodynamic_chord = None
        self.mean_geometric_chord = None
        self.chord_from_y = None

    def update(self):
        """
        Update geometric parameters
        :return:
        """

        # length (semi-span)
        self.semi_span = self.y[-1] - self.y[0]

        # area of the semi planform
        self.area = np.trapz(self.c, self.y)

        # compute interpolant
        self.chord_from_y = interpolate.interp1d(self.y, self.c, kind='linear')

        # mean geometric chord
        self.mean_geometric_chord = self.area/self.semi_span

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

    def get_aerodynamic_center(self, x_le, ac_chord_ratio=0.25):
        """
        Estimate the local ac
        :param x_le:
        :param ac_chord_ratio:
        :return:
        """

        # TODO: this is an estimate
        # TODO: May have to relocate this to the analysis since ac position is Mach-dependent
        x_qc_local = self.__get_x_local(ac_chord_ratio, x_le)  # compute the quarter chord of the definition sections
        x_ac = integrate.simps(x_qc_local*self.c, self.y) / self.area  # this is approximate
        # x_ac = self.__integrate_ac(x_qc_local) / self.area
        return x_ac

    def __get_x_local(self, x_ratio, x_le):
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

    def __calc_mac(self):
        """
        Panel-area weight mean aerodynamic chord of general planform
        :return:
        """

        # panel areas
        area_panel = (self.c[:-1] + self.c[1:]) / 2. * np.diff(self.y)

        c_inner = self.c[:-1]
        c_outer = self.c[1:]

        # linear estimate
        mac = 2 / 3. * np.dot((c_inner + c_outer - c_inner * c_outer / (c_inner + c_outer)),
                              area_panel) / self.area

        return mac

    def __integrate_ac(self, x_ac_local):

        # greate a linear local ac interpolant
        x_ac_local_interpolant = interpolate.interp1d(self.y, x_ac_local)

        # function for x_ac_local(y)*c(y)
        g = lambda y: x_ac_local_interpolant(y)*self.chord_from_y(y)

        # integrate over the semispan using quadrature
        x_ac, _ = integrate.quadrature(g, self.y[0], self.y[-1])

        # return area-weight x_ac
        return x_ac/self.area
