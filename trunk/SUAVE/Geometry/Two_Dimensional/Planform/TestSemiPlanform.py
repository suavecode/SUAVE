import unittest
from SUAVE.Geometry.Two_Dimensional.Planform.SemiPlanform import SemiPlanform
import numpy as np


class TestSemiPanform(unittest.TestCase):

    def setUp(self):

        # set up the straight semi-planform
        c = np.array((1, 1, 1))
        y = np.array((0, 10, 20))
        self.semi_planform_straight = SemiPlanform(c, y)
        self.semi_planform_straight.sort_chord_by_y()
        self.semi_planform_straight.update()

        # Tapered wing extracted from AA241 calculator
        taper = 0.2036
        c_root = 17.820307
        span = 93.31179
        c = np.array((c_root, c_root*taper))
        y = np.array((0, span/2))
        self.semi_planform_tapered = SemiPlanform(c, y)
        self.semi_planform_tapered.sort_chord_by_y()
        self.semi_planform_tapered.update()

    def test_cranked_semi_planform(self):

        # test area
        self.assertAlmostEqual(self.semi_planform_tapered.area, 1000.7/2, places=4)

    def test_straight_wing(self):

        # test area
        self.assertEqual(self.semi_planform_straight.area, 20)

        # test length
        self.assertEqual(self.semi_planform_straight.semi_span, 20)

        # test mac
        c_root = self.semi_planform_straight.c[0]
        c_tip = self.semi_planform_straight.c[-1]
        mac = 2/3.*(c_root+c_tip-c_root*c_tip/(c_root+c_tip))
        self.assertEqual(self.semi_planform_straight.mean_aerodynamic_chord, mac)

        # test aerodynamic center (unswept)
        x_le = np.array((0, 0, 0))
        self.assertEqual(self.semi_planform_straight.get_aerodynamic_center(x_le, 0), 0)
        self.assertEqual(self.semi_planform_straight.get_aerodynamic_center(x_le, 0.25), 0.25)
        self.assertEqual(self.semi_planform_straight.get_aerodynamic_center(x_le, 0.5), 0.5)
        self.assertEqual(self.semi_planform_straight.get_aerodynamic_center(x_le, 1), 1)

        # test aerodynamic center (45 deg)
        x_le = np.array((0, 10, 20))
        self.assertEqual(self.semi_planform_straight.get_aerodynamic_center(x_le, 0), 10)

        x_le = np.array((0, 10, 20))
        self.assertEqual(self.semi_planform_straight.get_aerodynamic_center(x_le, 0.25), 10+0.25)

        # test aerodynamic center (-45 deg)
        x_le = np.array((0, -10, -20))
        self.assertEqual(self.semi_planform_straight.get_aerodynamic_center(x_le, 0.25), -10+0.25)

if __name__ == '__main__':
    unittest.main()