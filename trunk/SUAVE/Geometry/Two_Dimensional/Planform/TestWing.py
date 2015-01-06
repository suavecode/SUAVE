import unittest
from SUAVE.Geometry.Two_Dimensional.Planform.CrankedPlanform import CrankedPlanform
import numpy as np
from SUAVE.Attributes import Units


class TestWing(unittest.TestCase):

    def setUp(self):
        pass

    def test_cranked_wing(self):

        # default wing from pass
        taper = 0.3
        sref = 1500
        sweep = 45*Units.degree
        ar = 8
        span_ratio_break = 0.5
        lex_ratio = 0.3
        tex_ratio = 0.4
        fuse_width = 11.448
        thickness_to_chord = 0.1123  # undefined

        span = np.sqrt(sref*ar)
        span_ratio_fuselage = fuse_width/span

        # compute wing planform geometry
        wpt = CrankedPlanform(sref, ar, sweep, taper, span_ratio_fuselage,
                              span_ratio_break, lex_ratio, tex_ratio)

        # pass-computed quantities
        span_pass = 109.54451
        wing_x_position = 0.417
        fuse_length = 107.82533
        mac_pass = 21.811018
        sref_pass = 1500
        chord_wing_fuse_intersection_pass = 31.189398
        x_ac_pass = 73.60473

        # set the wing origin
        origin = np.array((fuse_length*wing_x_position, 0, 0))
        wpt.set_origin(origin)

        wpt.update()

        # span
        self.assertAlmostEqual(wpt.span, span_pass, 2)

        # chords
        self.assertAlmostEqual(wpt.mean_aerodynamic_chord, mac_pass, 2)
        # self.assertAlmostEqual(wpt.mean_geometric_chord, sref_pass/span_pass, 2)
        self.assertAlmostEqual(wpt.chord_from_y(fuse_width/2.), chord_wing_fuse_intersection_pass, 3)

        # areas
        # self.assertAlmostEqual(wpt.area_gross, sgross_pass, 2)

        # TODO: test wetted and exposed areas
        self.assertLessEqual(wpt.area_exposed, wpt.area_gross)
        self.assertLessEqual(wpt.area_exposed, wpt.calc_area_wetted(thickness_to_chord)/2.)

        # test flapped area

        # aerodynamic center - this is not very close

        ac_delta = (x_ac_pass - wpt.x_aerodynamic_center)/x_ac_pass
        self.assertLessEqual(abs(ac_delta), 0.01)

    def test_tapered_wing(self):

        # default wing from pass
        taper = 0.2036
        sref = 1000.7
        sweep = 24.5*Units.degree
        ar = 8.701
        span_ratio_break = 0.3
        lex_ratio = 0
        tex_ratio = 0
        fuse_width = 11.448
        thickness_to_chord = 0.1123  # undefined

        span = np.sqrt(sref*ar)
        span_ratio_fuselage = fuse_width/span

        # compute wing planform geometry
        wpt = CrankedPlanform(sref, ar, sweep, taper, span_ratio_fuselage,
                              span_ratio_break, lex_ratio, tex_ratio)

        # pass-computed quantities
        span_pass = 93.31179
        wing_x_position = 0.417
        fuse_length = 107.82533
        mac_pass = 12.289369
        sref_pass = 1000.7
        chord_wing_fuse_intersection_pass = 16.079144
        x_ac_pass = 57.70458

        # set the wing origin
        origin = np.array((fuse_length*wing_x_position, 0, 0))
        wpt.set_origin(origin)

        wpt.update()

        # span
        self.assertAlmostEqual(wpt.span, span_pass, 2)

        # chords
        self.assertAlmostEqual(wpt.mean_aerodynamic_chord, mac_pass, 2)
        self.assertAlmostEqual(wpt.mean_geometric_chord, sref_pass/span_pass, 2)
        self.assertAlmostEqual(wpt.chord_from_y(fuse_width/2.), chord_wing_fuse_intersection_pass, 3)

        # areas
        self.assertAlmostEqual(wpt.area_gross, sref_pass, 2)

        # TODO: test wetted and exposed areas
        self.assertLessEqual(wpt.area_exposed, wpt.area_gross)
        self.assertLessEqual(wpt.area_exposed, wpt.calc_area_wetted(thickness_to_chord)/2.)

        # test flapped area

        # aerodynamic center
        self.assertAlmostEqual(wpt.x_aerodynamic_center, x_ac_pass , 2)

if __name__ == '__main__':
    unittest.main()