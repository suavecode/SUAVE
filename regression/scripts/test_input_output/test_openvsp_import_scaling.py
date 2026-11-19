"""Check public OpenVSP geometry import using a generated wing and fuselage."""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import vsp
except ImportError:
    try:
        import openvsp as vsp
    except ImportError:
        vsp = None

from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read
from SUAVE.Input_Output.OpenVSP.vsp_fuselage import read_vsp_fuselage
from SUAVE.Input_Output.OpenVSP.vsp_wing import read_vsp_wing


def write_model(filename, scale):
    vsp.ClearVSPModel()
    wing = vsp.AddGeom("WING")
    vsp.SetGeomName(wing, "TestWing")
    vsp.InsertXSec(wing, 1, vsp.XS_FOUR_SERIES)
    vsp.Update()
    surface = vsp.GetXSecSurf(wing, 0)
    for index, root, tip in ((1, 4, 3), (2, 3, 2)):
        vsp.SetDriverGroup(wing, index, vsp.SPAN_WSECT_DRIVER,
                           vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
        section = vsp.GetXSec(surface, index)
        for name, value in (("Span", 5), ("Root_Chord", root), ("Tip_Chord", tip),
                            ("Sweep", 0), ("Dihedral", 0)):
            vsp.SetParmVal(vsp.GetXSecParm(section, name), value)
    fuselage = vsp.AddGeom("FUSELAGE")
    vsp.SetGeomName(fuselage, "TestFuselage")
    vsp.SetParmVal(fuselage, "Length", "Design", 12)
    vsp.SetParmVal(wing, "X_Rel_Location", "XForm", 3)
    vsp.SetParmVal(fuselage, "X_Rel_Location", "XForm", 7)
    vsp.Update()
    for component in (wing, fuselage):
        vsp.SetParmValUpdate(component, "Scale", "XForm", scale)
    vsp.WriteVSPFile(str(filename), vsp.SET_ALL)


@unittest.skipIf(vsp is None, "OpenVSP Python bindings are unavailable")
class OpenVSPImportScaling(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        self.addCleanup(vsp.ClearVSPModel)
        self.directory = Path(directory.name)
        os.chdir(directory.name)
        self.filename = self.directory / "geometry.vsp3"


    def test_dimensions(self):
        for scale in (0.5, 1.0, 2.0):
            write_model(self.filename, scale)
            for units, to_meters in (("SI", 1.0), ("imperial", 0.3048), ("inches", 0.0254)):
                with self.subTest(scale=scale, units=units):
                    vehicle = vsp_read(str(self.filename), units_type=units)
                    wing = next(iter(vehicle.wings.values()))
                    fuselage = next(iter(vehicle.fuselages.values()))
                    self.assertAlmostEqual(wing.spans.projected, 20 * scale * to_meters)
                    self.assertAlmostEqual(wing.areas.reference, 60 * (scale * to_meters)**2)
                    self.assertAlmostEqual(wing.chords.root, 4 * scale * to_meters)
                    self.assertAlmostEqual(wing.chords.tip, 2 * scale * to_meters)
                    self.assertAlmostEqual(wing.origin[0][0], 3 * to_meters)
                    self.assertAlmostEqual(fuselage.origin[0][0], 7 * to_meters)
                    length = 12 * scale * to_meters
                    self.assertAlmostEqual(fuselage.lengths.total, length)
                    self.assertAlmostEqual(sum(s.length for s in fuselage.Segments.values()), length)
                    fuse_id = vsp.FindGeomsWithName("TestFuselage")[0]
                    section = vsp.GetXSec(vsp.GetXSecSurf(fuse_id, 0), 1)
                    imported = list(fuselage.Segments.values())[1]
                    self.assertAlmostEqual(imported.height, vsp.GetXSecHeight(section) * to_meters)
                    self.assertAlmostEqual(imported.width, vsp.GetXSecWidth(section) * to_meters)

    def test_scaling_flag_does_not_disable_airfoil_export(self):
        write_model(self.filename, 0.5)
        wing_id = vsp.FindGeomsWithName("TestWing")[0]
        read_vsp_wing(wing_id, write_airfoil_file=False)
        self.assertFalse(list(self.directory.glob("*_airfoil_XSec_*.dat")))
        vehicle = vsp_read(str(self.filename), use_scaling=False)
        self.assertEqual(len(list(self.directory.glob("*_airfoil_XSec_*.dat"))), 3)
        self.assertAlmostEqual(next(iter(vehicle.wings.values())).spans.projected, 10.0)
        self.assertAlmostEqual(next(iter(vehicle.fuselages.values())).lengths.total, 6.0)

    def test_fineness_can_be_disabled_independently(self):
        write_model(self.filename, 0.5)
        fuselage_id = vsp.FindGeomsWithName("TestFuselage")[0]
        with patch("SUAVE.Input_Output.OpenVSP.vsp_fuselage.compute_fuselage_fineness") as compute_fineness:
            fuselage = read_vsp_fuselage(fuselage_id, 0, 1, fineness=False)
        compute_fineness.assert_not_called()
        self.assertAlmostEqual(fuselage.lengths.total, 6.0)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OpenVSPImportScaling)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise AssertionError("OpenVSP import regression failed")


if __name__ == "__main__":
    main()
