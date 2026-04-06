import contextlib
import io
import unittest

import radarsimpy_whitebox as rs


class Phase6PackageSurfaceTests(unittest.TestCase):
    def test_metadata_constants_are_exposed(self):
        self.assertEqual(rs.__version__, "15.1.0")
        self.assertEqual(rs.__author__, "RadarSimX")
        self.assertEqual(rs.__email__, "info@radarsimx.com")
        self.assertEqual(rs.__url__, "https://radarsimx.com")
        self.assertIn("__author__", rs.__all__)
        self.assertIn("__email__", rs.__all__)
        self.assertIn("__url__", rs.__all__)

    def test_get_version_matches_package_version(self):
        self.assertEqual(rs.get_version(), rs.__version__)

    def test_get_info_returns_expected_keys(self):
        info = rs.get_info()

        self.assertEqual(
            set(info),
            {
                "package",
                "version",
                "author",
                "website",
                "python_version",
                "platform",
                "modules",
                "simulation_engines",
                "dependencies",
            },
        )
        self.assertEqual(info["package"], "RadarSimPy")
        self.assertEqual(info["version"], rs.__version__)
        self.assertEqual(info["author"], rs.__author__)
        self.assertEqual(info["website"], rs.__url__)
        self.assertIn("sim_lidar", info["simulation_engines"])
        self.assertIn("numpy", info["dependencies"])
        self.assertIn("scipy", info["dependencies"])

    def test_print_info_writes_human_readable_summary(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = rs.print_info()

        output = buffer.getvalue()
        self.assertIsNone(result)
        self.assertIn("RadarSimPy v15.1.0", output)
        self.assertIn("Core Modules:", output)
        self.assertIn("Simulation Engines:", output)
        self.assertIn("Dependencies:", output)

    def test_check_installation_reports_success_on_current_host(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = rs.check_installation()

        output = buffer.getvalue()
        self.assertTrue(status)
        self.assertIn("RadarSimPy installation appears complete", output)

    def test_hello_prints_quick_start(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = rs.hello()

        output = buffer.getvalue()
        self.assertIsNone(result)
        self.assertIn("Welcome to RadarSimPy!", output)
        self.assertIn("Quick Start:", output)
        self.assertIn("rs.sim_radar", output)


if __name__ == "__main__":
    unittest.main()
