from pathlib import Path
import tempfile
import unittest

import radarsimpy_whitebox as rs


TEST_LICENSE_TEXT = """===========================================
        RADARSIMX LICENSE KEY
===========================================

Order ID: 99999
License ID: 11111111-2222-3333-4444-555555555555
Product: RadarSimPy
Version: 15
Platform: All
Customer: test@example.com
Name: WHITEBOX TEST USER
Purchase Date: 2099-01-01
Expiration Date: 2099-12-31
Generated: 2099-01-01 00:00:00
"""


class Phase8LicenseSurfaceTests(unittest.TestCase):
    def tearDown(self):
        rs.set_license()

    def test_license_functions_are_exposed(self):
        self.assertIn("set_license", rs.__all__)
        self.assertIn("is_licensed", rs.__all__)
        self.assertIn("get_license_info", rs.__all__)

    def test_default_license_state_is_human_readable(self):
        self.assertTrue(rs.is_licensed())

        info = rs.get_license_info()
        self.assertIsInstance(info, str)
        self.assertIn("License Status: Licensed", info)
        self.assertIn("Product: RadarSimPy", info)
        self.assertIn("Current Platform:", info)

    def test_set_license_accepts_explicit_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            license_path = Path(temp_dir) / "test_license.lic"
            license_path.write_text(TEST_LICENSE_TEXT, encoding="utf-8")

            result = rs.set_license(license_path)

        self.assertIsNone(result)
        self.assertTrue(rs.is_licensed())
        info = rs.get_license_info()
        self.assertIn("Licensed to: WHITEBOX TEST USER", info)
        self.assertIn("License Platform: All", info)

    def test_set_license_rejects_missing_path(self):
        missing_path = Path("/Users/seongcheoljeong/Documents/RadarSimPy/does-not-exist.lic")

        with self.assertRaises(FileNotFoundError):
            rs.set_license(missing_path)


if __name__ == "__main__":
    unittest.main()
