import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import radarsimpy_whitebox as wb

from oracle_capture.capture_suite import run_capture
from oracle_capture.compare_suite import compare_against_oracle
from oracle_capture.scenario_index import build_compare_scenario_results


VENDOR_ORACLE_DIR = Path(
    "/Users/seongcheoljeong/Documents/RadarSimPy/oracle_capture_output/macos_arm_py311"
)


class OracleCompareSuiteTests(unittest.TestCase):
    def test_scenario_results_do_not_mark_unavailable_reports_as_matched(self):
        report = {
            "comparisons": {
                "sim_rcs": {
                    "polarization": {
                        "available": False,
                        "overall": True,
                    }
                }
            }
        }

        scenario_results = build_compare_scenario_results(report)

        self.assertEqual(scenario_results["RCS-003"]["available"], False)
        self.assertIsNone(scenario_results["RCS-003"]["overall"])
        self.assertEqual(scenario_results["RCS-003"]["status"], "not_compared")

    def test_compare_against_self_capture_reports_full_match(self):
        with TemporaryDirectory() as temp_dir:
            oracle_dir = Path(temp_dir) / "oracle"
            run_capture(wb, oracle_dir, module_name="radarsimpy_whitebox")

            report = compare_against_oracle(
                wb,
                oracle_dir,
                candidate_module_name="radarsimpy_whitebox",
            )

        self.assertTrue(report["overall"])
        self.assertTrue(report["comparisons"]["license"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_target"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_negative_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_negative_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_phase"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_phase"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_moving_doppler"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_moving_doppler"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_moving_doppler_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_moving_doppler_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_multi"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_multi"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving_moving"]["overall"])
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_static_static_moving_moving_negative"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_static_static_moving_moving_negative"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign_multiframe"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign_multiframe"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_negative_multiframe"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_negative_multiframe"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_multiframe"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_multiframe"]["overall"]
        )
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_negative_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_negative_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_negative_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_negative_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["seed_repeat"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["mesh_target"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["mesh_target"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["mesh_aspect"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["mesh_aspect"]["overall"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["overall"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["multi_hit"]["available"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["multi_hit"]["overall"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["overall"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["cases"]["obs_backscatter"]["available"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["sweep"]["available"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["sweep"]["overall"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["polarization"]["available"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["polarization"]["overall"])
        self.assertIn("scenario_results", report)
        self.assertEqual(report["scenario_results"]["SIM-PT-002"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-004"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-005"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-006"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-007"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-008"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-009"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-010"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-011"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-012"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-013"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-014"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-015"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-016"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-017"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-018"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-019"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-020"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-021"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-022"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-023"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-024"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-025"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-026"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-027"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-028"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-029"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-030"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-031"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-032"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-MESH-002"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["LIDAR-003"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["RCS-002"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["RCS-003"]["status"], "matched")

    def test_compare_against_oracle_writes_report_file(self):
        with TemporaryDirectory() as temp_dir:
            oracle_dir = Path(temp_dir) / "oracle"
            output_path = Path(temp_dir) / "report.json"
            run_capture(wb, oracle_dir, module_name="radarsimpy_whitebox")

            report = compare_against_oracle(
                wb,
                oracle_dir,
                candidate_module_name="radarsimpy_whitebox",
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            saved_report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["candidate_module_name"], "radarsimpy_whitebox")
            self.assertEqual(saved_report["overall"], report["overall"])
            self.assertIn("scenario_results", saved_report)

    def test_vendor_oracle_now_reports_full_extended_match(self):
        if not (VENDOR_ORACLE_DIR / "manifest.json").exists():
            raise unittest.SkipTest("Vendor oracle bundle is not available")

        report = compare_against_oracle(
            wb,
            VENDOR_ORACLE_DIR,
            candidate_module_name="radarsimpy_whitebox",
        )

        self.assertTrue(report["comparisons"]["license"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_negative_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_moving_negative_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_phase"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_phase"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_moving_doppler"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_moving_doppler"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_moving_doppler_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_moving_doppler_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_multi"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_multi"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_static_static_moving_moving"]["overall"])
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_static_static_moving_moving_negative"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_static_static_moving_moving_negative"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign_multiframe"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign_multiframe"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_negative_multiframe"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_negative_multiframe"]["overall"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_multiframe"]["available"]
        )
        self.assertTrue(
            report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_multiframe"]["overall"]
        )
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_negative"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_negative"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_negative_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_static_moving_negative_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_negative_multiframe"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["point_mimo_offaxis_static_moving_negative_multiframe"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["mesh_target"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["mesh_target"]["overall"])
        self.assertTrue(report["comparisons"]["sim_radar"]["mesh_aspect"]["available"])
        self.assertTrue(report["comparisons"]["sim_radar"]["mesh_aspect"]["overall"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["overall"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["moving"]["available"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["moving"]["overall"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["multi_hit"]["available"])
        self.assertTrue(report["comparisons"]["sim_lidar"]["multi_hit"]["overall"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["overall"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["cases"]["normal_incidence"]["available"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["cases"]["obs_backscatter"]["available"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["sweep"]["available"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["sweep"]["overall"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["polarization"]["available"])
        self.assertTrue(report["comparisons"]["sim_rcs"]["polarization"]["overall"])
        self.assertEqual(report["scenario_results"]["SIM-PT-001"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-002"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-004"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-005"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-006"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-007"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-008"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-009"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-010"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-011"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-012"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-013"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-014"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-015"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-016"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-017"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-018"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-019"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-020"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-021"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-022"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-023"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-024"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-025"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-026"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-027"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-028"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-029"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-030"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-031"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-PT-032"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["SIM-MESH-002"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["LIDAR-002"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["LIDAR-003"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["RCS-002"]["status"], "matched")
        self.assertEqual(report["scenario_results"]["RCS-003"]["status"], "matched")
        self.assertTrue(report["overall"])
