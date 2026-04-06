import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import radarsimpy_whitebox as wb
from oracle_capture.capture_suite import run_capture
from oracle_capture.run_oracle_capture import DEFAULT_MODULE_NAMES, _import_module


class OracleCaptureSuiteTests(unittest.TestCase):
    def test_run_capture_writes_manifest_and_artifacts(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            manifest = run_capture(wb, output_dir, module_name="radarsimpy_whitebox")

            self.assertEqual(manifest["module_name"], "radarsimpy_whitebox")
            self.assertEqual(manifest["metadata"]["version"], wb.__version__)

            manifest_path = output_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())

            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["module_name"], "radarsimpy_whitebox")
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_target"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_moving"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_moving_negative"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_moving_multiframe"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_moving_negative_multiframe"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_multiframe"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_phase"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_moving_doppler"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_moving_doppler_negative"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_multi"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_static_moving"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_static_moving_negative"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_static_static_moving"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_static_static_moving_negative"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_moving"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_moving_negative"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_static_moving"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_static_moving_negative"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_static_mixed_sign_multiframe"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_negative_multiframe"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_static_moving_moving_multiframe"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_static_moving_multiframe"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_moving_multiframe"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["point_mimo_offaxis_static_moving_negative_multiframe"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["dry_run"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["mesh_target"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_radar"]["mesh_aspect"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_lidar"]["baseline"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_lidar"]["moving"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_lidar"]["multi_hit"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_rcs"]["broadside"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_rcs"]["obs_backscatter"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_rcs"]["obs_phi_sweep"]["ok"])
            self.assertTrue(saved_manifest["captures"]["sim_rcs"]["polarization_cases"]["ok"])
            self.assertIn("scenario_index", saved_manifest)
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-001"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-002"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-004"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-005"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-006"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-007"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-008"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-009"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-010"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-011"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-012"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-013"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-014"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-015"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-016"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-017"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-018"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-019"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-020"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-021"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-022"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-023"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-024"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-025"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-026"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-027"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-028"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-029"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-030"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-031"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-PT-032"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["SIM-MESH-002"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["LIDAR-003"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["RCS-002"]["status"], "captured")
            self.assertEqual(saved_manifest["scenario_index"]["RCS-003"]["status"], "captured")

            artifacts_dir = output_dir / "artifacts"
            self.assertTrue((artifacts_dir / "plate.stl").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_target.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_moving.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_moving_negative.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_moving_multiframe.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_moving_negative_multiframe.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_multiframe.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_phase.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_moving_doppler.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_moving_doppler_negative.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_multi.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_static_moving.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_static_moving_negative.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_static_static_moving.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_static_static_moving_negative.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_static_static_moving_moving.npz").exists())
            self.assertTrue(
                (artifacts_dir / "sim_radar_point_static_static_moving_moving_negative.npz").exists()
            )
            self.assertTrue(
                (artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_moving_moving.npz").exists()
            )
            self.assertTrue(
                (artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_mixed_sign.npz").exists()
            )
            self.assertTrue(
                (
                    artifacts_dir
                    / "sim_radar_point_mimo_offaxis_static_static_mixed_sign_multiframe.npz"
                ).exists()
            )
            self.assertTrue(
                (
                    artifacts_dir
                    / "sim_radar_point_mimo_offaxis_static_static_moving_moving_negative_multiframe.npz"
                ).exists()
            )
            self.assertTrue(
                (
                    artifacts_dir
                    / "sim_radar_point_mimo_offaxis_static_static_moving_moving_multiframe.npz"
                ).exists()
            )
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_static_moving.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_static_moving_negative.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_moving.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_moving_negative.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_moving_multiframe.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_moving_negative_multiframe.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_static_moving_multiframe.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_point_mimo_offaxis_static_moving_negative_multiframe.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_dry_run.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_mesh_target.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_mesh_broadside.npz").exists())
            self.assertTrue((artifacts_dir / "sim_radar_mesh_edge_on.npz").exists())
            self.assertTrue((artifacts_dir / "sim_lidar_hits.npy").exists())
            self.assertTrue((artifacts_dir / "sim_lidar_moving_hits.npy").exists())
            self.assertTrue((artifacts_dir / "sim_lidar_multi_hits.npy").exists())
            self.assertTrue((artifacts_dir / "sim_rcs_obs_phi_sweep.npz").exists())
            self.assertTrue((artifacts_dir / "sim_rcs_polarization_cases.npz").exists())

    def test_saved_artifacts_are_loadable(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            run_capture(wb, output_dir, module_name="radarsimpy_whitebox")

            with np.load(output_dir / "artifacts" / "sim_radar_point_target.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving_negative.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving_multiframe.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving_negative_multiframe.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_multiframe.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_phase.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_moving_doppler.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_moving_doppler_negative.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_multi.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_static_moving.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_static_moving_negative.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_static_static_moving.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_static_static_moving_negative.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_static_static_moving_moving.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(
                output_dir / "artifacts" / "sim_radar_point_static_static_moving_moving_negative.npz"
            ) as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(
                output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving_moving.npz"
            ) as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(
                output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_mixed_sign.npz"
            ) as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(
                output_dir
                / "artifacts"
                / "sim_radar_point_mimo_offaxis_static_static_mixed_sign_multiframe.npz"
            ) as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(
                output_dir
                / "artifacts"
                / "sim_radar_point_mimo_offaxis_static_static_moving_moving_negative_multiframe.npz"
            ) as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(
                output_dir
                / "artifacts"
                / "sim_radar_point_mimo_offaxis_static_static_moving_moving_multiframe.npz"
            ) as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving_negative.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving_negative.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving_multiframe.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving_negative_multiframe.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving_multiframe.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving_negative_multiframe.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_mesh_target.npz") as data:
                self.assertIn("baseband", data.files)
                self.assertIn("noise", data.files)
                self.assertIn("timestamp", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_mesh_broadside.npz") as data:
                self.assertIn("baseband", data.files)

            with np.load(output_dir / "artifacts" / "sim_radar_mesh_edge_on.npz") as data:
                self.assertIn("baseband", data.files)

            lidar_hits = np.load(
                output_dir / "artifacts" / "sim_lidar_hits.npy",
                allow_pickle=False,
            )
            self.assertIsNotNone(lidar_hits.dtype.names)
            self.assertIn("positions", lidar_hits.dtype.names)

            moving_hits = np.load(
                output_dir / "artifacts" / "sim_lidar_moving_hits.npy",
                allow_pickle=False,
            )
            self.assertIsNotNone(moving_hits.dtype.names)
            self.assertIn("positions", moving_hits.dtype.names)

            multi_hits = np.load(
                output_dir / "artifacts" / "sim_lidar_multi_hits.npy",
                allow_pickle=False,
            )
            self.assertIsNotNone(multi_hits.dtype.names)
            self.assertIn("positions", multi_hits.dtype.names)
            self.assertEqual(len(multi_hits), 3)

            with np.load(output_dir / "artifacts" / "sim_rcs_obs_phi_sweep.npz") as data:
                self.assertIn("obs_phi", data.files)
                self.assertIn("rcs", data.files)

            with np.load(output_dir / "artifacts" / "sim_rcs_polarization_cases.npz") as data:
                self.assertIn("case_names", data.files)
                self.assertIn("rcs", data.files)

    def test_import_module_uses_fallback_order(self):
        module_name, module = _import_module(
            ["definitely_missing_capture_module", "radarsimpy_whitebox"]
        )

        self.assertEqual(module_name, "radarsimpy_whitebox")
        self.assertIs(module, wb)

    def test_default_module_order_prefers_vendor_runtime_names(self):
        self.assertEqual(
            DEFAULT_MODULE_NAMES,
            [
                "radarsimpy",
                "radarsimpy_macos_arm",
                "radarsimpy_macos",
                "radarsimpy_origin",
            ],
        )
