import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from radarsimpy_whitebox import (
    Radar,
    Receiver,
    Transmitter,
    processing,
    sim_lidar,
    sim_radar,
    sim_rcs,
)


PLATE_STL = """solid plate
facet normal 1 0 0
  outer loop
    vertex 0 -0.5 -0.5
    vertex 0 0.5 -0.5
    vertex 0 0.5 0.5
  endloop
endfacet
facet normal 1 0 0
  outer loop
    vertex 0 -0.5 -0.5
    vertex 0 0.5 0.5
    vertex 0 -0.5 0.5
  endloop
endfacet
endsolid plate
"""

VENDOR_ORACLE_DIR = Path(
    "/Users/seongcheoljeong/Documents/RadarSimPy/oracle_capture_output/macos_arm_py311"
)


class Phase7VendorOracleParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest_path = VENDOR_ORACLE_DIR / "manifest.json"
        if not manifest_path.exists():
            raise unittest.SkipTest("Vendor oracle bundle is not available")
        cls.vendor_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def build_radar(self, *, seed=None):
        tx = Transmitter(
            f=[76.0e9, 76.2e9],
            t=[0.0, 40e-6],
            tx_power=13,
            pulses=4,
            prp=60e-6,
            channels=[{"location": [0.0, 0.0, 0.0]}],
        )
        rx = Receiver(
            fs=4e6,
            noise_figure=11,
            rf_gain=0,
            load_resistor=500,
            baseband_gain=0,
            bb_type="complex",
            channels=[{"location": [0.0, 0.0, 0.0]}],
        )
        return Radar(tx, rx, seed=seed)

    def build_mimo_radar(self, *, seed=None):
        tx = Transmitter(
            f=[76.0e9, 76.2e9],
            t=[0.0, 40e-6],
            tx_power=13,
            pulses=4,
            prp=60e-6,
            channels=[
                {"location": [0.0, 0.0, 0.0]},
                {"location": [0.0, 0.05, 0.0]},
            ],
        )
        rx = Receiver(
            fs=4e6,
            noise_figure=11,
            rf_gain=0,
            load_resistor=500,
            baseband_gain=0,
            bb_type="complex",
            channels=[
                {"location": [0.0, 0.0, 0.0]},
                {"location": [0.0, 0.03, 0.0]},
            ],
        )
        return Radar(tx, rx, seed=seed)

    def build_mimo_multiframe_radar(self, *, seed=None):
        tx = Transmitter(
            f=[76.0e9, 76.2e9],
            t=[0.0, 40e-6],
            tx_power=13,
            pulses=4,
            prp=60e-6,
            channels=[
                {"location": [0.0, 0.0, 0.0]},
                {"location": [0.0, 0.05, 0.0]},
            ],
        )
        rx = Receiver(
            fs=4e6,
            noise_figure=11,
            rf_gain=0,
            load_resistor=500,
            baseband_gain=0,
            bb_type="complex",
            channels=[
                {"location": [0.0, 0.0, 0.0]},
                {"location": [0.0, 0.03, 0.0]},
            ],
        )
        return Radar(tx, rx, frame_time=[0.0, 1e-3], seed=seed)

    def build_lidar(self):
        return {
            "position": [0.0, 0.1, 0.1],
            "phi": np.array([0.0]),
            "theta": np.array([90.0]),
        }

    def build_multi_hit_lidar(self):
        return {
            "position": [0.0, 0.0, 0.0],
            "phi": np.degrees(np.arctan2(np.array([-1.0, 0.0, 1.0]), 10.0)),
            "theta": np.array([90.0]),
        }

    def write_temp_stl(self) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        mesh_path = Path(tmpdir.name) / "plate.stl"
        mesh_path.write_text(PLATE_STL, encoding="utf-8")
        return str(mesh_path)

    def test_sim_radar_matches_vendor_oracle_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_target.npz"
        radar = self.build_radar(seed=2026)
        target = {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target])

        self.assertIsNone(ours["interference"])
        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["timestamp"].dtype, ours["timestamp"].dtype)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["baseband"].dtype, ours["baseband"].dtype)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertEqual(vendor["noise"].dtype, ours["noise"].dtype)

            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))

    def test_sim_radar_dry_run_matches_vendor_null_interference_contract(self):
        radar = self.build_radar(seed=2026)
        target = {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target], dry_run=True)

        self.assertIsNone(ours["interference"])
        self.assertTrue(np.all(ours["baseband"] == 0))
        self.assertTrue(np.all(ours["noise"] == 0))

    def test_sim_radar_mimo_matches_vendor_oracle_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo.npz"
        radar = self.build_mimo_radar(seed=2026)
        target = {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target])

        self.assertIsNone(ours["interference"])
        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))

    def test_sim_radar_offaxis_mimo_matches_vendor_spatial_phase_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_offaxis.npz"
        radar = self.build_mimo_radar(seed=2026)
        target = {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target])

        self.assertIsNone(ours["interference"])
        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))

            peak_sample_index = int(np.argmax(np.abs(vendor["baseband"][0, 0])))
            vendor_ref = vendor["baseband"][0, 0, peak_sample_index]
            ours_ref = ours["baseband"][0, 0, peak_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:, 0, peak_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:, 0, peak_sample_index] / ours_ref)
            )

        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))
        self.assertEqual(
            np.round(vendor_rel_deg, 6).tolist(),
            [0.0, -53.926637, -88.965609, -142.892246],
        )

    def test_sim_radar_offaxis_moving_mimo_matches_vendor_joint_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_offaxis_moving.npz"
        radar = self.build_mimo_radar(seed=2026)
        target = {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target])

        self.assertIsNone(ours["interference"])
        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(vendor_rd)), vendor_rd.shape))
            ours_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(ours_rd)), ours_rd.shape))

            vendor_phase_sample_index = 2
            ours_phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, vendor_phase_sample_index]
            ours_ref = ours["baseband"][0, 0, ours_phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:, 0, vendor_phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:, 0, ours_phase_sample_index] / ours_ref)
            )

        self.assertEqual(vendor_peak, (1, 67))
        self.assertEqual(ours_peak, vendor_peak)
        self.assertEqual(vendor_phase_sample_index, 2)
        self.assertEqual(ours_phase_sample_index, vendor_phase_sample_index)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))

    def test_sim_radar_offaxis_negative_speed_mimo_matches_vendor_joint_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_offaxis_moving_negative.npz"
        radar = self.build_mimo_radar(seed=2026)
        target = {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target])

        self.assertIsNone(ours["interference"])
        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(vendor_rd)), vendor_rd.shape))
            ours_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(ours_rd)), ours_rd.shape))

            vendor_phase_sample_index = 2
            ours_phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, vendor_phase_sample_index]
            ours_ref = ours["baseband"][0, 0, ours_phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:, 0, vendor_phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:, 0, ours_phase_sample_index] / ours_ref)
            )

        self.assertEqual(vendor_peak, (3, 67))
        self.assertEqual(ours_peak, vendor_peak)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))

    def test_sim_radar_offaxis_moving_mimo_multiframe_matches_vendor_joint_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_offaxis_moving_multiframe.npz"
        radar = self.build_mimo_multiframe_radar(seed=2026)
        target = {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target])

        self.assertIsNone(ours["interference"])
        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(vendor_rd)), vendor_rd.shape))
            ours_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(ours_rd)), ours_rd.shape))

            vendor_phase_sample_index = 2
            ours_phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, vendor_phase_sample_index]
            ours_ref = ours["baseband"][0, 0, ours_phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:4, 0, vendor_phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:4, 0, ours_phase_sample_index] / ours_ref)
            )

        self.assertEqual(vendor_peak, (1, 67))
        self.assertEqual(ours_peak, vendor_peak)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))

    def test_sim_radar_offaxis_negative_speed_mimo_multiframe_matches_vendor_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_offaxis_moving_negative_multiframe.npz"
        )
        radar = self.build_mimo_multiframe_radar(seed=2026)
        target = {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target])

        self.assertIsNone(ours["interference"])
        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(vendor_rd)), vendor_rd.shape))
            ours_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(ours_rd)), ours_rd.shape))

            vendor_phase_sample_index = 2
            ours_phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, vendor_phase_sample_index]
            ours_ref = ours["baseband"][0, 0, ours_phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:4, 0, vendor_phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:4, 0, ours_phase_sample_index] / ours_ref)
            )

        self.assertEqual(vendor_peak, (3, 67))
        self.assertEqual(ours_peak, vendor_peak)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))

    def test_sim_radar_negative_moving_point_target_matches_vendor_doppler_direction_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_moving_doppler_negative.npz"
        radar = self.build_radar(seed=2026)
        target = {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target])

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_peak = tuple(
                int(x) for x in np.unravel_index(int(np.argmax(vendor_rd)), vendor_rd.shape)
            )
            ours_peak = tuple(
                int(x) for x in np.unravel_index(int(np.argmax(ours_rd)), ours_rd.shape)
            )

        self.assertEqual(vendor_peak, (3, 67))
        self.assertEqual(ours_peak, vendor_peak)

    def test_sim_radar_two_point_targets_match_vendor_range_profile_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_multi.npz"
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            vendor_top_bins = np.argsort(vendor_profile)[-6:][::-1]
            ours_top_bins = np.argsort(ours_profile)[-6:][::-1]
            self.assertTrue(np.array_equal(vendor_top_bins, ours_top_bins))
            correlation = np.vdot(vendor_profile, ours_profile)
            correlation /= np.linalg.norm(vendor_profile)
            correlation /= np.linalg.norm(ours_profile)
            self.assertGreaterEqual(float(np.abs(correlation)), 0.999)

    def test_sim_radar_static_and_moving_targets_match_vendor_mixed_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_static_moving.npz"
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_static_window = vendor_rd[0:2, 23:32]
            ours_static_window = ours_rd[0:2, 23:32]
            vendor_moving_window = vendor_rd[0:4, 63:72]
            ours_moving_window = ours_rd[0:4, 63:72]

            vendor_static_peak = np.unravel_index(
                int(np.argmax(vendor_static_window)),
                vendor_static_window.shape,
            )
            ours_static_peak = np.unravel_index(
                int(np.argmax(ours_static_window)),
                ours_static_window.shape,
            )
            vendor_moving_peak = np.unravel_index(
                int(np.argmax(vendor_moving_window)),
                vendor_moving_window.shape,
            )
            ours_moving_peak = np.unravel_index(
                int(np.argmax(ours_moving_window)),
                ours_moving_window.shape,
            )

            vendor_static_peak = (int(vendor_static_peak[0]), int(vendor_static_peak[1] + 23))
            ours_static_peak = (int(ours_static_peak[0]), int(ours_static_peak[1] + 23))
            vendor_moving_peak = (int(vendor_moving_peak[0]), int(vendor_moving_peak[1] + 63))
            ours_moving_peak = (int(ours_moving_peak[0]), int(ours_moving_peak[1] + 63))

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_peak, (0, 27))
        self.assertEqual(ours_static_peak, vendor_static_peak)
        self.assertEqual(vendor_moving_peak, (1, 67))
        self.assertEqual(ours_moving_peak, vendor_moving_peak)
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_static_and_negative_moving_targets_match_vendor_mixed_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_static_moving_negative.npz"
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_static_window = vendor_rd[0:2, 23:32]
            ours_static_window = ours_rd[0:2, 23:32]
            vendor_moving_window = vendor_rd[2:4, 63:72]
            ours_moving_window = ours_rd[2:4, 63:72]

            vendor_static_peak = np.unravel_index(
                int(np.argmax(vendor_static_window)),
                vendor_static_window.shape,
            )
            ours_static_peak = np.unravel_index(
                int(np.argmax(ours_static_window)),
                ours_static_window.shape,
            )
            vendor_moving_peak = np.unravel_index(
                int(np.argmax(vendor_moving_window)),
                vendor_moving_window.shape,
            )
            ours_moving_peak = np.unravel_index(
                int(np.argmax(ours_moving_window)),
                ours_moving_window.shape,
            )

            vendor_static_peak = (int(vendor_static_peak[0]), int(vendor_static_peak[1] + 23))
            ours_static_peak = (int(ours_static_peak[0]), int(ours_static_peak[1] + 23))
            vendor_moving_peak = (int(vendor_moving_peak[0] + 2), int(vendor_moving_peak[1] + 63))
            ours_moving_peak = (int(ours_moving_peak[0] + 2), int(ours_moving_peak[1] + 63))

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_peak, (0, 27))
        self.assertEqual(ours_static_peak, vendor_static_peak)
        self.assertEqual(vendor_moving_peak, (3, 67))
        self.assertEqual(ours_moving_peak, vendor_moving_peak)
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_static_static_and_moving_targets_match_vendor_mixed_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_static_static_moving.npz"
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_static_a_window = vendor_rd[0:2, 23:32]
            ours_static_a_window = ours_rd[0:2, 23:32]
            vendor_static_b_window = vendor_rd[0:2, 43:52]
            ours_static_b_window = ours_rd[0:2, 43:52]
            vendor_moving_window = vendor_rd[0:4, 63:72]
            ours_moving_window = ours_rd[0:4, 63:72]

            vendor_static_a_peak = np.unravel_index(
                int(np.argmax(vendor_static_a_window)),
                vendor_static_a_window.shape,
            )
            ours_static_a_peak = np.unravel_index(
                int(np.argmax(ours_static_a_window)),
                ours_static_a_window.shape,
            )
            vendor_static_b_peak = np.unravel_index(
                int(np.argmax(vendor_static_b_window)),
                vendor_static_b_window.shape,
            )
            ours_static_b_peak = np.unravel_index(
                int(np.argmax(ours_static_b_window)),
                ours_static_b_window.shape,
            )
            vendor_moving_peak = np.unravel_index(
                int(np.argmax(vendor_moving_window)),
                vendor_moving_window.shape,
            )
            ours_moving_peak = np.unravel_index(
                int(np.argmax(ours_moving_window)),
                ours_moving_window.shape,
            )

            vendor_static_a_peak = (int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23))
            ours_static_a_peak = (int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23))
            vendor_static_b_peak = (int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43))
            ours_static_b_peak = (int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43))
            vendor_moving_peak = (int(vendor_moving_peak[0]), int(vendor_moving_peak[1] + 63))
            ours_moving_peak = (int(ours_moving_peak[0]), int(ours_moving_peak[1] + 63))

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peak, (0, 27))
        self.assertEqual(ours_static_a_peak, vendor_static_a_peak)
        self.assertEqual(vendor_static_b_peak, (0, 47))
        self.assertEqual(ours_static_b_peak, vendor_static_b_peak)
        self.assertEqual(vendor_moving_peak, (1, 67))
        self.assertEqual(ours_moving_peak, vendor_moving_peak)
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_static_static_and_negative_moving_targets_match_vendor_mixed_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_static_static_moving_negative.npz"
        )
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_static_a_window = vendor_rd[0:2, 23:32]
            ours_static_a_window = ours_rd[0:2, 23:32]
            vendor_static_b_window = vendor_rd[0:2, 43:52]
            ours_static_b_window = ours_rd[0:2, 43:52]
            vendor_moving_window = vendor_rd[2:4, 63:72]
            ours_moving_window = ours_rd[2:4, 63:72]

            vendor_static_a_peak = np.unravel_index(
                int(np.argmax(vendor_static_a_window)),
                vendor_static_a_window.shape,
            )
            ours_static_a_peak = np.unravel_index(
                int(np.argmax(ours_static_a_window)),
                ours_static_a_window.shape,
            )
            vendor_static_b_peak = np.unravel_index(
                int(np.argmax(vendor_static_b_window)),
                vendor_static_b_window.shape,
            )
            ours_static_b_peak = np.unravel_index(
                int(np.argmax(ours_static_b_window)),
                ours_static_b_window.shape,
            )
            vendor_moving_peak = np.unravel_index(
                int(np.argmax(vendor_moving_window)),
                vendor_moving_window.shape,
            )
            ours_moving_peak = np.unravel_index(
                int(np.argmax(ours_moving_window)),
                ours_moving_window.shape,
            )

            vendor_static_a_peak = (int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23))
            ours_static_a_peak = (int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23))
            vendor_static_b_peak = (int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43))
            ours_static_b_peak = (int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43))
            vendor_moving_peak = (int(vendor_moving_peak[0] + 2), int(vendor_moving_peak[1] + 63))
            ours_moving_peak = (int(ours_moving_peak[0] + 2), int(ours_moving_peak[1] + 63))

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peak, (0, 27))
        self.assertEqual(ours_static_a_peak, vendor_static_a_peak)
        self.assertEqual(vendor_static_b_peak, (0, 47))
        self.assertEqual(ours_static_b_peak, vendor_static_b_peak)
        self.assertEqual(vendor_moving_peak, (3, 67))
        self.assertEqual(ours_moving_peak, vendor_moving_peak)
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_static_static_and_moving_moving_targets_match_vendor_four_target_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_static_static_moving_moving.npz"
        )
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 0.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_static_a_window = vendor_rd[0:2, 23:32]
            ours_static_a_window = ours_rd[0:2, 23:32]
            vendor_static_b_window = vendor_rd[0:2, 43:52]
            ours_static_b_window = ours_rd[0:2, 43:52]
            vendor_moving_a_window = vendor_rd[0:4, 8:18]
            ours_moving_a_window = ours_rd[0:4, 8:18]
            vendor_moving_b_window = vendor_rd[0:4, 63:72]
            ours_moving_b_window = ours_rd[0:4, 63:72]

            vendor_static_a_peak = np.unravel_index(
                int(np.argmax(vendor_static_a_window)),
                vendor_static_a_window.shape,
            )
            ours_static_a_peak = np.unravel_index(
                int(np.argmax(ours_static_a_window)),
                ours_static_a_window.shape,
            )
            vendor_static_b_peak = np.unravel_index(
                int(np.argmax(vendor_static_b_window)),
                vendor_static_b_window.shape,
            )
            ours_static_b_peak = np.unravel_index(
                int(np.argmax(ours_static_b_window)),
                ours_static_b_window.shape,
            )
            vendor_moving_a_peak = np.unravel_index(
                int(np.argmax(vendor_moving_a_window)),
                vendor_moving_a_window.shape,
            )
            ours_moving_a_peak = np.unravel_index(
                int(np.argmax(ours_moving_a_window)),
                ours_moving_a_window.shape,
            )
            vendor_moving_b_peak = np.unravel_index(
                int(np.argmax(vendor_moving_b_window)),
                vendor_moving_b_window.shape,
            )
            ours_moving_b_peak = np.unravel_index(
                int(np.argmax(ours_moving_b_window)),
                ours_moving_b_window.shape,
            )

            vendor_static_a_peak = (int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23))
            ours_static_a_peak = (int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23))
            vendor_static_b_peak = (int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43))
            ours_static_b_peak = (int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43))
            vendor_moving_a_peak = (int(vendor_moving_a_peak[0]), int(vendor_moving_a_peak[1] + 8))
            ours_moving_a_peak = (int(ours_moving_a_peak[0]), int(ours_moving_a_peak[1] + 8))
            vendor_moving_b_peak = (int(vendor_moving_b_peak[0]), int(vendor_moving_b_peak[1] + 63))
            ours_moving_b_peak = (int(ours_moving_b_peak[0]), int(ours_moving_b_peak[1] + 63))

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peak, (0, 27))
        self.assertEqual(ours_static_a_peak, vendor_static_a_peak)
        self.assertEqual(vendor_static_b_peak, (0, 47))
        self.assertEqual(ours_static_b_peak, vendor_static_b_peak)
        self.assertEqual(vendor_moving_a_peak, (1, 14))
        self.assertEqual(ours_moving_a_peak, vendor_moving_a_peak)
        self.assertEqual(vendor_moving_b_peak, (1, 67))
        self.assertEqual(ours_moving_b_peak, vendor_moving_b_peak)
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_static_static_and_negative_moving_negative_moving_targets_match_vendor_four_target_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_static_static_moving_moving_negative.npz"
        )
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 0.0, 0.0], "rcs": 7.0, "speed": [-8.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_static_a_window = vendor_rd[0:2, 23:32]
            ours_static_a_window = ours_rd[0:2, 23:32]
            vendor_static_b_window = vendor_rd[0:2, 43:52]
            ours_static_b_window = ours_rd[0:2, 43:52]
            vendor_moving_a_window = vendor_rd[2:4, 8:18]
            ours_moving_a_window = ours_rd[2:4, 8:18]
            vendor_moving_b_window = vendor_rd[2:4, 63:72]
            ours_moving_b_window = ours_rd[2:4, 63:72]

            vendor_static_a_peak = np.unravel_index(
                int(np.argmax(vendor_static_a_window)),
                vendor_static_a_window.shape,
            )
            ours_static_a_peak = np.unravel_index(
                int(np.argmax(ours_static_a_window)),
                ours_static_a_window.shape,
            )
            vendor_static_b_peak = np.unravel_index(
                int(np.argmax(vendor_static_b_window)),
                vendor_static_b_window.shape,
            )
            ours_static_b_peak = np.unravel_index(
                int(np.argmax(ours_static_b_window)),
                ours_static_b_window.shape,
            )
            vendor_moving_a_peak = np.unravel_index(
                int(np.argmax(vendor_moving_a_window)),
                vendor_moving_a_window.shape,
            )
            ours_moving_a_peak = np.unravel_index(
                int(np.argmax(ours_moving_a_window)),
                ours_moving_a_window.shape,
            )
            vendor_moving_b_peak = np.unravel_index(
                int(np.argmax(vendor_moving_b_window)),
                vendor_moving_b_window.shape,
            )
            ours_moving_b_peak = np.unravel_index(
                int(np.argmax(ours_moving_b_window)),
                ours_moving_b_window.shape,
            )

            vendor_static_a_peak = (int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23))
            ours_static_a_peak = (int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23))
            vendor_static_b_peak = (int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43))
            ours_static_b_peak = (int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43))
            vendor_moving_a_peak = (int(vendor_moving_a_peak[0] + 2), int(vendor_moving_a_peak[1] + 8))
            ours_moving_a_peak = (int(ours_moving_a_peak[0] + 2), int(ours_moving_a_peak[1] + 8))
            vendor_moving_b_peak = (int(vendor_moving_b_peak[0] + 2), int(vendor_moving_b_peak[1] + 63))
            ours_moving_b_peak = (int(ours_moving_b_peak[0] + 2), int(ours_moving_b_peak[1] + 63))

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peak, (0, 27))
        self.assertEqual(ours_static_a_peak, vendor_static_a_peak)
        self.assertEqual(vendor_static_b_peak, (0, 47))
        self.assertEqual(ours_static_b_peak, vendor_static_b_peak)
        self.assertEqual(vendor_moving_a_peak, (3, 13))
        self.assertEqual(ours_moving_a_peak, vendor_moving_a_peak)
        self.assertEqual(vendor_moving_b_peak, (3, 66))
        self.assertEqual(ours_moving_b_peak, vendor_moving_b_peak)
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_static_and_moving_moving_mimo_matches_vendor_four_target_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_mimo_offaxis_static_static_moving_moving.npz"
        )
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_static_a_peaks = []
            ours_static_a_peaks = []
            vendor_static_b_peaks = []
            ours_static_b_peaks = []
            vendor_moving_a_peaks = []
            ours_moving_a_peaks = []
            vendor_moving_b_peaks = []
            ours_moving_b_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_a_window = vendor_rd[0:2, 23:32]
                ours_static_a_window = ours_rd[0:2, 23:32]
                vendor_static_b_window = vendor_rd[0:2, 43:52]
                ours_static_b_window = ours_rd[0:2, 43:52]
                vendor_moving_a_window = vendor_rd[0:4, 8:18]
                ours_moving_a_window = ours_rd[0:4, 8:18]
                vendor_moving_b_window = vendor_rd[0:4, 63:72]
                ours_moving_b_window = ours_rd[0:4, 63:72]

                vendor_static_a_peak = np.unravel_index(
                    int(np.argmax(vendor_static_a_window)),
                    vendor_static_a_window.shape,
                )
                ours_static_a_peak = np.unravel_index(
                    int(np.argmax(ours_static_a_window)),
                    ours_static_a_window.shape,
                )
                vendor_static_b_peak = np.unravel_index(
                    int(np.argmax(vendor_static_b_window)),
                    vendor_static_b_window.shape,
                )
                ours_static_b_peak = np.unravel_index(
                    int(np.argmax(ours_static_b_window)),
                    ours_static_b_window.shape,
                )
                vendor_moving_a_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_a_window)),
                    vendor_moving_a_window.shape,
                )
                ours_moving_a_peak = np.unravel_index(
                    int(np.argmax(ours_moving_a_window)),
                    ours_moving_a_window.shape,
                )
                vendor_moving_b_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_b_window)),
                    vendor_moving_b_window.shape,
                )
                ours_moving_b_peak = np.unravel_index(
                    int(np.argmax(ours_moving_b_window)),
                    ours_moving_b_window.shape,
                )

                vendor_static_a_peaks.append((int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23)))
                ours_static_a_peaks.append((int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23)))
                vendor_static_b_peaks.append((int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43)))
                ours_static_b_peaks.append((int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43)))
                vendor_moving_a_peaks.append((int(vendor_moving_a_peak[0]), int(vendor_moving_a_peak[1] + 8)))
                ours_moving_a_peaks.append((int(ours_moving_a_peak[0]), int(ours_moving_a_peak[1] + 8)))
                vendor_moving_b_peaks.append((int(vendor_moving_b_peak[0]), int(vendor_moving_b_peak[1] + 63)))
                ours_moving_b_peaks.append((int(ours_moving_b_peak[0]), int(ours_moving_b_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_phase_deg = np.degrees(
                np.angle(vendor["baseband"][:, 0, phase_sample_index] / vendor_ref)
            )
            ours_phase_deg = np.degrees(
                np.angle(ours["baseband"][:, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peaks, [(0, 27)] * 4)
        self.assertEqual(ours_static_a_peaks, vendor_static_a_peaks)
        self.assertEqual(vendor_static_b_peaks, [(0, 47)] * 4)
        self.assertEqual(ours_static_b_peaks, vendor_static_b_peaks)
        self.assertEqual(vendor_moving_a_peaks, [(1, 14)] * 4)
        self.assertEqual(ours_moving_a_peaks, vendor_moving_a_peaks)
        self.assertEqual(vendor_moving_b_peaks, [(1, 67)] * 4)
        self.assertEqual(ours_moving_b_peaks, vendor_moving_b_peaks)
        self.assertTrue(np.allclose(vendor_phase_deg, ours_phase_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_static_and_mixed_sign_movers_mimo_matches_vendor_four_target_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_mimo_offaxis_static_static_mixed_sign.npz"
        )
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_static_a_peaks = []
            ours_static_a_peaks = []
            vendor_static_b_peaks = []
            ours_static_b_peaks = []
            vendor_moving_pos_peaks = []
            ours_moving_pos_peaks = []
            vendor_moving_neg_peaks = []
            ours_moving_neg_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_a_window = vendor_rd[0:2, 23:32]
                ours_static_a_window = ours_rd[0:2, 23:32]
                vendor_static_b_window = vendor_rd[0:2, 43:52]
                ours_static_b_window = ours_rd[0:2, 43:52]
                vendor_moving_pos_window = vendor_rd[0:4, 8:18]
                ours_moving_pos_window = ours_rd[0:4, 8:18]
                vendor_moving_neg_window = vendor_rd[2:4, 63:72]
                ours_moving_neg_window = ours_rd[2:4, 63:72]

                vendor_static_a_peak = np.unravel_index(
                    int(np.argmax(vendor_static_a_window)),
                    vendor_static_a_window.shape,
                )
                ours_static_a_peak = np.unravel_index(
                    int(np.argmax(ours_static_a_window)),
                    ours_static_a_window.shape,
                )
                vendor_static_b_peak = np.unravel_index(
                    int(np.argmax(vendor_static_b_window)),
                    vendor_static_b_window.shape,
                )
                ours_static_b_peak = np.unravel_index(
                    int(np.argmax(ours_static_b_window)),
                    ours_static_b_window.shape,
                )
                vendor_moving_pos_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_pos_window)),
                    vendor_moving_pos_window.shape,
                )
                ours_moving_pos_peak = np.unravel_index(
                    int(np.argmax(ours_moving_pos_window)),
                    ours_moving_pos_window.shape,
                )
                vendor_moving_neg_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_neg_window)),
                    vendor_moving_neg_window.shape,
                )
                ours_moving_neg_peak = np.unravel_index(
                    int(np.argmax(ours_moving_neg_window)),
                    ours_moving_neg_window.shape,
                )

                vendor_static_a_peaks.append((int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23)))
                ours_static_a_peaks.append((int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23)))
                vendor_static_b_peaks.append((int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43)))
                ours_static_b_peaks.append((int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43)))
                vendor_moving_pos_peaks.append((int(vendor_moving_pos_peak[0]), int(vendor_moving_pos_peak[1] + 8)))
                ours_moving_pos_peaks.append((int(ours_moving_pos_peak[0]), int(ours_moving_pos_peak[1] + 8)))
                vendor_moving_neg_peaks.append((int(vendor_moving_neg_peak[0] + 2), int(vendor_moving_neg_peak[1] + 63)))
                ours_moving_neg_peaks.append((int(ours_moving_neg_peak[0] + 2), int(ours_moving_neg_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_phase_deg = np.degrees(
                np.angle(vendor["baseband"][:, 0, phase_sample_index] / vendor_ref)
            )
            ours_phase_deg = np.degrees(
                np.angle(ours["baseband"][:, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peaks, [(0, 27)] * 4)
        self.assertEqual(ours_static_a_peaks, vendor_static_a_peaks)
        self.assertEqual(vendor_static_b_peaks, [(0, 47)] * 4)
        self.assertEqual(ours_static_b_peaks, vendor_static_b_peaks)
        self.assertEqual(vendor_moving_pos_peaks, [(1, 14)] * 4)
        self.assertEqual(ours_moving_pos_peaks, vendor_moving_pos_peaks)
        self.assertEqual(vendor_moving_neg_peaks, [(3, 67)] * 4)
        self.assertEqual(ours_moving_neg_peaks, vendor_moving_neg_peaks)
        self.assertTrue(np.allclose(vendor_phase_deg, ours_phase_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_static_and_mixed_sign_movers_mimo_multiframe_matches_vendor_four_target_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_mimo_offaxis_static_static_mixed_sign_multiframe.npz"
        )
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))

            vendor_static_a_peaks = []
            ours_static_a_peaks = []
            vendor_static_b_peaks = []
            ours_static_b_peaks = []
            vendor_moving_pos_peaks = []
            ours_moving_pos_peaks = []
            vendor_moving_neg_peaks = []
            ours_moving_neg_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_a_window = vendor_rd[0:2, 23:32]
                ours_static_a_window = ours_rd[0:2, 23:32]
                vendor_static_b_window = vendor_rd[0:2, 43:52]
                ours_static_b_window = ours_rd[0:2, 43:52]
                vendor_moving_pos_window = vendor_rd[0:4, 8:18]
                ours_moving_pos_window = ours_rd[0:4, 8:18]
                vendor_moving_neg_window = vendor_rd[2:4, 63:72]
                ours_moving_neg_window = ours_rd[2:4, 63:72]

                vendor_static_a_peak = np.unravel_index(
                    int(np.argmax(vendor_static_a_window)),
                    vendor_static_a_window.shape,
                )
                ours_static_a_peak = np.unravel_index(
                    int(np.argmax(ours_static_a_window)),
                    ours_static_a_window.shape,
                )
                vendor_static_b_peak = np.unravel_index(
                    int(np.argmax(vendor_static_b_window)),
                    vendor_static_b_window.shape,
                )
                ours_static_b_peak = np.unravel_index(
                    int(np.argmax(ours_static_b_window)),
                    ours_static_b_window.shape,
                )
                vendor_moving_pos_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_pos_window)),
                    vendor_moving_pos_window.shape,
                )
                ours_moving_pos_peak = np.unravel_index(
                    int(np.argmax(ours_moving_pos_window)),
                    ours_moving_pos_window.shape,
                )
                vendor_moving_neg_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_neg_window)),
                    vendor_moving_neg_window.shape,
                )
                ours_moving_neg_peak = np.unravel_index(
                    int(np.argmax(ours_moving_neg_window)),
                    ours_moving_neg_window.shape,
                )

                vendor_static_a_peaks.append((int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23)))
                ours_static_a_peaks.append((int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23)))
                vendor_static_b_peaks.append((int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43)))
                ours_static_b_peaks.append((int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43)))
                vendor_moving_pos_peaks.append((int(vendor_moving_pos_peak[0]), int(vendor_moving_pos_peak[1] + 8)))
                ours_moving_pos_peaks.append((int(ours_moving_pos_peak[0]), int(ours_moving_pos_peak[1] + 8)))
                vendor_moving_neg_peaks.append((int(vendor_moving_neg_peak[0] + 2), int(vendor_moving_neg_peak[1] + 63)))
                ours_moving_neg_peaks.append((int(ours_moving_neg_peak[0] + 2), int(ours_moving_neg_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_phase_deg = np.degrees(
                np.angle(vendor["baseband"][:4, 0, phase_sample_index] / vendor_ref)
            )
            ours_phase_deg = np.degrees(
                np.angle(ours["baseband"][:4, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(ours_static_a_peaks, vendor_static_a_peaks)
        self.assertEqual(vendor_static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(ours_static_b_peaks, vendor_static_b_peaks)
        self.assertEqual(vendor_moving_pos_peaks, [(1, 14)] * 8)
        self.assertEqual(ours_moving_pos_peaks, vendor_moving_pos_peaks)
        self.assertEqual(vendor_moving_neg_peaks, [(3, 67)] * 8)
        self.assertEqual(ours_moving_neg_peaks, vendor_moving_neg_peaks)
        self.assertTrue(np.allclose(vendor_phase_deg, ours_phase_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_static_and_negative_moving_negative_moving_mimo_multiframe_matches_vendor_four_target_accepted_band_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_mimo_offaxis_static_static_moving_moving_negative_multiframe.npz"
        )
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [-8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))

            vendor_static_a_peaks = []
            ours_static_a_peaks = []
            vendor_static_b_peaks = []
            ours_static_b_peaks = []
            vendor_moving_a_peaks = []
            ours_moving_a_peaks = []
            vendor_moving_b_peaks = []
            ours_moving_b_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_a_window = vendor_rd[0:2, 23:32]
                ours_static_a_window = ours_rd[0:2, 23:32]
                vendor_static_b_window = vendor_rd[0:2, 43:52]
                ours_static_b_window = ours_rd[0:2, 43:52]
                vendor_moving_a_window = vendor_rd[2:4, 8:18]
                ours_moving_a_window = ours_rd[2:4, 8:18]
                vendor_moving_b_window = vendor_rd[2:4, 63:72]
                ours_moving_b_window = ours_rd[2:4, 63:72]

                vendor_static_a_peak = np.unravel_index(
                    int(np.argmax(vendor_static_a_window)),
                    vendor_static_a_window.shape,
                )
                ours_static_a_peak = np.unravel_index(
                    int(np.argmax(ours_static_a_window)),
                    ours_static_a_window.shape,
                )
                vendor_static_b_peak = np.unravel_index(
                    int(np.argmax(vendor_static_b_window)),
                    vendor_static_b_window.shape,
                )
                ours_static_b_peak = np.unravel_index(
                    int(np.argmax(ours_static_b_window)),
                    ours_static_b_window.shape,
                )
                vendor_moving_a_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_a_window)),
                    vendor_moving_a_window.shape,
                )
                ours_moving_a_peak = np.unravel_index(
                    int(np.argmax(ours_moving_a_window)),
                    ours_moving_a_window.shape,
                )
                vendor_moving_b_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_b_window)),
                    vendor_moving_b_window.shape,
                )
                ours_moving_b_peak = np.unravel_index(
                    int(np.argmax(ours_moving_b_window)),
                    ours_moving_b_window.shape,
                )

                vendor_static_a_peaks.append((int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23)))
                ours_static_a_peaks.append((int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23)))
                vendor_static_b_peaks.append((int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43)))
                ours_static_b_peaks.append((int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43)))
                vendor_moving_a_peaks.append((int(vendor_moving_a_peak[0] + 2), int(vendor_moving_a_peak[1] + 8)))
                ours_moving_a_peaks.append((int(ours_moving_a_peak[0] + 2), int(ours_moving_a_peak[1] + 8)))
                vendor_moving_b_peaks.append((int(vendor_moving_b_peak[0] + 2), int(vendor_moving_b_peak[1] + 63)))
                ours_moving_b_peaks.append((int(ours_moving_b_peak[0] + 2), int(ours_moving_b_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_phase_deg = np.degrees(
                np.angle(vendor["baseband"][:4, 0, phase_sample_index] / vendor_ref)
            )
            ours_phase_deg = np.degrees(
                np.angle(ours["baseband"][:4, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(ours_static_a_peaks, vendor_static_a_peaks)
        self.assertEqual(vendor_static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(ours_static_b_peaks, vendor_static_b_peaks)
        self.assertEqual(vendor_moving_a_peaks, [(3, 13)] * 8)
        self.assertEqual(ours_moving_a_peaks, vendor_moving_a_peaks)
        self.assertEqual(vendor_moving_b_peaks, [(3, 67)] * 8)
        self.assertTrue(all(peak in {(3, 66), (3, 67)} for peak in ours_moving_b_peaks))
        self.assertTrue(np.allclose(vendor_phase_deg, ours_phase_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_static_and_moving_moving_mimo_multiframe_matches_vendor_four_target_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_mimo_offaxis_static_static_moving_moving_multiframe.npz"
        )
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))

            vendor_static_a_peaks = []
            ours_static_a_peaks = []
            vendor_static_b_peaks = []
            ours_static_b_peaks = []
            vendor_moving_a_peaks = []
            ours_moving_a_peaks = []
            vendor_moving_b_peaks = []
            ours_moving_b_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_a_window = vendor_rd[0:2, 23:32]
                ours_static_a_window = ours_rd[0:2, 23:32]
                vendor_static_b_window = vendor_rd[0:2, 43:52]
                ours_static_b_window = ours_rd[0:2, 43:52]
                vendor_moving_a_window = vendor_rd[0:4, 8:18]
                ours_moving_a_window = ours_rd[0:4, 8:18]
                vendor_moving_b_window = vendor_rd[0:4, 63:72]
                ours_moving_b_window = ours_rd[0:4, 63:72]

                vendor_static_a_peak = np.unravel_index(
                    int(np.argmax(vendor_static_a_window)),
                    vendor_static_a_window.shape,
                )
                ours_static_a_peak = np.unravel_index(
                    int(np.argmax(ours_static_a_window)),
                    ours_static_a_window.shape,
                )
                vendor_static_b_peak = np.unravel_index(
                    int(np.argmax(vendor_static_b_window)),
                    vendor_static_b_window.shape,
                )
                ours_static_b_peak = np.unravel_index(
                    int(np.argmax(ours_static_b_window)),
                    ours_static_b_window.shape,
                )
                vendor_moving_a_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_a_window)),
                    vendor_moving_a_window.shape,
                )
                ours_moving_a_peak = np.unravel_index(
                    int(np.argmax(ours_moving_a_window)),
                    ours_moving_a_window.shape,
                )
                vendor_moving_b_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_b_window)),
                    vendor_moving_b_window.shape,
                )
                ours_moving_b_peak = np.unravel_index(
                    int(np.argmax(ours_moving_b_window)),
                    ours_moving_b_window.shape,
                )

                vendor_static_a_peaks.append((int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23)))
                ours_static_a_peaks.append((int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23)))
                vendor_static_b_peaks.append((int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43)))
                ours_static_b_peaks.append((int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43)))
                vendor_moving_a_peaks.append((int(vendor_moving_a_peak[0]), int(vendor_moving_a_peak[1] + 8)))
                ours_moving_a_peaks.append((int(ours_moving_a_peak[0]), int(ours_moving_a_peak[1] + 8)))
                vendor_moving_b_peaks.append((int(vendor_moving_b_peak[0]), int(vendor_moving_b_peak[1] + 63)))
                ours_moving_b_peaks.append((int(ours_moving_b_peak[0]), int(ours_moving_b_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_phase_deg = np.degrees(
                np.angle(vendor["baseband"][:4, 0, phase_sample_index] / vendor_ref)
            )
            ours_phase_deg = np.degrees(
                np.angle(ours["baseband"][:4, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(ours_static_a_peaks, vendor_static_a_peaks)
        self.assertEqual(vendor_static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(ours_static_b_peaks, vendor_static_b_peaks)
        self.assertEqual(vendor_moving_a_peaks, [(1, 14)] * 8)
        self.assertEqual(ours_moving_a_peaks, vendor_moving_a_peaks)
        self.assertEqual(vendor_moving_b_peaks, [(1, 67)] * 8)
        self.assertEqual(ours_moving_b_peaks, vendor_moving_b_peaks)
        self.assertTrue(np.allclose(vendor_phase_deg, ours_phase_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_and_moving_mimo_matches_vendor_mixed_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving.npz"
        )
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_static_peaks = []
            ours_static_peaks = []
            vendor_moving_peaks = []
            ours_moving_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_window = vendor_rd[0:2, 23:32]
                ours_static_window = ours_rd[0:2, 23:32]
                vendor_moving_window = vendor_rd[0:4, 63:72]
                ours_moving_window = ours_rd[0:4, 63:72]
                vendor_static_peak = np.unravel_index(
                    int(np.argmax(vendor_static_window)),
                    vendor_static_window.shape,
                )
                ours_static_peak = np.unravel_index(
                    int(np.argmax(ours_static_window)),
                    ours_static_window.shape,
                )
                vendor_moving_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_window)),
                    vendor_moving_window.shape,
                )
                ours_moving_peak = np.unravel_index(
                    int(np.argmax(ours_moving_window)),
                    ours_moving_window.shape,
                )
                vendor_static_peaks.append((int(vendor_static_peak[0]), int(vendor_static_peak[1] + 23)))
                ours_static_peaks.append((int(ours_static_peak[0]), int(ours_static_peak[1] + 23)))
                vendor_moving_peaks.append((int(vendor_moving_peak[0]), int(vendor_moving_peak[1] + 63)))
                ours_moving_peaks.append((int(ours_moving_peak[0]), int(ours_moving_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:, 0, phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_peaks, [(0, 27)] * 4)
        self.assertEqual(ours_static_peaks, vendor_static_peaks)
        self.assertEqual(vendor_moving_peaks, [(1, 67)] * 4)
        self.assertEqual(ours_moving_peaks, vendor_moving_peaks)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_and_negative_moving_mimo_matches_vendor_mixed_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving_negative.npz"
        )
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_static_peaks = []
            ours_static_peaks = []
            vendor_moving_peaks = []
            ours_moving_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_window = vendor_rd[0:2, 23:32]
                ours_static_window = ours_rd[0:2, 23:32]
                vendor_moving_window = vendor_rd[2:4, 63:72]
                ours_moving_window = ours_rd[2:4, 63:72]
                vendor_static_peak = np.unravel_index(
                    int(np.argmax(vendor_static_window)),
                    vendor_static_window.shape,
                )
                ours_static_peak = np.unravel_index(
                    int(np.argmax(ours_static_window)),
                    ours_static_window.shape,
                )
                vendor_moving_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_window)),
                    vendor_moving_window.shape,
                )
                ours_moving_peak = np.unravel_index(
                    int(np.argmax(ours_moving_window)),
                    ours_moving_window.shape,
                )
                vendor_static_peaks.append((int(vendor_static_peak[0]), int(vendor_static_peak[1] + 23)))
                ours_static_peaks.append((int(ours_static_peak[0]), int(ours_static_peak[1] + 23)))
                vendor_moving_peaks.append((int(vendor_moving_peak[0] + 2), int(vendor_moving_peak[1] + 63)))
                ours_moving_peaks.append((int(ours_moving_peak[0] + 2), int(ours_moving_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:, 0, phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_peaks, [(0, 27)] * 4)
        self.assertEqual(ours_static_peaks, vendor_static_peaks)
        self.assertEqual(vendor_moving_peaks, [(3, 67)] * 4)
        self.assertEqual(ours_moving_peaks, vendor_moving_peaks)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_static_and_moving_mimo_matches_vendor_three_target_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving.npz"
        )
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_static_a_peaks = []
            ours_static_a_peaks = []
            vendor_static_b_peaks = []
            ours_static_b_peaks = []
            vendor_moving_peaks = []
            ours_moving_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_a_window = vendor_rd[0:2, 23:32]
                ours_static_a_window = ours_rd[0:2, 23:32]
                vendor_static_b_window = vendor_rd[0:2, 43:52]
                ours_static_b_window = ours_rd[0:2, 43:52]
                vendor_moving_window = vendor_rd[0:4, 63:72]
                ours_moving_window = ours_rd[0:4, 63:72]
                vendor_static_a_peak = np.unravel_index(
                    int(np.argmax(vendor_static_a_window)),
                    vendor_static_a_window.shape,
                )
                ours_static_a_peak = np.unravel_index(
                    int(np.argmax(ours_static_a_window)),
                    ours_static_a_window.shape,
                )
                vendor_static_b_peak = np.unravel_index(
                    int(np.argmax(vendor_static_b_window)),
                    vendor_static_b_window.shape,
                )
                ours_static_b_peak = np.unravel_index(
                    int(np.argmax(ours_static_b_window)),
                    ours_static_b_window.shape,
                )
                vendor_moving_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_window)),
                    vendor_moving_window.shape,
                )
                ours_moving_peak = np.unravel_index(
                    int(np.argmax(ours_moving_window)),
                    ours_moving_window.shape,
                )
                vendor_static_a_peaks.append((int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23)))
                ours_static_a_peaks.append((int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23)))
                vendor_static_b_peaks.append((int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43)))
                ours_static_b_peaks.append((int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43)))
                vendor_moving_peaks.append((int(vendor_moving_peak[0]), int(vendor_moving_peak[1] + 63)))
                ours_moving_peaks.append((int(ours_moving_peak[0]), int(ours_moving_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:, 0, phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peaks, [(0, 27)] * 4)
        self.assertEqual(ours_static_a_peaks, vendor_static_a_peaks)
        self.assertEqual(vendor_static_b_peaks, [(0, 47)] * 4)
        self.assertEqual(ours_static_b_peaks, vendor_static_b_peaks)
        self.assertEqual(vendor_moving_peaks, [(1, 67)] * 4)
        self.assertEqual(ours_moving_peaks, vendor_moving_peaks)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_static_and_negative_moving_mimo_matches_vendor_three_target_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_mimo_offaxis_static_static_moving_negative.npz"
        )
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))

            vendor_static_a_peaks = []
            ours_static_a_peaks = []
            vendor_static_b_peaks = []
            ours_static_b_peaks = []
            vendor_moving_peaks = []
            ours_moving_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_a_window = vendor_rd[0:2, 23:32]
                ours_static_a_window = ours_rd[0:2, 23:32]
                vendor_static_b_window = vendor_rd[0:2, 43:52]
                ours_static_b_window = ours_rd[0:2, 43:52]
                vendor_moving_window = vendor_rd[2:4, 63:72]
                ours_moving_window = ours_rd[2:4, 63:72]
                vendor_static_a_peak = np.unravel_index(
                    int(np.argmax(vendor_static_a_window)),
                    vendor_static_a_window.shape,
                )
                ours_static_a_peak = np.unravel_index(
                    int(np.argmax(ours_static_a_window)),
                    ours_static_a_window.shape,
                )
                vendor_static_b_peak = np.unravel_index(
                    int(np.argmax(vendor_static_b_window)),
                    vendor_static_b_window.shape,
                )
                ours_static_b_peak = np.unravel_index(
                    int(np.argmax(ours_static_b_window)),
                    ours_static_b_window.shape,
                )
                vendor_moving_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_window)),
                    vendor_moving_window.shape,
                )
                ours_moving_peak = np.unravel_index(
                    int(np.argmax(ours_moving_window)),
                    ours_moving_window.shape,
                )
                vendor_static_a_peaks.append((int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23)))
                ours_static_a_peaks.append((int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23)))
                vendor_static_b_peaks.append((int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43)))
                ours_static_b_peaks.append((int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43)))
                vendor_moving_peaks.append((int(vendor_moving_peak[0] + 2), int(vendor_moving_peak[1] + 63)))
                ours_moving_peaks.append((int(ours_moving_peak[0] + 2), int(ours_moving_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:, 0, phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peaks, [(0, 27)] * 4)
        self.assertEqual(ours_static_a_peaks, vendor_static_a_peaks)
        self.assertEqual(vendor_static_b_peaks, [(0, 47)] * 4)
        self.assertEqual(ours_static_b_peaks, vendor_static_b_peaks)
        self.assertEqual(vendor_moving_peaks, [(3, 67)] * 4)
        self.assertEqual(ours_moving_peaks, vendor_moving_peaks)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_and_moving_mimo_multiframe_matches_vendor_mixed_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving_multiframe.npz"
        )
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))

            vendor_static_peaks = []
            ours_static_peaks = []
            vendor_moving_peaks = []
            ours_moving_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_window = vendor_rd[0:2, 23:32]
                ours_static_window = ours_rd[0:2, 23:32]
                vendor_moving_window = vendor_rd[0:4, 63:72]
                ours_moving_window = ours_rd[0:4, 63:72]
                vendor_static_peak = np.unravel_index(
                    int(np.argmax(vendor_static_window)),
                    vendor_static_window.shape,
                )
                ours_static_peak = np.unravel_index(
                    int(np.argmax(ours_static_window)),
                    ours_static_window.shape,
                )
                vendor_moving_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_window)),
                    vendor_moving_window.shape,
                )
                ours_moving_peak = np.unravel_index(
                    int(np.argmax(ours_moving_window)),
                    ours_moving_window.shape,
                )
                vendor_static_peaks.append((int(vendor_static_peak[0]), int(vendor_static_peak[1] + 23)))
                ours_static_peaks.append((int(ours_static_peak[0]), int(ours_static_peak[1] + 23)))
                vendor_moving_peaks.append((int(vendor_moving_peak[0]), int(vendor_moving_peak[1] + 63)))
                ours_moving_peaks.append((int(ours_moving_peak[0]), int(ours_moving_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:4, 0, phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:4, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_peaks, [(0, 27)] * 8)
        self.assertEqual(ours_static_peaks, vendor_static_peaks)
        self.assertEqual(vendor_moving_peaks, [(1, 67)] * 8)
        self.assertEqual(ours_moving_peaks, vendor_moving_peaks)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_static_and_moving_mimo_multiframe_matches_vendor_three_target_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_mimo_offaxis_static_static_moving_multiframe.npz"
        )
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))

            vendor_static_a_peaks = []
            ours_static_a_peaks = []
            vendor_static_b_peaks = []
            ours_static_b_peaks = []
            vendor_moving_peaks = []
            ours_moving_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_a_window = vendor_rd[0:2, 23:32]
                ours_static_a_window = ours_rd[0:2, 23:32]
                vendor_static_b_window = vendor_rd[0:2, 43:52]
                ours_static_b_window = ours_rd[0:2, 43:52]
                vendor_moving_window = vendor_rd[0:4, 63:72]
                ours_moving_window = ours_rd[0:4, 63:72]
                vendor_static_a_peak = np.unravel_index(
                    int(np.argmax(vendor_static_a_window)),
                    vendor_static_a_window.shape,
                )
                ours_static_a_peak = np.unravel_index(
                    int(np.argmax(ours_static_a_window)),
                    ours_static_a_window.shape,
                )
                vendor_static_b_peak = np.unravel_index(
                    int(np.argmax(vendor_static_b_window)),
                    vendor_static_b_window.shape,
                )
                ours_static_b_peak = np.unravel_index(
                    int(np.argmax(ours_static_b_window)),
                    ours_static_b_window.shape,
                )
                vendor_moving_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_window)),
                    vendor_moving_window.shape,
                )
                ours_moving_peak = np.unravel_index(
                    int(np.argmax(ours_moving_window)),
                    ours_moving_window.shape,
                )
                vendor_static_a_peaks.append((int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23)))
                ours_static_a_peaks.append((int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23)))
                vendor_static_b_peaks.append((int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43)))
                ours_static_b_peaks.append((int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43)))
                vendor_moving_peaks.append((int(vendor_moving_peak[0]), int(vendor_moving_peak[1] + 63)))
                ours_moving_peaks.append((int(ours_moving_peak[0]), int(ours_moving_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:4, 0, phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:4, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(ours_static_a_peaks, vendor_static_a_peaks)
        self.assertEqual(vendor_static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(ours_static_b_peaks, vendor_static_b_peaks)
        self.assertEqual(vendor_moving_peaks, [(1, 67)] * 8)
        self.assertEqual(ours_moving_peaks, vendor_moving_peaks)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_static_and_negative_moving_mimo_multiframe_matches_vendor_three_target_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_mimo_offaxis_static_static_moving_negative_multiframe.npz"
        )
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))

            vendor_static_a_peaks = []
            ours_static_a_peaks = []
            vendor_static_b_peaks = []
            ours_static_b_peaks = []
            vendor_moving_peaks = []
            ours_moving_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_a_window = vendor_rd[0:2, 23:32]
                ours_static_a_window = ours_rd[0:2, 23:32]
                vendor_static_b_window = vendor_rd[0:2, 43:52]
                ours_static_b_window = ours_rd[0:2, 43:52]
                vendor_moving_window = vendor_rd[2:4, 63:72]
                ours_moving_window = ours_rd[2:4, 63:72]
                vendor_static_a_peak = np.unravel_index(
                    int(np.argmax(vendor_static_a_window)),
                    vendor_static_a_window.shape,
                )
                ours_static_a_peak = np.unravel_index(
                    int(np.argmax(ours_static_a_window)),
                    ours_static_a_window.shape,
                )
                vendor_static_b_peak = np.unravel_index(
                    int(np.argmax(vendor_static_b_window)),
                    vendor_static_b_window.shape,
                )
                ours_static_b_peak = np.unravel_index(
                    int(np.argmax(ours_static_b_window)),
                    ours_static_b_window.shape,
                )
                vendor_moving_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_window)),
                    vendor_moving_window.shape,
                )
                ours_moving_peak = np.unravel_index(
                    int(np.argmax(ours_moving_window)),
                    ours_moving_window.shape,
                )
                vendor_static_a_peaks.append((int(vendor_static_a_peak[0]), int(vendor_static_a_peak[1] + 23)))
                ours_static_a_peaks.append((int(ours_static_a_peak[0]), int(ours_static_a_peak[1] + 23)))
                vendor_static_b_peaks.append((int(vendor_static_b_peak[0]), int(vendor_static_b_peak[1] + 43)))
                ours_static_b_peaks.append((int(ours_static_b_peak[0]), int(ours_static_b_peak[1] + 43)))
                vendor_moving_peaks.append((int(vendor_moving_peak[0] + 2), int(vendor_moving_peak[1] + 63)))
                ours_moving_peaks.append((int(ours_moving_peak[0] + 2), int(ours_moving_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:4, 0, phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:4, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(ours_static_a_peaks, vendor_static_a_peaks)
        self.assertEqual(vendor_static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(ours_static_b_peaks, vendor_static_b_peaks)
        self.assertEqual(vendor_moving_peaks, [(3, 67)] * 8)
        self.assertEqual(ours_moving_peaks, vendor_moving_peaks)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_offaxis_static_and_negative_moving_mimo_multiframe_matches_vendor_mixed_joint_contract(self):
        vendor_path = (
            VENDOR_ORACLE_DIR
            / "artifacts"
            / "sim_radar_point_mimo_offaxis_static_moving_negative_multiframe.npz"
        )
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        ours = sim_radar(radar, targets)

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))

            vendor_static_peaks = []
            ours_static_peaks = []
            vendor_moving_peaks = []
            ours_moving_peaks = []
            for channel_index in range(vendor["baseband"].shape[0]):
                vendor_rd = np.abs(
                    processing.range_doppler_fft(vendor["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                ours_rd = np.abs(
                    processing.range_doppler_fft(ours["baseband"][channel_index : channel_index + 1])
                )[0, :, :80]
                vendor_static_window = vendor_rd[0:2, 23:32]
                ours_static_window = ours_rd[0:2, 23:32]
                vendor_moving_window = vendor_rd[2:4, 63:72]
                ours_moving_window = ours_rd[2:4, 63:72]
                vendor_static_peak = np.unravel_index(
                    int(np.argmax(vendor_static_window)),
                    vendor_static_window.shape,
                )
                ours_static_peak = np.unravel_index(
                    int(np.argmax(ours_static_window)),
                    ours_static_window.shape,
                )
                vendor_moving_peak = np.unravel_index(
                    int(np.argmax(vendor_moving_window)),
                    vendor_moving_window.shape,
                )
                ours_moving_peak = np.unravel_index(
                    int(np.argmax(ours_moving_window)),
                    ours_moving_window.shape,
                )
                vendor_static_peaks.append((int(vendor_static_peak[0]), int(vendor_static_peak[1] + 23)))
                ours_static_peaks.append((int(ours_static_peak[0]), int(ours_static_peak[1] + 23)))
                vendor_moving_peaks.append((int(vendor_moving_peak[0] + 2), int(vendor_moving_peak[1] + 63)))
                ours_moving_peaks.append((int(ours_moving_peak[0] + 2), int(ours_moving_peak[1] + 63)))

            phase_sample_index = 2
            vendor_ref = vendor["baseband"][0, 0, phase_sample_index]
            ours_ref = ours["baseband"][0, 0, phase_sample_index]
            vendor_rel_deg = np.degrees(
                np.angle(vendor["baseband"][:4, 0, phase_sample_index] / vendor_ref)
            )
            ours_rel_deg = np.degrees(
                np.angle(ours["baseband"][:4, 0, phase_sample_index] / ours_ref)
            )

            vendor_profile = np.abs(processing.range_fft(vendor["baseband"])[0, 0, :80])
            ours_profile = np.abs(processing.range_fft(ours["baseband"])[0, 0, :80])
            profile_correlation = np.vdot(vendor_profile, ours_profile)
            profile_correlation /= np.linalg.norm(vendor_profile)
            profile_correlation /= np.linalg.norm(ours_profile)
            waveform_correlation = np.vdot(vendor["baseband"].ravel(), ours["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(vendor["baseband"].ravel())
            waveform_correlation /= np.linalg.norm(ours["baseband"].ravel())

        self.assertEqual(vendor_static_peaks, [(0, 27)] * 8)
        self.assertEqual(ours_static_peaks, vendor_static_peaks)
        self.assertEqual(vendor_moving_peaks, [(3, 67)] * 8)
        self.assertEqual(ours_moving_peaks, vendor_moving_peaks)
        self.assertTrue(np.allclose(vendor_rel_deg, ours_rel_deg, atol=1e-6))
        self.assertGreaterEqual(float(np.abs(profile_correlation)), 0.999)
        self.assertGreaterEqual(float(np.abs(waveform_correlation)), 0.995)

    def test_sim_radar_mimo_multiframe_matches_vendor_oracle_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_mimo_multiframe.npz"
        radar = self.build_mimo_multiframe_radar(seed=2026)
        target = {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [1500.0, 0.0, 0.0]}

        ours = sim_radar(radar, [target])

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))
            self.assertTrue(np.array_equal(vendor["timestamp"][:4] + 1e-3, vendor["timestamp"][4:]))
            self.assertTrue(np.array_equal(ours["timestamp"][:4] + 1e-3, ours["timestamp"][4:]))
            self.assertIsNone(ours["interference"])

    def test_sim_radar_point_phase_matches_vendor_quadrature_contract(self):
        vendor_phase_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_phase.npz"
        vendor_base_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_target.npz"
        baseline = sim_radar(
            self.build_radar(seed=2026),
            [{"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}],
        )
        phase_shifted = sim_radar(
            self.build_radar(seed=2026),
            [{"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0], "phase": 90.0}],
        )

        self.assertIsNone(phase_shifted["interference"])
        with np.load(vendor_base_path) as vendor_base, np.load(vendor_phase_path) as vendor_phase:
            self.assertEqual(vendor_phase["timestamp"].shape, phase_shifted["timestamp"].shape)
            self.assertEqual(vendor_phase["baseband"].shape, phase_shifted["baseband"].shape)
            self.assertTrue(np.array_equal(vendor_phase["timestamp"], phase_shifted["timestamp"]))
            self.assertTrue(np.allclose(vendor_phase["baseband"], phase_shifted["baseband"], atol=1e-6))

            vendor_valid = np.abs(vendor_base["baseband"]) > np.finfo(float).eps
            vendor_ratios = vendor_phase["baseband"][vendor_valid] / vendor_base["baseband"][vendor_valid]
            ours_valid = np.abs(baseline["baseband"]) > np.finfo(float).eps
            ours_ratios = phase_shifted["baseband"][ours_valid] / baseline["baseband"][ours_valid]

        self.assertGreater(vendor_ratios.size, 0)
        self.assertGreater(ours_ratios.size, 0)
        self.assertLessEqual(float(np.max(np.abs(vendor_ratios - 1j))), 1e-6)
        self.assertLessEqual(float(np.max(np.abs(ours_ratios - 1j))), 1e-6)

    def test_sim_radar_moving_point_target_matches_vendor_doppler_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_point_moving_doppler.npz"
        ours = sim_radar(
            self.build_radar(seed=2026),
            [{"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]}],
        )

        with np.load(vendor_path) as vendor:
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor["baseband"].shape, ours["baseband"].shape)
            self.assertEqual(vendor["noise"].shape, ours["noise"].shape)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertTrue(np.allclose(vendor["baseband"], ours["baseband"], atol=1e-6))

            vendor_rd = np.abs(processing.range_doppler_fft(vendor["baseband"]))[0, :, :80]
            ours_rd = np.abs(processing.range_doppler_fft(ours["baseband"]))[0, :, :80]
            vendor_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(vendor_rd)), vendor_rd.shape))
            ours_peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(ours_rd)), ours_rd.shape))

        self.assertEqual(vendor_peak, (1, 67))
        self.assertEqual(ours_peak, vendor_peak)
        self.assertIsNone(ours["interference"])

    def test_sim_radar_seed_behavior_matches_vendor_oracle_summary(self):
        radar_a = self.build_radar(seed=2026)
        radar_b = self.build_radar(seed=2026)
        target = {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}

        run_a = sim_radar(radar_a, [target])
        run_b = sim_radar(radar_b, [target])

        self.assertTrue(np.allclose(run_a["baseband"], run_b["baseband"]))
        self.assertFalse(np.allclose(run_a["noise"], run_b["noise"]))

    def test_sim_lidar_matches_vendor_shared_fields(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_lidar_hits.npy"
        mesh_path = self.write_temp_stl()

        ours = sim_lidar(
            self.build_lidar(),
            [{"model": mesh_path, "location": [10.0, 0.0, 0.0]}],
        )
        vendor = np.load(vendor_path, allow_pickle=False)

        self.assertEqual(len(vendor), len(ours))
        self.assertIn("positions", ours.dtype.names)
        self.assertIn("directions", ours.dtype.names)
        self.assertTrue(np.allclose(vendor["positions"], ours["positions"], atol=1e-6))
        self.assertTrue(np.allclose(vendor["directions"], ours["directions"], atol=1e-6))

    def test_sim_radar_mesh_target_matches_vendor_mesh_contract(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_mesh_target.npz"
        radar = self.build_radar(seed=2026)
        mesh_path = self.write_temp_stl()

        ours = sim_radar(
            radar,
            [{"model": mesh_path, "location": [50.0, 0.0, 0.0]}],
            density=1.0,
        )

        self.assertIsNone(ours["interference"])
        with np.load(vendor_path) as vendor:
            vendor_baseband = vendor["baseband"]
            self.assertEqual(vendor["timestamp"].shape, ours["timestamp"].shape)
            self.assertEqual(vendor_baseband.shape, ours["baseband"].shape)
            self.assertEqual(vendor_baseband.dtype, ours["baseband"].dtype)
            self.assertTrue(np.array_equal(vendor["timestamp"], ours["timestamp"]))
            self.assertGreater(float(np.max(np.abs(vendor_baseband))), 0.0)
            self.assertGreater(float(np.max(np.abs(ours["baseband"]))), 0.0)
            correlation = np.vdot(vendor_baseband.reshape(-1), ours["baseband"].reshape(-1))
            correlation /= np.linalg.norm(vendor_baseband.reshape(-1))
            correlation /= np.linalg.norm(ours["baseband"].reshape(-1))
            self.assertGreaterEqual(float(np.abs(correlation)), 0.99)

    def test_sim_radar_mesh_aspect_trend_matches_vendor_behavior(self):
        broadside_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_mesh_broadside.npz"
        edge_on_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_radar_mesh_edge_on.npz"
        radar = self.build_radar(seed=2026)
        mesh_path = self.write_temp_stl()

        broadside = sim_radar(
            radar,
            [{"model": mesh_path, "location": [50.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}],
            density=1.0,
        )
        edge_on = sim_radar(
            radar,
            [{"model": mesh_path, "location": [50.0, 0.0, 0.0], "rotation": [0.0, 90.0, 0.0]}],
            density=1.0,
        )
        with np.load(broadside_path) as vendor_broadside, np.load(edge_on_path) as vendor_edge_on:
            vendor_broadside_max = float(np.max(np.abs(vendor_broadside["baseband"])))
            vendor_edge_on_max = float(np.max(np.abs(vendor_edge_on["baseband"])))
            ours_broadside_max = float(np.max(np.abs(broadside["baseband"])))
            ours_edge_on_max = float(np.max(np.abs(edge_on["baseband"])))

        self.assertGreater(vendor_broadside_max, vendor_edge_on_max)
        self.assertGreater(ours_broadside_max, ours_edge_on_max)

    def test_sim_lidar_moving_matches_vendor_shared_fields(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_lidar_moving_hits.npy"
        mesh_path = self.write_temp_stl()

        ours = sim_lidar(
            self.build_lidar(),
            [{"model": mesh_path, "location": [10.0, 0.0, 0.0], "speed": [2.0, 0.0, 0.0]}],
            frame_time=1.5,
        )
        vendor = np.load(vendor_path, allow_pickle=False)

        self.assertEqual(len(vendor), len(ours))
        self.assertTrue(np.allclose(vendor["positions"], ours["positions"], atol=1e-6))
        self.assertTrue(np.allclose(vendor["directions"], ours["directions"], atol=1e-6))

    def test_sim_lidar_multi_hit_matches_vendor_reflection_and_ordering(self):
        vendor_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_lidar_multi_hits.npy"
        mesh_path = self.write_temp_stl()

        ours = sim_lidar(
            self.build_multi_hit_lidar(),
            [
                {"model": mesh_path, "location": [10.0, -1.0, 0.0]},
                {"model": mesh_path, "location": [10.0, 0.0, 0.0]},
                {"model": mesh_path, "location": [10.0, 1.0, 0.0]},
            ],
            frame_time=0.0,
        )
        vendor = np.load(vendor_path, allow_pickle=False)

        self.assertEqual(len(vendor), len(ours))
        self.assertTrue(np.allclose(vendor["positions"], ours["positions"], atol=1e-6))
        self.assertTrue(np.allclose(vendor["directions"], ours["directions"], atol=1e-6))
        vendor_distances = np.linalg.norm(vendor["positions"], axis=1)
        self.assertTrue(np.allclose(vendor_distances, ours["distance"], atol=1e-5))
        self.assertTrue(np.all(np.diff(vendor["positions"][:, 1]) > 0.0))
        self.assertTrue(np.array_equal(ours["target_index"], [0, 1, 2]))

    def test_sim_rcs_matches_vendor_extended_angle_cases(self):
        mesh_path = self.write_temp_stl()
        vendor_cases = self.vendor_manifest["captures"]["sim_rcs"]
        case_specs = {
            "normal_incidence": {"f": 77e9, "inc_phi": 0.0, "inc_theta": 0.0},
            "broadside": {"f": 77e9, "inc_phi": 0.0, "inc_theta": 90.0},
            "edge_on": {"f": 77e9, "inc_phi": 90.0, "inc_theta": 90.0},
            "obs_backscatter": {
                "f": 77e9,
                "inc_phi": 0.0,
                "inc_theta": 90.0,
                "obs_phi": 0.0,
                "obs_theta": 90.0,
            },
            "obs_forwardscatter": {
                "f": 77e9,
                "inc_phi": 0.0,
                "inc_theta": 90.0,
                "obs_phi": 180.0,
                "obs_theta": 90.0,
            },
        }

        for case_name, kwargs in case_specs.items():
            with self.subTest(case=case_name):
                vendor_value = float(vendor_cases[case_name]["summary"])
                ours = float(sim_rcs([{"model": mesh_path}], **kwargs))
                self.assertTrue(np.isclose(vendor_value, ours, rtol=0.01, atol=1e-9))

    def test_sim_rcs_sweep_matches_vendor_main_lobe_shape_contract(self):
        sweep_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_rcs_obs_phi_sweep.npz"
        mesh_path = self.write_temp_stl()

        with np.load(sweep_path) as vendor:
            obs_phi = vendor["obs_phi"]
            vendor_rcs = vendor["rcs"]

        ours = sim_rcs(
            [{"model": mesh_path}],
            f=77e9,
            inc_phi=np.zeros_like(obs_phi),
            inc_theta=np.full_like(obs_phi, 90.0),
            obs_phi=obs_phi,
            obs_theta=np.full_like(obs_phi, 90.0),
        )

        self.assertEqual(vendor_rcs.shape, ours.shape)
        self.assertEqual(int(np.argmax(vendor_rcs)), int(np.argmax(ours)))
        self.assertTrue(np.all(np.diff(vendor_rcs[:3]) < 0.0))
        self.assertTrue(np.all(np.diff(ours[:3]) < 0.0))
        self.assertLessEqual(float(np.max(vendor_rcs[3:]) / np.max(vendor_rcs)), 1e-6)
        self.assertLessEqual(float(np.max(ours[3:]) / np.max(ours)), 1e-6)
        correlation = np.vdot(vendor_rcs, ours) / (np.linalg.norm(vendor_rcs) * np.linalg.norm(ours))
        self.assertGreaterEqual(float(np.abs(correlation)), 0.85)

    def test_sim_rcs_polarization_matches_vendor_discrimination_contract(self):
        pol_path = VENDOR_ORACLE_DIR / "artifacts" / "sim_rcs_polarization_cases.npz"
        mesh_path = self.write_temp_stl()

        with np.load(pol_path) as vendor:
            case_names = vendor["case_names"]
            vendor_values = vendor["rcs"]

        ours_values = np.asarray(
            [
                sim_rcs(
                    [{"model": mesh_path}],
                    f=77e9,
                    inc_phi=0.0,
                    inc_theta=90.0,
                    inc_pol=[0.0, 0.0, 1.0],
                    obs_phi=0.0,
                    obs_theta=90.0,
                    obs_pol=obs_pol,
                )
                for obs_pol in (
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                )
            ],
            dtype=float,
        )

        self.assertEqual(case_names.tolist(), ["co_pol", "cross_y", "cross_x", "cross_xy"])
        self.assertTrue(np.isclose(vendor_values[0], ours_values[0], rtol=0.01, atol=1e-9))
        self.assertGreater(vendor_values[0], np.max(vendor_values[1:]))
        self.assertGreater(ours_values[0], np.max(ours_values[1:]))
        self.assertLessEqual(float(np.max(vendor_values[1:]) / vendor_values[0]), 1e-20)
        self.assertLessEqual(float(np.max(ours_values[1:]) / ours_values[0]), 1e-20)
