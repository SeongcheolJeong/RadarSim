import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

from radarsimpy_whitebox import Radar, Receiver, Transmitter, processing, sim_radar


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


class Phase3PointTargetSimulatorTests(unittest.TestCase):
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

    def write_temp_stl(self) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        mesh_path = Path(tmpdir.name) / "plate.stl"
        mesh_path.write_text(PLATE_STL, encoding="utf-8")
        return str(mesh_path)

    def test_sim_radar_signature(self):
        signature = str(inspect.signature(sim_radar))
        self.assertEqual(
            signature,
            "(radar: 'Radar', targets: 'Iterable[Dict]', density: 'float' = 1, "
            "level=None, interf=None, ray_filter=None, back_propagating: 'bool' = False, "
            "device: 'str' = 'gpu', log_path=None, dry_run: 'bool' = False)",
        )

    def test_sim_radar_returns_expected_contract(self):
        radar = self.build_radar(seed=123)
        target = {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}

        data = sim_radar(radar, [target])

        self.assertEqual(set(data), {"timestamp", "baseband", "noise", "interference"})
        self.assertEqual(data["timestamp"].shape, radar.time_prop["timestamp"].shape)
        self.assertEqual(data["baseband"].shape, radar.time_prop["timestamp"].shape)
        self.assertEqual(data["noise"].shape, radar.time_prop["timestamp"].shape)
        self.assertIsNone(data["interference"])
        self.assertTrue(np.array_equal(data["timestamp"], radar.time_prop["timestamp"]))

    def test_sim_radar_dry_run_returns_zero_arrays(self):
        radar = self.build_radar(seed=123)
        target = {"location": [20.0, 0.0, 0.0], "rcs": 0.0}

        data = sim_radar(radar, [target], dry_run=True)

        self.assertTrue(np.all(data["baseband"] == 0))
        self.assertTrue(np.all(data["noise"] == 0))
        self.assertIsNone(data["interference"])

    def test_sim_radar_seed_reproducibility(self):
        radar_a = self.build_radar(seed=2026)
        radar_b = self.build_radar(seed=2026)
        target = {"location": [30.0, 0.0, 0.0], "rcs": 5.0}

        data_a = sim_radar(radar_a, [target])
        data_b = sim_radar(radar_b, [target])

        self.assertTrue(np.allclose(data_a["baseband"], data_b["baseband"]))
        self.assertFalse(np.allclose(data_a["noise"], data_b["noise"]))

    def test_sim_radar_mimo_shape_tracks_tx_rx_channel_count(self):
        radar = self.build_mimo_radar(seed=2026)
        target = {"location": [50.0, 0.0, 0.0], "rcs": 10.0}

        data = sim_radar(radar, [target])

        self.assertEqual(data["baseband"].shape[0], 4)
        self.assertEqual(data["timestamp"].shape, (4, 4, 160))
        self.assertEqual(data["noise"].shape, (4, 4, 160))
        self.assertIsNone(data["interference"])

    def test_sim_radar_offaxis_mimo_target_creates_channel_phase_gradient(self):
        radar = self.build_mimo_radar(seed=2026)
        data = sim_radar(
            radar,
            [{"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}],
        )

        peak_sample_index = int(np.argmax(np.abs(data["baseband"][0, 0])))
        ref = data["baseband"][0, 0, peak_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:, 0, peak_sample_index] / ref)
        )

        self.assertTrue(np.all(np.abs(channel_phase_deg[1:]) > 10.0))
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_moving_mimo_combines_phase_gradient_and_doppler_shift(self):
        radar = self.build_mimo_radar(seed=2026)
        data = sim_radar(
            radar,
            [{"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]}],
        )

        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(range_doppler)), range_doppler.shape))
        valid = np.all(np.abs(data["baseband"][:, 0, :]) > 1e-12, axis=0)
        phase_sample_index = int(np.argmax(valid))
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:, 0, phase_sample_index] / ref)
        )

        self.assertEqual(peak, (1, 67))
        self.assertEqual(phase_sample_index, 2)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_negative_speed_mimo_combines_phase_gradient_and_reversed_doppler(self):
        radar = self.build_mimo_radar(seed=2026)
        data = sim_radar(
            radar,
            [{"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]}],
        )

        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(range_doppler)), range_doppler.shape))
        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:, 0, phase_sample_index] / ref)
        )

        self.assertEqual(peak, (3, 67))
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_moving_mimo_multiframe_preserves_joint_contract(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        data = sim_radar(
            radar,
            [{"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]}],
        )

        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(range_doppler)), range_doppler.shape))
        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:4, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertEqual(peak, (1, 67))
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_negative_speed_mimo_multiframe_preserves_joint_contract(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        data = sim_radar(
            radar,
            [{"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]}],
        )

        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(range_doppler)), range_doppler.shape))
        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:4, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertEqual(peak, (3, 67))
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_mimo_multiframe_flattens_frames_after_channel_blocks(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        target = {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [1500.0, 0.0, 0.0]}

        data = sim_radar(radar, [target])

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertEqual(data["timestamp"].shape, (8, 4, 160))
        self.assertEqual(data["noise"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertIsNone(data["interference"])

    def test_sim_radar_point_phase_rotates_baseband_by_requested_degrees(self):
        radar = self.build_radar(seed=2026)
        baseline = sim_radar(radar, [{"location": [50.0, 0.0, 0.0], "rcs": 10.0}])
        phase_shifted = sim_radar(
            self.build_radar(seed=2026),
            [{"location": [50.0, 0.0, 0.0], "rcs": 10.0, "phase": 90.0}],
        )

        valid = np.abs(baseline["baseband"]) > np.finfo(float).eps
        ratios = phase_shifted["baseband"][valid] / baseline["baseband"][valid]

        self.assertGreater(ratios.size, 0)
        self.assertLessEqual(float(np.max(np.abs(ratios - 1j))), 1e-12)

    def test_sim_radar_moving_point_target_shifts_doppler_peak(self):
        static = sim_radar(
            self.build_radar(seed=2026),
            [{"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}],
        )
        moving = sim_radar(
            self.build_radar(seed=2026),
            [{"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]}],
        )

        static_rd = np.abs(processing.range_doppler_fft(static["baseband"]))[0, :, :80]
        moving_rd = np.abs(processing.range_doppler_fft(moving["baseband"]))[0, :, :80]
        static_peak = np.unravel_index(int(np.argmax(static_rd)), static_rd.shape)
        moving_peak = np.unravel_index(int(np.argmax(moving_rd)), moving_rd.shape)

        self.assertEqual(tuple(int(x) for x in static_peak), (0, 67))
        self.assertEqual(tuple(int(x) for x in moving_peak), (1, 67))

    def test_sim_radar_negative_radial_speed_flips_doppler_direction(self):
        forward = sim_radar(
            self.build_radar(seed=2026),
            [{"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]}],
        )
        reverse = sim_radar(
            self.build_radar(seed=2026),
            [{"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]}],
        )

        forward_rd = np.abs(processing.range_doppler_fft(forward["baseband"]))[0, :, :80]
        reverse_rd = np.abs(processing.range_doppler_fft(reverse["baseband"]))[0, :, :80]
        forward_peak = tuple(
            int(x) for x in np.unravel_index(int(np.argmax(forward_rd)), forward_rd.shape)
        )
        reverse_peak = tuple(
            int(x) for x in np.unravel_index(int(np.argmax(reverse_rd)), reverse_rd.shape)
        )

        self.assertEqual(forward_peak, (1, 67))
        self.assertEqual(reverse_peak, (3, 67))

    def test_sim_radar_two_point_targets_create_two_dominant_range_regions(self):
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0},
        ]

        data = sim_radar(radar, targets)
        range_profile = np.abs(processing.range_fft(data["baseband"])[0, 0, :80])
        top_bins = np.argsort(range_profile)[-6:][::-1]

        self.assertIn(27, top_bins.tolist())
        self.assertIn(47, top_bins.tolist())

    def test_sim_radar_static_and_moving_targets_create_separate_range_doppler_regions(self):
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        static_window = range_doppler[0:2, 23:32]
        moving_window = range_doppler[0:4, 63:72]
        static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
        moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
        static_peak = (int(static_peak[0]), int(static_peak[1] + 23))
        moving_peak = (int(moving_peak[0]), int(moving_peak[1] + 63))

        self.assertEqual(static_peak, (0, 27))
        self.assertEqual(moving_peak, (1, 67))

    def test_sim_radar_static_and_negative_moving_targets_create_separate_range_doppler_regions(self):
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        static_window = range_doppler[0:2, 23:32]
        moving_window = range_doppler[2:4, 63:72]
        static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
        moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
        static_peak = (int(static_peak[0]), int(static_peak[1] + 23))
        moving_peak = (int(moving_peak[0] + 2), int(moving_peak[1] + 63))

        self.assertEqual(static_peak, (0, 27))
        self.assertEqual(moving_peak, (3, 67))

    def test_sim_radar_static_static_and_moving_targets_create_three_separate_regions(self):
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        static_a_window = range_doppler[0:2, 23:32]
        static_b_window = range_doppler[0:2, 43:52]
        moving_window = range_doppler[0:4, 63:72]
        static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
        static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
        moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
        static_a_peak = (int(static_a_peak[0]), int(static_a_peak[1] + 23))
        static_b_peak = (int(static_b_peak[0]), int(static_b_peak[1] + 43))
        moving_peak = (int(moving_peak[0]), int(moving_peak[1] + 63))

        self.assertEqual(static_a_peak, (0, 27))
        self.assertEqual(static_b_peak, (0, 47))
        self.assertEqual(moving_peak, (1, 67))

    def test_sim_radar_static_static_and_negative_moving_targets_create_three_separate_regions(self):
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        static_a_window = range_doppler[0:2, 23:32]
        static_b_window = range_doppler[0:2, 43:52]
        moving_window = range_doppler[2:4, 63:72]
        static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
        static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
        moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
        static_a_peak = (int(static_a_peak[0]), int(static_a_peak[1] + 23))
        static_b_peak = (int(static_b_peak[0]), int(static_b_peak[1] + 43))
        moving_peak = (int(moving_peak[0] + 2), int(moving_peak[1] + 63))

        self.assertEqual(static_a_peak, (0, 27))
        self.assertEqual(static_b_peak, (0, 47))
        self.assertEqual(moving_peak, (3, 67))

    def test_sim_radar_static_static_and_moving_moving_targets_create_four_separate_regions(self):
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 0.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        static_a_window = range_doppler[0:2, 23:32]
        static_b_window = range_doppler[0:2, 43:52]
        moving_a_window = range_doppler[0:4, 8:18]
        moving_b_window = range_doppler[0:4, 63:72]
        static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
        static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
        moving_a_peak = np.unravel_index(int(np.argmax(moving_a_window)), moving_a_window.shape)
        moving_b_peak = np.unravel_index(int(np.argmax(moving_b_window)), moving_b_window.shape)
        static_a_peak = (int(static_a_peak[0]), int(static_a_peak[1] + 23))
        static_b_peak = (int(static_b_peak[0]), int(static_b_peak[1] + 43))
        moving_a_peak = (int(moving_a_peak[0]), int(moving_a_peak[1] + 8))
        moving_b_peak = (int(moving_b_peak[0]), int(moving_b_peak[1] + 63))

        self.assertEqual(static_a_peak, (0, 27))
        self.assertEqual(static_b_peak, (0, 47))
        self.assertEqual(moving_a_peak, (1, 14))
        self.assertEqual(moving_b_peak, (1, 67))

    def test_sim_radar_static_static_and_negative_moving_negative_moving_targets_create_four_separate_regions(self):
        radar = self.build_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 0.0, 0.0], "rcs": 7.0, "speed": [-8.0, 0.0, 0.0]},
            {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        range_doppler = np.abs(processing.range_doppler_fft(data["baseband"]))[0, :, :80]
        static_a_window = range_doppler[0:2, 23:32]
        static_b_window = range_doppler[0:2, 43:52]
        moving_a_window = range_doppler[2:4, 8:18]
        moving_b_window = range_doppler[2:4, 63:72]
        static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
        static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
        moving_a_peak = np.unravel_index(int(np.argmax(moving_a_window)), moving_a_window.shape)
        moving_b_peak = np.unravel_index(int(np.argmax(moving_b_window)), moving_b_window.shape)
        static_a_peak = (int(static_a_peak[0]), int(static_a_peak[1] + 23))
        static_b_peak = (int(static_b_peak[0]), int(static_b_peak[1] + 43))
        moving_a_peak = (int(moving_a_peak[0] + 2), int(moving_a_peak[1] + 8))
        moving_b_peak = (int(moving_b_peak[0] + 2), int(moving_b_peak[1] + 63))

        self.assertEqual(static_a_peak, (0, 27))
        self.assertEqual(static_b_peak, (0, 47))
        self.assertEqual(moving_a_peak, (3, 13))
        self.assertEqual(moving_b_peak, (3, 66))

    def test_sim_radar_offaxis_static_static_and_moving_moving_mimo_preserves_four_target_structure(self):
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_a_peaks = []
        static_b_peaks = []
        moving_a_peaks = []
        moving_b_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_a_window = range_doppler[0:4, 8:18]
            moving_b_window = range_doppler[0:4, 63:72]
            static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
            static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
            moving_a_peak = np.unravel_index(int(np.argmax(moving_a_window)), moving_a_window.shape)
            moving_b_peak = np.unravel_index(int(np.argmax(moving_b_window)), moving_b_window.shape)
            static_a_peaks.append((int(static_a_peak[0]), int(static_a_peak[1] + 23)))
            static_b_peaks.append((int(static_b_peak[0]), int(static_b_peak[1] + 43)))
            moving_a_peaks.append((int(moving_a_peak[0]), int(moving_a_peak[1] + 8)))
            moving_b_peaks.append((int(moving_b_peak[0]), int(moving_b_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (4, 4, 160))
        self.assertEqual(static_a_peaks, [(0, 27)] * 4)
        self.assertEqual(static_b_peaks, [(0, 47)] * 4)
        self.assertEqual(moving_a_peaks, [(1, 14)] * 4)
        self.assertEqual(moving_b_peaks, [(1, 67)] * 4)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_static_and_mixed_sign_movers_mimo_preserves_four_target_structure(self):
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_a_peaks = []
        static_b_peaks = []
        moving_pos_peaks = []
        moving_neg_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_pos_window = range_doppler[0:4, 8:18]
            moving_neg_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
            static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
            moving_pos_peak = np.unravel_index(int(np.argmax(moving_pos_window)), moving_pos_window.shape)
            moving_neg_peak = np.unravel_index(int(np.argmax(moving_neg_window)), moving_neg_window.shape)
            static_a_peaks.append((int(static_a_peak[0]), int(static_a_peak[1] + 23)))
            static_b_peaks.append((int(static_b_peak[0]), int(static_b_peak[1] + 43)))
            moving_pos_peaks.append((int(moving_pos_peak[0]), int(moving_pos_peak[1] + 8)))
            moving_neg_peaks.append((int(moving_neg_peak[0] + 2), int(moving_neg_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (4, 4, 160))
        self.assertEqual(static_a_peaks, [(0, 27)] * 4)
        self.assertEqual(static_b_peaks, [(0, 47)] * 4)
        self.assertEqual(moving_pos_peaks, [(1, 14)] * 4)
        self.assertEqual(moving_neg_peaks, [(3, 67)] * 4)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_static_and_mixed_sign_movers_mimo_multiframe_preserves_four_target_structure(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_a_peaks = []
        static_b_peaks = []
        moving_pos_peaks = []
        moving_neg_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_pos_window = range_doppler[0:4, 8:18]
            moving_neg_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
            static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
            moving_pos_peak = np.unravel_index(int(np.argmax(moving_pos_window)), moving_pos_window.shape)
            moving_neg_peak = np.unravel_index(int(np.argmax(moving_neg_window)), moving_neg_window.shape)
            static_a_peaks.append((int(static_a_peak[0]), int(static_a_peak[1] + 23)))
            static_b_peaks.append((int(static_b_peak[0]), int(static_b_peak[1] + 43)))
            moving_pos_peaks.append((int(moving_pos_peak[0]), int(moving_pos_peak[1] + 8)))
            moving_neg_peaks.append((int(moving_neg_peak[0] + 2), int(moving_neg_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:4, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertEqual(static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(moving_pos_peaks, [(1, 14)] * 8)
        self.assertEqual(moving_neg_peaks, [(3, 67)] * 8)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_static_and_negative_moving_negative_moving_mimo_multiframe_preserves_four_target_structure(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [-8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_a_peaks = []
        static_b_peaks = []
        moving_a_peaks = []
        moving_b_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_a_window = range_doppler[2:4, 8:18]
            moving_b_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
            static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
            moving_a_peak = np.unravel_index(int(np.argmax(moving_a_window)), moving_a_window.shape)
            moving_b_peak = np.unravel_index(int(np.argmax(moving_b_window)), moving_b_window.shape)
            static_a_peaks.append((int(static_a_peak[0]), int(static_a_peak[1] + 23)))
            static_b_peaks.append((int(static_b_peak[0]), int(static_b_peak[1] + 43)))
            moving_a_peaks.append((int(moving_a_peak[0] + 2), int(moving_a_peak[1] + 8)))
            moving_b_peaks.append((int(moving_b_peak[0] + 2), int(moving_b_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:4, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertEqual(static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(moving_a_peaks, [(3, 13)] * 8)
        self.assertTrue(all(peak in {(3, 66), (3, 67)} for peak in moving_b_peaks))
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_and_moving_mimo_preserves_mixed_scene_and_phase(self):
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_peaks = []
        moving_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_window = range_doppler[0:2, 23:32]
            moving_window = range_doppler[0:4, 63:72]
            static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_peaks.append((int(static_peak[0]), int(static_peak[1] + 23)))
            moving_peaks.append((int(moving_peak[0]), int(moving_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (4, 4, 160))
        self.assertEqual(static_peaks, [(0, 27)] * 4)
        self.assertEqual(moving_peaks, [(1, 67)] * 4)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_and_negative_moving_mimo_preserves_mixed_scene_and_phase(self):
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_peaks = []
        moving_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_window = range_doppler[0:2, 23:32]
            moving_window = range_doppler[2:4, 63:72]
            static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_peaks.append((int(static_peak[0]), int(static_peak[1] + 23)))
            moving_peaks.append((int(moving_peak[0] + 2), int(moving_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (4, 4, 160))
        self.assertEqual(static_peaks, [(0, 27)] * 4)
        self.assertEqual(moving_peaks, [(3, 67)] * 4)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_static_and_moving_mimo_preserves_three_target_structure(self):
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_a_peaks = []
        static_b_peaks = []
        moving_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_window = range_doppler[0:4, 63:72]
            static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
            static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_a_peaks.append((int(static_a_peak[0]), int(static_a_peak[1] + 23)))
            static_b_peaks.append((int(static_b_peak[0]), int(static_b_peak[1] + 43)))
            moving_peaks.append((int(moving_peak[0]), int(moving_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (4, 4, 160))
        self.assertEqual(static_a_peaks, [(0, 27)] * 4)
        self.assertEqual(static_b_peaks, [(0, 47)] * 4)
        self.assertEqual(moving_peaks, [(1, 67)] * 4)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_static_and_negative_moving_mimo_preserves_three_target_structure(self):
        radar = self.build_mimo_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_a_peaks = []
        static_b_peaks = []
        moving_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
            static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_a_peaks.append((int(static_a_peak[0]), int(static_a_peak[1] + 23)))
            static_b_peaks.append((int(static_b_peak[0]), int(static_b_peak[1] + 43)))
            moving_peaks.append((int(moving_peak[0] + 2), int(moving_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (4, 4, 160))
        self.assertEqual(static_a_peaks, [(0, 27)] * 4)
        self.assertEqual(static_b_peaks, [(0, 47)] * 4)
        self.assertEqual(moving_peaks, [(3, 67)] * 4)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_and_moving_mimo_multiframe_preserves_mixed_scene_and_phase(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_peaks = []
        moving_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_window = range_doppler[0:2, 23:32]
            moving_window = range_doppler[0:4, 63:72]
            static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_peaks.append((int(static_peak[0]), int(static_peak[1] + 23)))
            moving_peaks.append((int(moving_peak[0]), int(moving_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:4, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertEqual(static_peaks, [(0, 27)] * 8)
        self.assertEqual(moving_peaks, [(1, 67)] * 8)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_and_negative_moving_mimo_multiframe_preserves_mixed_scene_and_phase(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_peaks = []
        moving_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_window = range_doppler[0:2, 23:32]
            moving_window = range_doppler[2:4, 63:72]
            static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_peaks.append((int(static_peak[0]), int(static_peak[1] + 23)))
            moving_peaks.append((int(moving_peak[0] + 2), int(moving_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:4, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertEqual(static_peaks, [(0, 27)] * 8)
        self.assertEqual(moving_peaks, [(3, 67)] * 8)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_static_and_moving_mimo_multiframe_preserves_three_target_structure(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_a_peaks = []
        static_b_peaks = []
        moving_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_window = range_doppler[0:4, 63:72]
            static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
            static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_a_peaks.append((int(static_a_peak[0]), int(static_a_peak[1] + 23)))
            static_b_peaks.append((int(static_b_peak[0]), int(static_b_peak[1] + 43)))
            moving_peaks.append((int(moving_peak[0]), int(moving_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:4, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertEqual(static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(moving_peaks, [(1, 67)] * 8)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_static_and_negative_moving_mimo_multiframe_preserves_three_target_structure(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_a_peaks = []
        static_b_peaks = []
        moving_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
            static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_a_peaks.append((int(static_a_peak[0]), int(static_a_peak[1] + 23)))
            static_b_peaks.append((int(static_b_peak[0]), int(static_b_peak[1] + 43)))
            moving_peaks.append((int(moving_peak[0] + 2), int(moving_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:4, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertEqual(static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(moving_peaks, [(3, 67)] * 8)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_offaxis_static_static_and_moving_moving_mimo_multiframe_preserves_four_target_structure(self):
        radar = self.build_mimo_multiframe_radar(seed=2026)
        targets = [
            {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
            {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
            {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
        ]

        data = sim_radar(radar, targets)
        static_a_peaks = []
        static_b_peaks = []
        moving_a_peaks = []
        moving_b_peaks = []
        for channel_index in range(data["baseband"].shape[0]):
            range_doppler = np.abs(
                processing.range_doppler_fft(data["baseband"][channel_index : channel_index + 1])
            )[0, :, :80]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_a_window = range_doppler[0:4, 8:18]
            moving_b_window = range_doppler[0:4, 63:72]
            static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
            static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
            moving_a_peak = np.unravel_index(int(np.argmax(moving_a_window)), moving_a_window.shape)
            moving_b_peak = np.unravel_index(int(np.argmax(moving_b_window)), moving_b_window.shape)
            static_a_peaks.append((int(static_a_peak[0]), int(static_a_peak[1] + 23)))
            static_b_peaks.append((int(static_b_peak[0]), int(static_b_peak[1] + 43)))
            moving_a_peaks.append((int(moving_a_peak[0]), int(moving_a_peak[1] + 8)))
            moving_b_peaks.append((int(moving_b_peak[0]), int(moving_b_peak[1] + 63)))

        phase_sample_index = 2
        ref = data["baseband"][0, 0, phase_sample_index]
        channel_phase_deg = np.degrees(
            np.angle(data["baseband"][:4, 0, phase_sample_index] / ref)
        )

        self.assertEqual(data["baseband"].shape, (8, 4, 160))
        self.assertTrue(np.array_equal(data["timestamp"][:4] + 1e-3, data["timestamp"][4:]))
        self.assertEqual(static_a_peaks, [(0, 27)] * 8)
        self.assertEqual(static_b_peaks, [(0, 47)] * 8)
        self.assertEqual(moving_a_peaks, [(1, 14)] * 8)
        self.assertEqual(moving_b_peaks, [(1, 67)] * 8)
        self.assertEqual(len(np.unique(np.round(channel_phase_deg, 6))), 4)

    def test_sim_radar_range_peak_tracks_target_distance(self):
        radar = self.build_radar(seed=1)

        samples = radar.samples_per_pulse
        fs = radar.receiver.bb_prop["fs"]
        pulse_length = radar.transmitter.waveform_prop["pulse_length"]
        bandwidth = radar.transmitter.waveform_prop["bandwidth"]

        target_bin = 20
        target_range = (
            target_bin * SPEED_OF_LIGHT * fs * pulse_length / (2 * bandwidth * samples)
        )
        target = {"location": [target_range, 0.0, 0.0], "rcs": 10.0}

        data = sim_radar(radar, [target], dry_run=False)
        range_profile = processing.range_fft(data["baseband"])
        valid_bins = samples // 2
        peak_bin = int(np.argmax(np.abs(range_profile[0, 0, :valid_bins])))

        self.assertLessEqual(abs(peak_bin - target_bin), 1)

    def test_sim_radar_accepts_static_mesh_targets(self):
        radar = self.build_radar(seed=7)
        mesh_path = self.write_temp_stl()

        data = sim_radar(radar, [{"model": mesh_path, "location": [30.0, 0.0, 0.0]}])

        self.assertEqual(set(data), {"timestamp", "baseband", "noise", "interference"})
        self.assertGreater(np.linalg.norm(data["baseband"]), 0.0)

    def test_sim_radar_mesh_aspect_changes_return_strength(self):
        radar = self.build_radar(seed=7)
        mesh_path = self.write_temp_stl()

        broadside = sim_radar(
            radar,
            [{"model": mesh_path, "location": [30.0, 0.0, 0.0]}],
        )
        edge_on = sim_radar(
            radar,
            [{"model": mesh_path, "location": [30.0, 0.0, 0.0], "rotation": [90.0, 0.0, 0.0]}],
        )

        self.assertGreater(
            np.linalg.norm(broadside["baseband"]),
            np.linalg.norm(edge_on["baseband"]),
        )

    def test_sim_radar_rejects_interference_radar_for_now(self):
        radar = self.build_radar()
        interferer = self.build_radar()
        target = {"location": [25.0, 0.0, 0.0], "rcs": 0.0}

        with self.assertRaises(NotImplementedError):
            sim_radar(radar, [target], interf=interferer)


if __name__ == "__main__":
    unittest.main()
