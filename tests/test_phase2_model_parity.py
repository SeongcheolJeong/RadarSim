import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest

import numpy as np

from radarsimpy_whitebox import Radar, Receiver, Transmitter
from radarsimpy_whitebox.radar import cal_phase_noise as wb_cal_phase_noise


def load_origin_model_bundle(package_name: str):
    root = Path(__file__).resolve().parents[1]
    origin_dir = root / "radarsimpy_origin"

    package = types.ModuleType(package_name)
    package.__path__ = [str(origin_dir)]
    sys.modules[package_name] = package

    loaded = {}
    for module_name in ("transmitter", "receiver", "radar"):
        module_path = origin_dir / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module

    return loaded


def signature_shape(callable_obj):
    sig = inspect.signature(callable_obj)
    return [
        (param.name, param.kind.name, param.default is inspect._empty, param.default)
        for param in sig.parameters.values()
    ]


class Phase2ModelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = load_origin_model_bundle("_radarsimpy_origin_model_testbundle")
        cls.origin_transmitter_module = modules["transmitter"]
        cls.origin_receiver_module = modules["receiver"]
        cls.origin_radar_module = modules["radar"]
        cls.origin_transmitter_cls = modules["transmitter"].Transmitter
        cls.origin_receiver_cls = modules["receiver"].Receiver
        cls.origin_radar_cls = modules["radar"].Radar

    def test_constructor_signature_parity(self):
        pairs = [
            (Transmitter, self.origin_transmitter_cls),
            (Receiver, self.origin_receiver_cls),
            (Radar, self.origin_radar_cls),
        ]

        for whitebox_cls, origin_cls in pairs:
            with self.subTest(cls=whitebox_cls.__name__):
                self.assertEqual(
                    signature_shape(whitebox_cls),
                    signature_shape(origin_cls),
                )

    def test_set_motion_signature_parity(self):
        self.assertEqual(
            signature_shape(Radar.set_motion),
            signature_shape(self.origin_radar_cls.set_motion),
        )

    def test_transmitter_contract_matches_origin_source(self):
        channel_config = [
            {
                "location": [0.0, 0.0, 0.0],
                "polarization": [0, 0, 1],
                "delay": 1e-9,
                "grid": 2.0,
                "azimuth_angle": [-90, 0, 90],
                "azimuth_pattern": [-3, 0, -3],
                "elevation_angle": [-45, 0, 45],
                "elevation_pattern": [-2, 0, -2],
                "pulse_amp": [1.0, 0.5],
                "pulse_phs": [0.0, 90.0],
                "mod_t": [0.0, 1e-6],
                "amp": [1.0, 0.75],
                "phs": [0.0, 180.0],
            }
        ]

        expected = self.origin_transmitter_cls(
            f=[77e9, 77.1e9],
            t=[0.0, 1e-6],
            tx_power=12,
            pulses=2,
            prp=[2e-6, 2.5e-6],
            f_offset=[0.0, 1e6],
            pn_f=np.array([1e3, 1e4]),
            pn_power=np.array([-80.0, -100.0]),
            channels=channel_config,
        )
        actual = Transmitter(
            f=[77e9, 77.1e9],
            t=[0.0, 1e-6],
            tx_power=12,
            pulses=2,
            prp=[2e-6, 2.5e-6],
            f_offset=[0.0, 1e6],
            pn_f=np.array([1e3, 1e4]),
            pn_power=np.array([-80.0, -100.0]),
            channels=channel_config,
        )

        for key in expected.rf_prop:
            self.assertTrue(np.array_equal(actual.rf_prop[key], expected.rf_prop[key]))
        for key in expected.waveform_prop:
            self.assertTrue(
                np.array_equal(actual.waveform_prop[key], expected.waveform_prop[key])
            )
        for key in (
            "delay",
            "grid",
            "locations",
            "polarization",
            "pulse_mod",
            "antenna_gains",
        ):
            self.assertTrue(
                np.array_equal(actual.txchannel_prop[key], expected.txchannel_prop[key])
            )
        for key in ("az_angles", "az_patterns", "el_angles", "el_patterns"):
            for actual_arr, expected_arr in zip(
                actual.txchannel_prop[key], expected.txchannel_prop[key]
            ):
                self.assertTrue(np.array_equal(actual_arr, expected_arr))

        actual_info = actual.get_channel_info(0)
        expected_info = expected.get_channel_info(0)
        for key in expected_info:
            self.assertTrue(np.array_equal(actual_info[key], expected_info[key]))

    def test_receiver_contract_matches_origin_source(self):
        channels = [
            {
                "location": [0.0, 0.0, 0.0],
                "azimuth_angle": [-60, 0, 60],
                "azimuth_pattern": [-6, 0, -6],
                "elevation_angle": [-30, 0, 30],
                "elevation_pattern": [-4, 0, -4],
            },
            {"location": [0.0, 0.05, 0.0], "polarization": [0, 1, 0]},
        ]
        expected = self.origin_receiver_cls(
            fs=4e6,
            noise_figure=8,
            rf_gain=20,
            load_resistor=600,
            baseband_gain=10,
            bb_type="real",
            channels=channels,
        )
        actual = Receiver(
            fs=4e6,
            noise_figure=8,
            rf_gain=20,
            load_resistor=600,
            baseband_gain=10,
            bb_type="real",
            channels=channels,
        )

        self.assertEqual(actual.bb_prop, expected.bb_prop)
        self.assertEqual(actual.rf_prop, expected.rf_prop)
        for key in ("locations", "polarization", "antenna_gains"):
            self.assertTrue(
                np.array_equal(actual.rxchannel_prop[key], expected.rxchannel_prop[key])
            )
        for key in ("az_angles", "az_patterns", "el_angles", "el_patterns"):
            for actual_arr, expected_arr in zip(
                actual.rxchannel_prop[key], expected.rxchannel_prop[key]
            ):
                self.assertTrue(np.array_equal(actual_arr, expected_arr))

    def test_phase_noise_matches_origin_source_in_validation_mode(self):
        signal = np.ones((1, 8), dtype=complex)
        freq = np.array([1e3, 1e4, 1e5])
        power = np.array([-80.0, -90.0, -110.0])

        expected = self.origin_radar_module.cal_phase_noise(
            signal, 1e6, freq, power, seed=123, validation=True
        )
        actual = wb_cal_phase_noise(signal, 1e6, freq, power, seed=123, validation=True)

        self.assertTrue(np.allclose(actual, expected))

    def test_radar_single_frame_contract_matches_origin_source(self):
        origin_tx = self.origin_transmitter_cls(
            f=[77e9, 77.1e9],
            t=[0.0, 1e-6],
            pulses=2,
            prp=[2e-6, 3e-6],
            channels=[
                {"location": [0.0, 0.0, 0.0], "delay": 0.0},
                {"location": [0.1, 0.0, 0.0], "delay": 1e-9},
            ],
        )
        origin_rx = self.origin_receiver_cls(
            fs=4e6,
            bb_type="complex",
            channels=[
                {"location": [0.0, 0.0, 0.0]},
                {"location": [0.0, 0.05, 0.0]},
            ],
        )
        expected = self.origin_radar_cls(
            transmitter=origin_tx,
            receiver=origin_rx,
            frame_time=0.0,
            location=(1.0, 2.0, 3.0),
            speed=(0.1, 0.2, 0.3),
            rotation=(10.0, 20.0, 30.0),
            rotation_rate=(1.0, 2.0, 3.0),
        )

        actual_tx = Transmitter(
            f=[77e9, 77.1e9],
            t=[0.0, 1e-6],
            pulses=2,
            prp=[2e-6, 3e-6],
            channels=[
                {"location": [0.0, 0.0, 0.0], "delay": 0.0},
                {"location": [0.1, 0.0, 0.0], "delay": 1e-9},
            ],
        )
        actual_rx = Receiver(
            fs=4e6,
            bb_type="complex",
            channels=[
                {"location": [0.0, 0.0, 0.0]},
                {"location": [0.0, 0.05, 0.0]},
            ],
        )
        actual = Radar(
            transmitter=actual_tx,
            receiver=actual_rx,
            frame_time=0.0,
            location=(1.0, 2.0, 3.0),
            speed=(0.1, 0.2, 0.3),
            rotation=(10.0, 20.0, 30.0),
            rotation_rate=(1.0, 2.0, 3.0),
        )

        self.assertEqual(actual.num_channels, expected.num_channels)
        self.assertEqual(actual.samples_per_pulse, expected.samples_per_pulse)
        self.assertTrue(
            np.array_equal(actual.virtual_array_locations, expected.virtual_array_locations)
        )
        self.assertTrue(
            np.array_equal(actual.time_prop["origin_timestamp"], expected.time_prop["origin_timestamp"])
        )
        self.assertTrue(
            np.array_equal(actual.time_prop["timestamp"], expected.time_prop["timestamp"])
        )
        self.assertAlmostEqual(actual.sample_prop["noise"], expected.sample_prop["noise"])
        self.assertTrue(np.array_equal(actual.radar_prop["location"], expected.radar_prop["location"]))
        self.assertTrue(np.array_equal(actual.radar_prop["speed"], expected.radar_prop["speed"]))
        self.assertTrue(np.array_equal(actual.radar_prop["rotation"], expected.radar_prop["rotation"]))
        self.assertTrue(
            np.array_equal(actual.radar_prop["rotation_rate"], expected.radar_prop["rotation_rate"])
        )

    def test_radar_multi_frame_timestamp_matches_origin_source(self):
        origin_tx = self.origin_transmitter_cls(f=[1.0, 2.0], t=[0.0, 1.0], pulses=1)
        origin_rx = self.origin_receiver_cls(fs=2.0)
        expected = self.origin_radar_cls(
            transmitter=origin_tx,
            receiver=origin_rx,
            frame_time=[0.0, 10.0],
        )

        actual_tx = Transmitter(f=[1.0, 2.0], t=[0.0, 1.0], pulses=1)
        actual_rx = Receiver(fs=2.0)
        actual = Radar(
            transmitter=actual_tx,
            receiver=actual_rx,
            frame_time=[0.0, 10.0],
        )

        self.assertEqual(actual.time_prop["timestamp_shape"], expected.time_prop["timestamp_shape"])
        self.assertTrue(np.array_equal(actual.time_prop["timestamp"], expected.time_prop["timestamp"]))

    def test_radar_time_varying_motion_validation_matches_origin_source(self):
        origin_tx = self.origin_transmitter_cls(f=[1.0, 2.0], t=[0.0, 1.0], pulses=1)
        origin_rx = self.origin_receiver_cls(fs=2.0)
        origin_radar = self.origin_radar_cls(
            transmitter=origin_tx,
            receiver=origin_rx,
            frame_time=[0.0, 5.0],
        )

        actual_tx = Transmitter(f=[1.0, 2.0], t=[0.0, 1.0], pulses=1)
        actual_rx = Receiver(fs=2.0)
        actual_radar = Radar(
            transmitter=actual_tx,
            receiver=actual_rx,
            frame_time=[0.0, 5.0],
        )

        location_x = np.ones(actual_radar.time_prop["timestamp_shape"])
        origin_location_x = np.ones(origin_radar.time_prop["timestamp_shape"])

        with self.assertRaisesRegex(ValueError, "speed must be \\[0, 0, 0\\]"):
            actual_radar.set_motion(location=[location_x, 0, 0], speed=[1, 0, 0])
        with self.assertRaisesRegex(ValueError, "speed must be \\[0, 0, 0\\]"):
            origin_radar.set_motion(location=[origin_location_x, 0, 0], speed=[1, 0, 0])

        actual_radar.set_motion(location=[location_x, 0, 0], speed=[0, 0, 0])
        origin_radar.set_motion(location=[origin_location_x, 0, 0], speed=[0, 0, 0])

        self.assertTrue(
            np.array_equal(actual_radar.radar_prop["location"], origin_radar.radar_prop["location"])
        )


if __name__ == "__main__":
    unittest.main()
