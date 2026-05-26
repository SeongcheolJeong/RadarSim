import unittest

import numpy as np

import radarsimpy_whitebox as rs


class Phase9ModulationHelperTests(unittest.TestCase):
    def test_tdm_code_creates_round_robin_pulse_mask(self):
        code = rs.tdm_code(tx_channels=3, pulses=7)

        self.assertEqual(code["pulse_amp"].shape, (3, 7))
        self.assertEqual(code["pulse_phs"].shape, (3, 7))
        self.assertTrue(np.array_equal(np.sum(code["pulse_amp"], axis=0), np.ones(7)))
        self.assertTrue(np.array_equal(np.argmax(code["pulse_amp"], axis=0), [0, 1, 2, 0, 1, 2, 0]))
        self.assertTrue(np.array_equal(code["pulse_phs"], np.zeros((3, 7))))

    def test_tdm_code_accepts_custom_order_and_channel_phase(self):
        code = rs.tdm_code(
            tx_channels=3,
            pulses=5,
            order=[2, 0],
            active_phase=[0.0, 45.0, 90.0],
        )

        self.assertTrue(np.array_equal(np.argmax(code["pulse_amp"], axis=0), [2, 0, 2, 0, 2]))
        self.assertTrue(np.array_equal(code["pulse_phs"][2], [90.0, 0.0, 90.0, 0.0, 90.0]))

    def test_bpm_code_creates_repeated_hadamard_phase_code(self):
        code = rs.bpm_code(tx_channels=2, pulses=5)

        self.assertTrue(np.array_equal(code["pulse_amp"], np.ones((2, 5))))
        self.assertTrue(np.array_equal(code["pulse_phs"][0], [0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.array_equal(code["pulse_phs"][1], [0.0, 180.0, 0.0, 180.0, 0.0]))

    def test_bpm_code_uses_custom_binary_code_matrix(self):
        code = rs.bpm_code(
            tx_channels=3,
            pulses=4,
            code_matrix=[
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
            ],
        )

        self.assertTrue(np.array_equal(code["pulse_phs"][0], [0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.array_equal(code["pulse_phs"][1], [0.0, 180.0, 0.0, 180.0]))
        self.assertTrue(np.array_equal(code["pulse_phs"][2], [180.0, 0.0, 180.0, 0.0]))

    def test_bpm_default_requires_power_of_two_channel_count(self):
        with self.assertRaises(ValueError):
            rs.bpm_code(tx_channels=3, pulses=4)

    def test_apply_pulse_modulation_returns_channel_configs_without_mutating_input(self):
        channels = [
            {"location": [0.0, 0.0, 0.0]},
            {"location": [0.0, 0.05, 0.0]},
        ]
        modulated = rs.bpm_channels(channels, pulses=4)

        self.assertNotIn("pulse_amp", channels[0])
        self.assertNotIn("pulse_phs", channels[1])
        self.assertTrue(np.array_equal(modulated[1]["pulse_phs"], [0.0, 180.0, 0.0, 180.0]))

    def test_helpers_are_exported_at_package_root(self):
        for name in (
            "apply_pulse_modulation",
            "bpm",
            "bpm_channels",
            "bpm_code",
            "tdm",
            "tdm_channels",
            "tdm_code",
        ):
            self.assertIn(name, rs.__all__)
            self.assertTrue(hasattr(rs, name))

    def test_bpm_channels_affect_sim_radar_pulse_phase(self):
        base_channels = [
            {"location": [0.0, 0.0, 0.0]},
            {"location": [0.0, 0.05, 0.0]},
        ]
        rx = rs.Receiver(
            fs=4e6,
            noise_figure=11,
            rf_gain=0,
            load_resistor=500,
            baseband_gain=0,
            bb_type="complex",
            channels=[{"location": [0.0, 0.0, 0.0]}],
        )
        tx_plain = rs.Transmitter(
            f=[76.0e9, 76.2e9],
            t=[0.0, 40e-6],
            tx_power=13,
            pulses=4,
            prp=60e-6,
            channels=base_channels,
        )
        tx_bpm = rs.Transmitter(
            f=[76.0e9, 76.2e9],
            t=[0.0, 40e-6],
            tx_power=13,
            pulses=4,
            prp=60e-6,
            channels=rs.bpm_channels(base_channels, pulses=4),
        )
        target = [{"location": [50.0, 0.0, 0.0], "rcs": 10.0}]

        plain = rs.sim_radar(rs.Radar(tx_plain, rx, seed=2026), target)["baseband"]
        coded = rs.sim_radar(rs.Radar(tx_bpm, rx, seed=2026), target)["baseband"]

        peak_sample = int(np.argmax(np.abs(plain[1, 0])))
        tx0_ratio = coded[0, :, peak_sample] / plain[0, :, peak_sample]
        tx1_ratio = coded[1, :, peak_sample] / plain[1, :, peak_sample]

        self.assertTrue(np.allclose(tx0_ratio, np.ones(4)))
        self.assertTrue(np.allclose(tx1_ratio, [1.0, -1.0, 1.0, -1.0]))


if __name__ == "__main__":
    unittest.main()
