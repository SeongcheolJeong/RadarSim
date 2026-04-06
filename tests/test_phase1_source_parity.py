import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest

import numpy as np

from radarsimpy_whitebox import processing as wb_processing
from radarsimpy_whitebox import tools as wb_tools


def load_origin_source_bundle(package_name: str):
    root = Path(__file__).resolve().parents[1]
    origin_dir = root / "radarsimpy_origin"

    package = types.ModuleType(package_name)
    package.__path__ = [str(origin_dir)]
    sys.modules[package_name] = package

    loaded = {}
    for module_name in ("tools", "processing"):
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


class Phase1SourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = load_origin_source_bundle("_radarsimpy_origin_testbundle")
        cls.origin_tools = modules["tools"]
        cls.origin_processing = modules["processing"]

    def test_processing_signature_parity(self):
        public_names = [
            "range_fft",
            "doppler_fft",
            "range_doppler_fft",
            "cfar_ca_1d",
            "cfar_ca_2d",
            "os_cfar_threshold",
            "cfar_os_1d",
            "cfar_os_2d",
            "doa_music",
            "doa_root_music",
            "doa_esprit",
            "doa_iaa",
            "doa_bartlett",
            "doa_capon",
        ]

        for name in public_names:
            with self.subTest(name=name):
                self.assertEqual(
                    str(inspect.signature(getattr(wb_processing, name))),
                    str(inspect.signature(getattr(self.origin_processing, name))),
                )

    def test_tools_signature_parity(self):
        public_names = [
            "marcumq",
            "log_factorial",
            "threshold",
            "pd_swerling0",
            "pd_swerling1",
            "pd_swerling2",
            "pd_swerling3",
            "pd_swerling4",
            "roc_pd",
            "roc_snr",
        ]

        for name in public_names:
            with self.subTest(name=name):
                self.assertEqual(
                    str(inspect.signature(getattr(wb_tools, name))),
                    str(inspect.signature(getattr(self.origin_tools, name))),
                )

    def test_tools_module_is_independent_now(self):
        source_text = Path(wb_tools.__file__).read_text(encoding="utf-8")

        self.assertNotIn("load_origin_module", source_text)
        self.assertIn("def roc_pd(", source_text)

    def test_processing_module_is_independent_now(self):
        source_text = Path(wb_processing.__file__).read_text(encoding="utf-8")

        self.assertNotIn("load_origin_module", source_text)
        self.assertIn("def range_fft(", source_text)

    def test_range_fft_matches_origin_source(self):
        data = np.array(
            [
                [
                    [1 + 1j, 2 + 0j, 0 + 1j, 0 + 0j],
                    [0 + 0j, 1 + 0j, 2 + 2j, 3 + 0j],
                ]
            ],
            dtype=complex,
        )
        rwin = np.hanning(4)

        expected = self.origin_processing.range_fft(data, rwin=rwin, n=8)
        actual = wb_processing.range_fft(data, rwin=rwin, n=8)

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected))

    def test_cfar_ca_1d_matches_origin_source(self):
        data = np.array([0.2, 0.3, 1.1, 0.4, 0.5, 0.6, 0.7], dtype=float)

        expected = self.origin_processing.cfar_ca_1d(
            data, guard=1, trailing=2, pfa=1e-3, axis=0
        )
        actual = wb_processing.cfar_ca_1d(
            data, guard=1, trailing=2, pfa=1e-3, axis=0
        )

        self.assertTrue(np.allclose(actual, expected))

    def test_cfar_ca_1d_rejects_complex_input(self):
        data = np.array([1 + 1j, 2 + 0j])

        with self.assertRaises(ValueError):
            wb_processing.cfar_ca_1d(data, guard=1, trailing=1)

    def test_os_cfar_threshold_matches_origin_source(self):
        expected = self.origin_processing.os_cfar_threshold(4, 8, 1e-4)
        actual = wb_processing.os_cfar_threshold(4, 8, 1e-4)
        self.assertAlmostEqual(actual, expected)

    def test_roc_functions_match_origin_source(self):
        pfa = np.array([1e-3, 1e-4])
        snr = np.array([5.0, 10.0])

        expected_pd = self.origin_tools.roc_pd(pfa, snr, npulses=4, stype="Swerling 1")
        actual_pd = wb_tools.roc_pd(pfa, snr, npulses=4, stype="Swerling 1")

        self.assertTrue(np.allclose(actual_pd, expected_pd))

        pd = np.array([0.5, 0.8])
        expected_snr = self.origin_tools.roc_snr(
            pfa, pd, npulses=2, stype="Coherent"
        )
        actual_snr = wb_tools.roc_snr(pfa, pd, npulses=2, stype="Coherent")

        self.assertTrue(np.allclose(actual_snr, expected_snr))


if __name__ == "__main__":
    unittest.main()
