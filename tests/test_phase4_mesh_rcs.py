from pathlib import Path
import tempfile
import unittest

import numpy as np

from radarsimpy_whitebox import mesh_kit, sim_rcs


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


class Phase4MeshAndRcsTests(unittest.TestCase):
    def write_temp_stl(self) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        mesh_path = Path(tmpdir.name) / "plate.stl"
        mesh_path.write_text(PLATE_STL, encoding="utf-8")
        return str(mesh_path)

    def test_import_mesh_module_returns_builtin_when_no_external_libs(self):
        module = mesh_kit.import_mesh_module()
        self.assertEqual(module.__name__, "builtin_ascii")

    def test_load_mesh_builtin_ascii_stl(self):
        mesh_path = self.write_temp_stl()
        module = mesh_kit.import_mesh_module()
        mesh = mesh_kit.load_mesh(mesh_path, 1.0, module)

        self.assertEqual(set(mesh), {"points", "cells"})
        self.assertEqual(mesh["points"].shape, (4, 3))
        self.assertEqual(mesh["cells"].shape, (2, 3))

    def test_sim_rcs_returns_scalar_for_scalar_angles(self):
        mesh_path = self.write_temp_stl()
        value = sim_rcs(
            [{"model": mesh_path, "location": [0.0, 0.0, 0.0]}],
            f=77e9,
            inc_phi=0.0,
            inc_theta=0.0,
        )

        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0)

    def test_sim_rcs_returns_array_for_array_angles(self):
        mesh_path = self.write_temp_stl()
        values = sim_rcs(
            [{"model": mesh_path}],
            f=77e9,
            inc_phi=np.array([0.0, 90.0]),
            inc_theta=np.array([0.0, 0.0]),
        )

        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values.shape, (2,))

    def test_sim_rcs_broadside_exceeds_edge_on_for_flat_plate(self):
        mesh_path = self.write_temp_stl()
        broadside = sim_rcs(
            [{"model": mesh_path}],
            f=77e9,
            inc_phi=0.0,
            inc_theta=90.0,
        )
        edge_on = sim_rcs(
            [{"model": mesh_path}],
            f=77e9,
            inc_phi=90.0,
            inc_theta=90.0,
        )

        self.assertGreater(broadside, edge_on)

    def test_sim_rcs_translation_preserves_magnitude(self):
        mesh_path = self.write_temp_stl()
        base = sim_rcs([{"model": mesh_path}], f=77e9, inc_phi=0.0, inc_theta=90.0)
        shifted = sim_rcs(
            [{"model": mesh_path, "location": [10.0, 3.0, -2.0]}],
            f=77e9,
            inc_phi=0.0,
            inc_theta=90.0,
        )

        self.assertAlmostEqual(base, shifted, places=8)

    def test_sim_rcs_permittivity_reduces_response(self):
        mesh_path = self.write_temp_stl()
        pec_like = sim_rcs([{"model": mesh_path}], f=77e9, inc_phi=0.0, inc_theta=90.0)
        dielectric = sim_rcs(
            [{"model": mesh_path, "permittivity": 2.5 + 0.0j}],
            f=77e9,
            inc_phi=0.0,
            inc_theta=90.0,
        )

        self.assertLess(dielectric, pec_like)

    def test_sim_rcs_obs_phi_sweep_has_single_main_lobe_and_quiet_tail(self):
        mesh_path = self.write_temp_stl()
        obs_phi = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
        values = sim_rcs(
            [{"model": mesh_path}],
            f=77e9,
            inc_phi=np.zeros_like(obs_phi),
            inc_theta=np.full_like(obs_phi, 90.0),
            obs_phi=obs_phi,
            obs_theta=np.full_like(obs_phi, 90.0),
        )

        self.assertEqual(int(np.argmax(values)), 0)
        self.assertTrue(np.all(np.diff(values[:3]) < 0.0))
        self.assertLessEqual(float(np.max(values[3:]) / np.max(values)), 1e-6)

    def test_sim_rcs_cross_polarization_is_quiet_against_co_pol(self):
        mesh_path = self.write_temp_stl()
        co_pol = sim_rcs(
            [{"model": mesh_path}],
            f=77e9,
            inc_phi=0.0,
            inc_theta=90.0,
            inc_pol=[0.0, 0.0, 1.0],
            obs_phi=0.0,
            obs_theta=90.0,
            obs_pol=[0.0, 0.0, 1.0],
        )
        cross_pol = sim_rcs(
            [{"model": mesh_path}],
            f=77e9,
            inc_phi=0.0,
            inc_theta=90.0,
            inc_pol=[0.0, 0.0, 1.0],
            obs_phi=0.0,
            obs_theta=90.0,
            obs_pol=[0.0, 1.0, 0.0],
        )

        self.assertGreater(co_pol, 0.0)
        self.assertEqual(cross_pol, 0.0)


if __name__ == "__main__":
    unittest.main()
