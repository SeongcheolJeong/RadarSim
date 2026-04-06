import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from radarsimpy_whitebox import sim_lidar


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


class Phase5LidarTests(unittest.TestCase):
    def write_temp_stl(self) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        mesh_path = Path(tmpdir.name) / "plate.stl"
        mesh_path.write_text(PLATE_STL, encoding="utf-8")
        return str(mesh_path)

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

    def test_sim_lidar_signature(self):
        signature = str(inspect.signature(sim_lidar))
        self.assertEqual(signature, "(lidar, targets, frame_time=0)")

    def test_sim_lidar_returns_structured_hit_array(self):
        mesh_path = self.write_temp_stl()
        hits = sim_lidar(
            self.build_lidar(),
            [{"model": mesh_path, "location": [10.0, 0.0, 0.0]}],
        )

        self.assertIsInstance(hits, np.ndarray)
        self.assertEqual(
            hits.dtype.names,
            ("positions", "origins", "directions", "distance", "phi", "theta", "target_index"),
        )
        self.assertEqual(len(hits), 1)
        self.assertTrue(np.allclose(hits["positions"][0], [10.0, 0.1, 0.1]))
        self.assertTrue(np.allclose(hits["origins"][0], [0.0, 0.1, 0.1]))
        self.assertTrue(np.allclose(hits["directions"][0], [-1.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(hits["distance"][0]), 10.0, places=8)
        self.assertEqual(int(hits["target_index"][0]), 0)

    def test_sim_lidar_excludes_misses(self):
        mesh_path = self.write_temp_stl()
        lidar = {
            "position": [0.0, 0.1, 0.1],
            "phi": np.array([0.0, 180.0]),
            "theta": np.array([90.0]),
        }

        hits = sim_lidar(lidar, [{"model": mesh_path, "location": [10.0, 0.0, 0.0]}])

        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(float(hits["phi"][0]), 0.0, places=8)

    def test_sim_lidar_frame_time_applies_linear_motion(self):
        mesh_path = self.write_temp_stl()
        hits = sim_lidar(
            self.build_lidar(),
            [
                {
                    "model": mesh_path,
                    "location": [10.0, 0.0, 0.0],
                    "speed": [2.0, 0.0, 0.0],
                }
            ],
            frame_time=1.5,
        )

        self.assertEqual(len(hits), 1)
        self.assertTrue(np.allclose(hits["positions"][0], [13.0, 0.1, 0.1]))
        self.assertAlmostEqual(float(hits["distance"][0]), 13.0, places=8)

    def test_sim_lidar_returns_empty_array_for_empty_scan(self):
        mesh_path = self.write_temp_stl()
        lidar = {
            "position": [0.0, 0.1, 0.1],
            "phi": np.array([]),
            "theta": np.array([90.0]),
        }

        hits = sim_lidar(lidar, [{"model": mesh_path, "location": [10.0, 0.0, 0.0]}])

        self.assertEqual(len(hits), 0)
        self.assertEqual(
            hits.dtype.names,
            ("positions", "origins", "directions", "distance", "phi", "theta", "target_index"),
        )

    def test_sim_lidar_multi_hit_preserves_scan_order_and_reflected_direction(self):
        mesh_path = self.write_temp_stl()
        lidar = self.build_multi_hit_lidar()
        hits = sim_lidar(
            lidar,
            [
                {"model": mesh_path, "location": [10.0, -1.0, 0.0]},
                {"model": mesh_path, "location": [10.0, 0.0, 0.0]},
                {"model": mesh_path, "location": [10.0, 1.0, 0.0]},
            ],
        )

        self.assertEqual(len(hits), 3)
        self.assertTrue(np.allclose(hits["positions"][:, 1], [-1.0, 0.0, 1.0], atol=1e-6))
        self.assertTrue(np.array_equal(hits["target_index"], [0, 1, 2]))
        self.assertTrue(np.allclose(hits["phi"], lidar["phi"], atol=1e-8))
        self.assertTrue(np.allclose(hits["theta"], [90.0, 90.0, 90.0], atol=1e-8))

        incident = hits["positions"] - hits["origins"]
        incident /= np.linalg.norm(incident, axis=1, keepdims=True)
        normal = np.array([1.0, 0.0, 0.0])
        expected_reflected = incident - 2.0 * np.sum(
            incident * normal, axis=1, keepdims=True
        ) * normal
        self.assertTrue(np.allclose(hits["directions"], expected_reflected, atol=1e-8))
        self.assertTrue(
            np.allclose(
                hits["distance"],
                np.linalg.norm(hits["positions"] - hits["origins"], axis=1),
                atol=1e-8,
            )
        )


if __name__ == "__main__":
    unittest.main()
