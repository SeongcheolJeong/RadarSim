"""Whitebox mesh loading utilities.

This module preserves the public helper shape of `radarsimpy_origin.mesh_kit`
while adding a builtin ASCII mesh loader so the whitebox workspace can operate
without third-party mesh dependencies.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_BUILTIN_MESH_MODULE = SimpleNamespace(__name__="builtin_ascii")


def check_module_installed(module_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            return True
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        pass

    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def safe_import(module_name: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def import_mesh_module() -> object:
    module_list = ["trimesh", "pyvista", "pymeshlab", "meshio"]

    for module_name in module_list:
        if check_module_installed(module_name):
            module = safe_import(module_name)
            if module is not None:
                return module

    return _BUILTIN_MESH_MODULE


def _deduplicate_vertices(vertices):
    point_index = {}
    points = []
    cells = []

    for triangle in vertices:
        cell = []
        for vertex in triangle:
            key = tuple(float(coord) for coord in vertex)
            if key not in point_index:
                point_index[key] = len(points)
                points.append(key)
            cell.append(point_index[key])
        cells.append(cell)

    return np.asarray(points, dtype=float), np.asarray(cells, dtype=int)


def _load_ascii_stl(mesh_file_name: str, scale: float) -> dict:
    triangles = []
    current = []

    with open(mesh_file_name, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                current.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(current) == 3:
                    triangles.append(current)
                    current = []

    if not triangles:
        raise ValueError(f"No triangles found in ASCII STL mesh: {mesh_file_name}")

    points, cells = _deduplicate_vertices(triangles)
    return {"points": points / scale, "cells": cells}


def _load_obj(mesh_file_name: str, scale: float) -> dict:
    vertices = []
    cells = []

    with open(mesh_file_name, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                indices = []
                for face_entry in parts[1:]:
                    vertex_index = int(face_entry.split("/")[0]) - 1
                    indices.append(vertex_index)
                for idx in range(1, len(indices) - 1):
                    cells.append([indices[0], indices[idx], indices[idx + 1]])

    if not vertices or not cells:
        raise ValueError(f"Incomplete OBJ mesh: {mesh_file_name}")

    return {
        "points": np.asarray(vertices, dtype=float) / scale,
        "cells": np.asarray(cells, dtype=int),
    }


def _load_builtin(mesh_file_name: str, scale: float) -> dict:
    suffix = Path(mesh_file_name).suffix.lower()
    if suffix == ".stl":
        return _load_ascii_stl(mesh_file_name, scale)
    if suffix == ".obj":
        return _load_obj(mesh_file_name, scale)

    raise ImportError(
        "Builtin mesh loader only supports ASCII STL and OBJ files. "
        "Install trimesh, pyvista, pymeshlab, or meshio for broader support."
    )


def load_mesh(mesh_file_name: str, scale: float, mesh_module: object) -> dict:
    if mesh_module.__name__ == "pyvista":
        mesh_data = mesh_module.read(mesh_file_name)
        points = np.array(mesh_data.points) / scale
        cells = mesh_data.faces.reshape(-1, 4)[:, 1:]
        return {"points": points, "cells": cells}

    if mesh_module.__name__ == "pymeshlab":
        ms = mesh_module.MeshSet()
        ms.load_new_mesh(mesh_file_name)
        mesh_data = ms.current_mesh()
        points = np.array(mesh_data.vertex_matrix()) / scale
        cells = np.array(mesh_data.face_matrix())
        if np.isfortran(points):
            points = np.ascontiguousarray(points)
            cells = np.ascontiguousarray(cells)
        ms.clear()
        return {"points": points, "cells": cells}

    if mesh_module.__name__ == "trimesh":
        mesh_data = mesh_module.load(mesh_file_name)
        points = np.array(mesh_data.vertices) / scale
        cells = np.array(mesh_data.faces)
        return {"points": points, "cells": cells}

    if mesh_module.__name__ == "meshio":
        mesh_data = mesh_module.read(mesh_file_name)
        points = mesh_data.points / scale
        cells = mesh_data.cells[0].data
        return {"points": points, "cells": cells}

    if mesh_module.__name__ == "builtin_ascii":
        return _load_builtin(mesh_file_name, scale)

    raise ImportError(
        "\nMesh Processing Module Required\n"
        "-----------------------------\n"
        "No valid module was found to process 3D model files.\n\n"
        "Please install one of the following modules:\n"
        "1. PyVista (Recommended)\n"
        "    • pip install pyvista\n"
        "2. PyMeshLab\n"
        "    • pip install pymeshlab\n"
        "3. Trimesh\n"
        "    • pip install trimesh\n"
        "4. meshio\n"
        "    • pip install meshio\n"
    )
