"""Initial whitebox package for RadarSimPy.

This package is the first executable step of the whitebox roadmap. It exposes
the readable Python-layer functionality without importing the binary-only
`radarsimpy_origin` package root on unsupported hosts.

Current phase coverage:

- `processing`
- `tools`
- `Transmitter`
- `Receiver`
- `Radar`
- `mesh_kit`
- pulse modulation helpers for TDM/BPM-style channel codes
- `sim_radar`
- `sim_lidar`
- `sim_rcs`
- `set_license`
- `is_licensed`
- `get_license_info`

Later phases will replace the source-backed reference path with independent
whitebox implementations for the model and simulator layers.
"""

import platform
import sys

from . import mesh_kit
from . import modulation
from . import processing
from . import simulator
from . import tools
from .license import get_license_info, is_licensed, set_license
from .modulation import (
    apply_pulse_modulation,
    bpm,
    bpm_channels,
    bpm_code,
    tdm,
    tdm_channels,
    tdm_code,
)
from .radar import Radar
from .receiver import Receiver
from .simulator import sim_lidar, sim_radar, sim_rcs
from .transmitter import Transmitter

__version__ = "15.1.0"
__author__ = "RadarSimX"
__email__ = "info@radarsimx.com"
__url__ = "https://radarsimx.com"
__license__ = "Proprietary"
__description__ = "A comprehensive radar simulation library for Python"
__workspace_version__ = "15.1.0-phase9"

__all__ = [
    "Radar",
    "Receiver",
    "Transmitter",
    "sim_radar",
    "sim_lidar",
    "sim_rcs",
    "set_license",
    "is_licensed",
    "get_license_info",
    "apply_pulse_modulation",
    "bpm",
    "bpm_channels",
    "bpm_code",
    "tdm",
    "tdm_channels",
    "tdm_code",
    "processing",
    "simulator",
    "modulation",
    "mesh_kit",
    "tools",
    "__version__",
    "__author__",
    "__email__",
    "__url__",
    "get_version",
    "get_info",
    "print_info",
    "check_installation",
    "hello",
]


def get_version() -> str:
    """Return the current whitebox workspace version marker."""

    return __version__


def get_info():
    """Return package, platform, and dependency information."""

    info = {
        "package": "RadarSimPy",
        "version": __version__,
        "author": __author__,
        "website": __url__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "modules": {
            "radar": "Core radar system modeling",
            "transmitter": "Radar transmitter configuration",
            "receiver": "Radar receiver configuration",
            "processing": "Signal processing algorithms",
            "tools": "Analysis and characterization tools",
            "mesh_kit": "3D mesh file loading utilities",
            "modulation": "TDM/BPM pulse modulation helpers",
        },
        "simulation_engines": {
            "sim_radar": "Radar baseband simulation",
            "sim_lidar": "LiDAR point cloud simulation",
            "sim_rcs": "Radar cross-section calculation",
        },
    }

    optional_deps = {}
    try:
        import numpy as np

        optional_deps["numpy"] = np.__version__
    except ImportError:
        optional_deps["numpy"] = "Not installed"

    try:
        import scipy

        optional_deps["scipy"] = scipy.__version__
    except ImportError:
        optional_deps["scipy"] = "Not installed"

    try:
        import pymeshlab

        optional_deps["pymeshlab"] = pymeshlab.__version__
    except ImportError:
        optional_deps["pymeshlab"] = "Not installed"

    try:
        import pyvista

        optional_deps["pyvista"] = pyvista.__version__
    except ImportError:
        optional_deps["pyvista"] = "Not installed"

    info["dependencies"] = optional_deps
    return info


def print_info():
    """Print formatted package information."""

    info = get_info()

    print(f"\n{info['package']} v{info['version']}")
    print(f"Author: {info['author']}")
    print(f"Website: {info['website']}")
    print(f"Python: {info['python_version']}")
    print(f"Platform: {info['platform']}")

    print("\nCore Modules:")
    for module, description in info["modules"].items():
        print(f"  {module}: {description}")

    print("\nSimulation Engines:")
    for engine, description in info["simulation_engines"].items():
        print(f"  {engine}: {description}")

    print("\nDependencies:")
    for dep, version in info["dependencies"].items():
        if version == "Not installed":
            print(f"  {dep}: Not installed")
        else:
            print(f"  {dep}: v{version}")
    print()


def check_installation():
    """Return whether required runtime dependencies appear to be installed."""

    issues = []

    try:
        import numpy  # noqa: F401
    except ImportError:
        issues.append("NumPy is required but not installed")

    try:
        import scipy  # noqa: F401
    except ImportError:
        issues.append("SciPy is required but not installed")

    if issues:
        print("Installation Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("RadarSimPy installation appears complete")
    return True


def hello():
    """Print a short welcome message and quick-start pointers."""

    print(
        """
Welcome to RadarSimPy!

A comprehensive radar simulation library for Python.

Quick Start:
1. Create radar components:
   >>> import radarsimpy_whitebox as rs
   >>> tx = rs.Transmitter(...)
   >>> rx = rs.Receiver(...)
   >>> radar = rs.Radar(transmitter=tx, receiver=rx)

2. Run simulations:
   >>> result = rs.sim_radar(radar, targets)

3. Process results:
   >>> range_doppler = rs.processing.range_doppler_fft(result["baseband"])

Documentation: https://radarsimx.github.io/radarsimpy/
Support: info@radarsimx.com
"""
    )
