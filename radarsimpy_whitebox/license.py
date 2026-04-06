"""Whitebox-compatible license surface.

This module intentionally provides a lightweight compatibility layer for the
top-level RadarSimPy license API. It does not attempt to enforce vendor
licensing rules; instead it preserves the captured package-visible contract for
reference and test use inside the whitebox workspace.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import platform
import re
from typing import Optional


_LICENSE_LINE_RE = re.compile(
    r"^(Order ID|License ID|Product|Version|Platform|Customer|Name|Purchase Date|"
    r"Expiration Date|Generated):\s*(.*)$"
)

_LICENSE_STATE = {
    "license_path": None,
    "metadata": {},
    "licensed": True,
}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _platform_name() -> str:
    return {
        "Darwin": "macOS",
        "Windows": "Windows",
        "Linux": "Linux",
    }.get(platform.system(), platform.system())


def _default_license_candidates() -> list[Path]:
    root = _workspace_root()
    candidates = []
    for package_name in (
        "radarsimpy",
        "radarsimpy_macos_arm",
        "radarsimpy_macos",
        "radarsimpy_origin",
    ):
        package_dir = root / package_name
        if not package_dir.exists():
            continue
        candidates.extend(sorted(package_dir.glob("license*.lic")))
    return candidates


def _parse_license_text(text: str) -> dict[str, str]:
    metadata = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LICENSE_LINE_RE.match(line)
        if match is None:
            continue
        metadata[match.group(1)] = match.group(2)
    return metadata


def _load_license_metadata(path: Path) -> dict[str, str]:
    metadata = _parse_license_text(path.read_text(encoding="utf-8"))
    metadata["Path"] = str(path)
    return metadata


def _fallback_metadata() -> dict[str, str]:
    return {
        "Product": "RadarSimPy",
        "Platform": "Development",
        "Name": "Whitebox Reference User",
    }


def _days_remaining(metadata: dict[str, str]) -> Optional[int]:
    expiration_text = metadata.get("Expiration Date")
    if not expiration_text:
        return None

    try:
        expiration = datetime.strptime(expiration_text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

    return max((expiration - date.today()).days, 0)


def _license_summary(metadata: dict[str, str]) -> str:
    licensed_to = metadata.get("Name") or metadata.get("Customer") or "Unknown"
    product = metadata.get("Product", "RadarSimPy")
    license_platform = metadata.get("Platform", "Development")

    lines = [
        "License Status: Licensed",
        f"Licensed to: {licensed_to}",
        f"Product: {product}",
        f"License Platform: {license_platform}",
        f"Current Platform: {_platform_name()}",
    ]

    days_remaining = _days_remaining(metadata)
    if days_remaining is not None:
        lines.append(f"Days Remaining: {days_remaining}")

    return "\n".join(lines)


def _store_state(
    license_path: Optional[Path], metadata: dict[str, str], *, licensed: bool
) -> None:
    _LICENSE_STATE["license_path"] = str(license_path) if license_path is not None else None
    _LICENSE_STATE["metadata"] = dict(metadata)
    _LICENSE_STATE["licensed"] = bool(licensed)


def set_license(license_file_path=None):
    """Load a compatible license file, or fall back to reference mode.

    Parameters
    ----------
    license_file_path : str or path-like, optional
        Explicit license path. When omitted, the workspace searches for a
        bundled RadarSimPy vendor license file and falls back to whitebox
        reference mode if none is available.
    """

    if license_file_path is None:
        candidate = next(iter(_default_license_candidates()), None)
        if candidate is None:
            _store_state(None, _fallback_metadata(), licensed=True)
            return None
        metadata = _load_license_metadata(candidate)
        _store_state(candidate, metadata, licensed=True)
        return None

    license_path = Path(license_file_path).expanduser()
    if not license_path.is_absolute():
        license_path = (Path.cwd() / license_path).resolve()

    if not license_path.exists():
        raise FileNotFoundError(f"License file not found: {license_path}")

    metadata = _load_license_metadata(license_path)
    _store_state(license_path, metadata, licensed=True)
    return None


def is_licensed() -> bool:
    """Return whether the whitebox compatibility layer is licensed."""

    return bool(_LICENSE_STATE["licensed"])


def get_license_info() -> str:
    """Return a human-readable license summary string."""

    metadata = _LICENSE_STATE["metadata"] or _fallback_metadata()
    return _license_summary(metadata)


def _initialize_default_state() -> None:
    try:
        set_license()
    except Exception:
        _store_state(None, _fallback_metadata(), licensed=True)


_initialize_default_state()
