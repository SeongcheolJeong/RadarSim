# Runtime Gaps

## Current Host Reality

The current workstation is macOS arm64.

- `radarsimpy_origin` still ships Windows PE binaries and cannot be executed locally.
- `radarsimpy_macos` is x86_64-only and still needs Rosetta plus an x86_64 CPython 3.10+ interpreter.
- `radarsimpy_macos_arm` is now executable locally under CPython 3.11 after clearing quarantine and patching its runtime dependency chain to a local `mbedtls 4.0.0` build.

## Observed Binary Artifacts

| Artifact | Observed Type | Role | Current State | Evidence |
| --- | --- | --- | --- | --- |
| `radarsimpy_origin/simulator.cp310-win_amd64.pyd` through `simulator.cp314-win_amd64.pyd` | `PE32+ executable (DLL), x86-64, for MS Windows` | simulator bindings | not executable on current host | local `file` inspection |
| `radarsimpy_origin/license.cp310-win_amd64.pyd` through `license.cp314-win_amd64.pyd` | `PE32+ executable (DLL), x86-64, for MS Windows` | license bindings | not executable on current host | local `file` inspection |
| `radarsimpy_origin/radarsimcpp.dll` | `PE32+ executable (DLL), x86-64, for MS Windows` | underlying C++ runtime | not executable on current host | local `file` inspection |
| `radarsimpy_origin/lib/cp_radarsimc.cp310-win_amd64.pyd` through `cp314-win_amd64.pyd` | `PE32+ executable (DLL), x86-64, for MS Windows` | auxiliary compiled extension | not executable on current host | local `file` inspection |
| `radarsimpy_macos/simulator.cpython-311-darwin.so` | `Mach-O 64-bit bundle x86_64` | macOS simulator bindings | not executable with the available arm64 Python 3.11 runtime | local `file` inspection |
| `radarsimpy_macos_arm/simulator.cpython-311-darwin.so` | `Mach-O universal (x86_64 arm64)` | macOS simulator bindings | runtime-verified on current host | local `file` inspection, `oracle_capture_output/macos_arm_py311/manifest.json` |
| `radarsimpy_macos_arm/license.cpython-311-darwin.so` | `Mach-O universal (x86_64 arm64)` | macOS license bindings | runtime-verified on current host | local `file` inspection, `oracle_capture_output/macos_arm_py311/manifest.json` |
| `radarsimpy_macos_arm/libradarsimcpp.dylib` | `Mach-O thin arm64` | macOS C++ runtime | runtime-verified after local `mbedtls 4.0.0` dependency patch | local `otool` inspection, `oracle_capture_output/macos_arm_py311/manifest.json` |

## Completed macOS Arm Oracle Capture

- Date: `2026-04-06`
- Python ABI: `CPython 3.11 arm64`
- Package root: `radarsimpy_macos_arm/`
- Canonical import: local `radarsimpy -> radarsimpy_macos_arm` symlink
- Host preparation:
  - removed `com.apple.quarantine` from `radarsimpy_macos_arm/`
  - built local shared libraries in `.local-runtime/mbedtls-4.0.0-install/`
  - patched `radarsimpy_macos_arm/libradarsimcpp.dylib` to use those local libraries
- Output bundle: `oracle_capture_output/macos_arm_py311/manifest.json`

## What Can Be Confirmed Now

- top-level package exports named in `__init__.py`
- constructor signatures and defaults in the readable Python layer
- model dictionary contracts and timestamp rules
- processing, analysis, and mesh utility contracts
- the existence of package examples such as `result['baseband']`
- official v15.1.0 simulator docs for `sim_radar`, `sim_lidar`, and `sim_rcs`
- local whitebox reference paths for `sim_radar` point and static mesh targets, `sim_lidar`, and `sim_rcs`
- a local structured-array LiDAR reference contract with hit positions, ray origins, directions, distances, and target indices
- vendor package import now succeeds on macOS arm with the captured runtime setup
- `set_license(license_file_path=None)` returns `None` on the captured vendor runtime
- `is_licensed()` returns `True` on the captured vendor runtime
- `get_license_info()` returns a human-readable license summary string on the captured vendor runtime
- minimal `sim_radar` runs return `baseband`, `noise`, `timestamp`, and `interference=None`, with arrays shaped `[1, 4, 160]`
- captured moving point-target `sim_radar` runs return the same dict contract with arrays shaped `[1, 4, 160]`, and the captured `speed=[8.0, 0.0, 0.0]` baseline shifts the dominant range-Doppler peak to the matched moving-target bin
- captured negative-radial-speed point-target `sim_radar` runs return the same dict contract with arrays shaped `[1, 4, 160]`, and the captured `speed=[-8.0, 0.0, 0.0]` baseline shifts the dominant range-Doppler peak to the opposite matched moving-target bin
- captured off-axis MIMO point-target `sim_radar` runs return the same dict contract with arrays shaped `[4, 4, 160]`, and the captured `location=[50.0, 1.0, 0.0]` baseline preserves a stable channel-relative spatial phase pattern at the peak sample
- captured off-axis moving MIMO point-target `sim_radar` runs return the same dict contract with arrays shaped `[4, 4, 160]`, and the captured `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` baseline preserves both the moving-target range-Doppler peak and a stable channel-relative spatial phase pattern at captured sample index `2`
- captured off-axis negative-speed MIMO point-target `sim_radar` runs return the same dict contract with arrays shaped `[4, 4, 160]`, and the captured `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` baseline preserves the opposite moving-target range-Doppler peak and a stable channel-relative spatial phase pattern at captured sample index `2`
- captured MIMO point-target `sim_radar` runs return the same dict contract with arrays shaped `[4, 4, 160]`
- captured MIMO multi-frame point-target `sim_radar` runs return the same dict contract with arrays shaped `[8, 4, 160]`, and frame-major timestamp blocks are separated by the exact `0.001` second frame offset
- captured off-axis moving MIMO multi-frame point-target `sim_radar` runs return the same dict contract with arrays shaped `[8, 4, 160]`, and the captured `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` baseline preserves the exact `+0.001` second frame offset, the moving-target range-Doppler peak, and a stable first-frame channel-relative spatial phase pattern at captured sample index `2`
- captured off-axis negative-speed MIMO multi-frame point-target `sim_radar` runs return the same dict contract with arrays shaped `[8, 4, 160]`, and the captured `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` baseline preserves the exact `+0.001` second frame offset, the opposite moving-target range-Doppler peak, and a stable first-frame channel-relative spatial phase pattern at captured sample index `2`
- captured phase-tagged point-target `sim_radar` runs return the same dict contract with arrays shaped `[1, 4, 160]`, and the `phase=90.0` artifact behaves as a quadrature rotation of the baseline point-target waveform
- captured two-point-target `sim_radar` runs return the same dict contract with arrays shaped `[1, 4, 160]`, and the accepted oracle contract is now dominant-range-bin equality plus high normalized range-profile correlation
- captured static-plus-moving two-point-target `sim_radar` runs return the same dict contract with arrays shaped `[1, 4, 160]`, and the accepted oracle contract is now exact local range-Doppler peak parity for the static and moving targets plus high range-profile and waveform correlation
- captured static-plus-negative-moving two-point-target `sim_radar` runs return the same dict contract with arrays shaped `[1, 4, 160]`, and the accepted oracle contract is now exact local range-Doppler peak parity for the static and negative-speed moving targets plus high range-profile and waveform correlation
- captured static-plus-static-plus-moving three-target `sim_radar` runs return the same dict contract with arrays shaped `[1, 4, 160]`, and the accepted oracle contract is now exact local range-Doppler peak parity for both static targets and the moving target plus high range-profile and waveform correlation
- captured static-plus-static-plus-negative-moving three-target `sim_radar` runs return the same dict contract with arrays shaped `[1, 4, 160]`, and the accepted oracle contract is now exact local range-Doppler peak parity for both static targets and the negative-speed moving target plus high range-profile and waveform correlation
- captured off-axis MIMO static-plus-moving `sim_radar` runs return the same dict contract with arrays shaped `[4, 4, 160]`, and the accepted oracle contract is now per-channel static and moving local range-Doppler peak parity plus exact channel-relative phase parity at sample index `2` and high range-profile and waveform correlation
- captured off-axis MIMO static-plus-negative-moving `sim_radar` runs return the same dict contract with arrays shaped `[4, 4, 160]`, and the accepted oracle contract is now per-channel static and negative-speed moving local range-Doppler peak parity plus exact channel-relative phase parity at sample index `2` and high range-profile and waveform correlation
- captured off-axis MIMO static-plus-moving multi-frame `sim_radar` runs return the same dict contract with arrays shaped `[8, 4, 160]`, and the accepted oracle contract is now exact `+0.001` frame-block flattening plus per-channel static and moving local range-Doppler peak parity, exact first-frame channel-relative phase parity at sample index `2`, and high range-profile and waveform correlation
- captured off-axis MIMO static-plus-negative-moving multi-frame `sim_radar` runs return the same dict contract with arrays shaped `[8, 4, 160]`, and the accepted oracle contract is now exact `+0.001` frame-block flattening plus per-channel static and negative-speed moving local range-Doppler peak parity, exact first-frame channel-relative phase parity at sample index `2`, and high range-profile and waveform correlation
- captured off-axis MIMO static-plus-static-plus-negative-moving multi-frame `sim_radar` runs return the same dict contract with arrays shaped `[8, 4, 160]`, and the accepted oracle contract is now exact `+0.001` frame-block flattening plus per-channel dual-static local range-Doppler peak parity, exact negative-speed moving local range-Doppler peak parity, exact first-frame channel-relative phase parity at sample index `2`, and high range-profile and waveform correlation
- captured off-axis MIMO static-plus-static-plus-negative-moving-plus-negative-moving multi-frame `sim_radar` runs return the same dict contract with arrays shaped `[8, 4, 160]`, and the accepted oracle contract is now exact `+0.001` frame-block flattening plus per-channel dual-static local range-Doppler peak parity, exact near negative-speed moving local range-Doppler peak parity, an accepted far negative-moving local band at `(3, 66)` or `(3, 67)`, exact first-frame channel-relative phase parity at sample index `2`, and high range-profile and waveform correlation
- captured off-axis MIMO static-plus-static-plus-mixed-sign-moving multi-frame `sim_radar` runs return the same dict contract with arrays shaped `[8, 4, 160]`, and the accepted oracle contract is now exact `+0.001` frame-block flattening plus per-channel dual-static local range-Doppler peak parity, exact positive- and negative-moving local range-Doppler peak parity, exact first-frame channel-relative phase parity at sample index `2`, and high range-profile and waveform correlation
- captured mesh-target `sim_radar` runs also return the same dict contract with arrays shaped `[1, 4, 160]`
- seeded `sim_radar` repeats keep `baseband` stable while `noise` differs on the captured vendor runtime
- baseline, moving, and multi-hit `sim_lidar` runs return a structured ndarray with fields `positions` and `directions`
- the captured multi-hit `sim_lidar` artifact preserves left-to-right hit ordering and indicates that vendor `directions` follow a reflected-ray convention for planar hits
- captured `sim_rcs` plate cases distinguish normal incidence, broadside, edge-on, and observation-angle variants
- captured `sim_rcs` observation-angle sweep now has an accepted main-lobe shape contract with peak-index, early-decay, quiet-tail, and normalized-correlation checks
- captured `sim_rcs` polarization cases now show strong co-pol dominance over all captured cross-pol cases
- `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` confirms that the current whitebox package matches the full currently captured extended macOS arm scenario set, including single-target, positive- and negative-radial-speed moving-point, off-axis MIMO, positive- and negative-speed off-axis moving MIMO, MIMO, MIMO multi-frame, positive- and negative-speed off-axis moving MIMO multi-frame, phase-tagged, static-static two-target, static-plus-moving two-target, static-plus-negative-moving two-target, signed static-plus-static-plus-moving three-target, signed static-plus-static-plus-moving-plus-moving four-target, off-axis MIMO static-plus-moving two-target, off-axis MIMO static-plus-negative-moving two-target, signed off-axis MIMO static-plus-static-plus-moving three-target, off-axis MIMO static-plus-static-plus-moving-plus-moving four-target, off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target, off-axis MIMO static-plus-static-plus-moving-plus-moving multi-frame four-target, off-axis MIMO static-plus-static-plus-mixed-sign-moving multi-frame four-target, accepted-band off-axis MIMO static-plus-static-plus-negative-moving-plus-negative-moving multi-frame four-target contracts, signed off-axis MIMO static-plus-static-plus-moving multi-frame three-target contracts, and signed off-axis MIMO multi-frame two-target `sim_radar`, mesh-target `sim_radar`, baseline, moving, and multi-hit `sim_lidar`, scalar and polarization `sim_rcs` cases, and the accepted `RCS-002` sweep-shape contract

## What Still Cannot Be Confirmed

- exact behavior of the Windows-only `radarsimpy_origin` binaries
- cross-ABI consistency across `cp310` through `cp314` and across macOS x86_64 versus arm64 builds
- richer simulator error text, warning behavior, and optional-key rules beyond the current captured scenarios
- whether the captured `sim_lidar` field set generalizes beyond `positions` and `directions`, and beyond the current planar reflected-direction cases
- blackbox mesh and ray-tracing parity beyond the currently captured point-target, single-plate, and single-hit cases
- how well the current `sim_rcs` match generalizes beyond the captured plate cases, polarization discrimination baseline, and accepted sweep-shape contract

## Windows Oracle Capture Plan

### Recommended First Target

- OS: Windows x86-64
- Python ABI: `cp312-win_amd64`
- Reason: the repository ships a matching `simulator.cp312-win_amd64.pyd` and `license.cp312-win_amd64.pyd`, and Python 3.12 is a practical modern default for first capture work

### Capture Tasks

| Task ID | Target | Goal | Output |
| --- | --- | --- | --- |
| `WIN-001` | import package | confirm import path, license init side effects, and root export visibility | import notes and callable inventory |
| `WIN-002` | `help()` or introspection for binary functions | extract callable signatures if available | signature notes for simulator and license exports |
| `WIN-003` | single-channel, positive- and negative-radial-speed moving-point, phase-tagged, off-axis MIMO, positive- and negative-speed off-axis moving MIMO, MIMO, MIMO multi-frame, positive- and negative-speed off-axis moving MIMO multi-frame, and multi-target point `sim_radar` runs | capture output dict keys, shapes, dtypes, channel ordering evidence, single-frame moving-target Doppler-direction evidence, off-axis spatial-phase evidence, signed joint off-axis moving MIMO evidence, signed multi-frame joint phase-Doppler evidence, frame-block flattening evidence, requested point-target phase behavior, and accepted multi-target comparator evidence including mixed static-plus-moving, static-plus-negative-moving, off-axis MIMO static-plus-moving, off-axis MIMO static-plus-negative-moving, and signed off-axis MIMO multi-frame separation | golden oracle bundle for `SIM-PT-*` |
| `WIN-004` | seeded and mesh-target `sim_radar` runs | measure reproducibility and capture mesh-contract evidence | seed-parity notes plus `SIM-MESH-*` oracle bundle |
| `WIN-005` | baseline, moving, and multi-hit `sim_lidar` runs | capture output container, key set, motion behavior, and direction-ordering evidence | golden oracle bundle for `LIDAR-*` |
| `WIN-006` | expanded `sim_rcs` angle set, polarization cases, and accepted sweep scenario | capture scalar behavior for normal, broadside, edge-on, observation-angle cases, one polarization-discrimination baseline, and one qualitative sweep baseline | golden oracle bundle for `RCS-*` |
| `WIN-007` | license query functions | capture return types and error behavior | license contract notes |

## Optional Expanded ABI Matrix

If compatibility across every shipped wheel matters, repeat the same capture set on:

- `cp310-win_amd64`
- `cp311-win_amd64`
- `cp312-win_amd64`
- `cp313-win_amd64`
- `cp314-win_amd64`

If only one oracle baseline is needed for the first whitebox iteration, use `cp312-win_amd64` first and defer the rest.

## Documentation Rule Until Windows Capture Exists

Until Windows-origin oracle capture is available:

- it is acceptable to mark simulator and license rows as `Runtime verified` when they are backed by `oracle_capture_output/macos_arm_py311/manifest.json`
- do not generalize macOS arm runtime findings to the Windows-origin binaries or every shipped ABI
- keep richer mesh or ray-tracing claims labeled as pending unless they are separately captured
- do not hard-code undocumented result keys into the whitebox implementation plan unless they are explicitly labeled as vendor-runtime evidence or local whitebox reference fields
