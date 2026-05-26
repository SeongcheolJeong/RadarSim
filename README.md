# RadarSimPy Whitebox Workspace

This repository is a documentation-first workspace for turning the packaged `radarsimpy_origin` blackbox distribution into a clean-room, whitebox implementation target.

## Current State

- Version baseline is locked to `15.1.0` from `radarsimpy_origin/__init__.py`.
- An executable whitebox package now exists at `radarsimpy_whitebox/`.
- A vendor macOS arm runtime is now available locally at `radarsimpy_macos_arm/`.
- A canonical local import alias now exists at `radarsimpy -> radarsimpy_macos_arm` for vendor-runtime capture on this host.
- Readable source is limited to the Python layer:
  - `radarsimpy_origin/__init__.py`
  - `radarsimpy_origin/transmitter.py`
  - `radarsimpy_origin/receiver.py`
  - `radarsimpy_origin/radar.py`
  - `radarsimpy_origin/processing.py`
  - `radarsimpy_origin/tools.py`
  - `radarsimpy_origin/mesh_kit.py`
- Core simulator and licensing paths are binary-only Windows artifacts:
  - `radarsimpy_origin/simulator.cp3xx-win_amd64.pyd`
  - `radarsimpy_origin/license.cp3xx-win_amd64.pyd`
  - `radarsimpy_origin/radarsimcpp.dll`

## Clean-Room Principles

- Do not treat this project as a binary reconstruction effort.
- Use public behavior, Python-visible contracts, and oracle comparisons as the source of truth.
- Keep the blackbox package as an oracle until each whitebox subsystem matches the required contract.
- Separate static specifications from runtime-verified vendor oracle evidence; current runtime capture exists for the macOS arm build, while Windows-origin capture is still pending.

## Document Map

- [`docs/spec/WHITEBOX_STRATEGY.md`](docs/spec/WHITEBOX_STRATEGY.md): goals, non-goals, migration order, backend strategy
- [`docs/spec/API_INVENTORY.md`](docs/spec/API_INVENTORY.md): public package surface and model member inventory
- [`docs/spec/DATA_CONTRACTS.md`](docs/spec/DATA_CONTRACTS.md): shapes, dtypes, units, ordering, result contracts
- [`docs/spec/PARITY_MATRIX.md`](docs/spec/PARITY_MATRIX.md): parity status tracker for APIs and scenarios
- [`docs/spec/SCENARIO_CATALOG.md`](docs/spec/SCENARIO_CATALOG.md): parity scenarios and acceptance checks
- [`docs/spec/IMPLEMENTATION_ROADMAP.md`](docs/spec/IMPLEMENTATION_ROADMAP.md): staged execution plan
- [`docs/spec/RUNTIME_GAPS.md`](docs/spec/RUNTIME_GAPS.md): remaining runtime gaps plus host-specific oracle constraints
- [`docs/spec/DECISIONS.md`](docs/spec/DECISIONS.md): ADR log for early project decisions

## User Docs

- [`docs/user/README_KR.md`](docs/user/README_KR.md): whitebox user-document entry point in Korean
- [`docs/user/INSTALL_GUIDE_KR.md`](docs/user/INSTALL_GUIDE_KR.md): installation and environment setup for the current repository
- [`docs/user/USAGE_GUIDE_KR.md`](docs/user/USAGE_GUIDE_KR.md): practical whitebox usage guide with runnable examples
- [`docs/user/OFFICIAL_DOCS_COMPATIBILITY_KR.md`](docs/user/OFFICIAL_DOCS_COMPATIBILITY_KR.md): what can and cannot be reused from the official RadarSimPy documentation

## Source Evidence

- `Initial_plan.rtf` is the initial strategic brief for the whitebox effort.
- `radarsimpy_origin/__init__.py` defines package version, top-level exports, and package examples.
- The Python model and signal-processing modules define the best currently visible contracts.
- `radarsimpy_origin` binary simulator and license modules still require Windows oracle capture, while an expanded vendor oracle bundle now exists for `radarsimpy_macos_arm`.

## Executable Coverage

- `radarsimpy_whitebox.processing` is now an independent whitebox implementation of the readable signal-processing layer.
- `radarsimpy_whitebox.tools` is now an independent whitebox implementation of the readable analysis utility layer.
- `radarsimpy_whitebox.Transmitter`, `radarsimpy_whitebox.Receiver`, and `radarsimpy_whitebox.Radar` are independently implemented whitebox model classes.
- `radarsimpy_whitebox.mesh_kit` provides a dependency-light mesh ingestion path with builtin ASCII STL and OBJ support.
- `radarsimpy_whitebox.sim_radar` now provides a reference simulator for ideal point targets and static triangular mesh targets that returns `baseband`, `noise`, `timestamp`, and `interference` (`None` when no interferer is present); single-channel positive- and negative-radial-speed Doppler, phase-tagged, off-axis MIMO spatial-phase, positive- and negative-speed off-axis moving MIMO joint phase-Doppler, captured MIMO, captured MIMO multi-frame, and positive- and negative-speed off-axis moving MIMO multi-frame point-target parity are exact against the macOS arm oracle, while the accepted multi-target baselines now include a static-static range-profile contract, a static-plus-moving local range-Doppler plus correlation contract, a static-plus-negative-moving local range-Doppler plus correlation contract, signed static-plus-static-plus-moving three-target separation contracts, signed static-plus-static-plus-moving-plus-moving four-target separation contracts, an off-axis MIMO static-plus-moving joint contract that adds channel-phase parity at sample index `2`, an off-axis MIMO static-plus-negative-moving joint contract that keeps the same sample-index phase rule with reversed Doppler, signed off-axis MIMO static-plus-static-plus-moving three-target joint contracts that add per-channel dual-static peak parity with the same sample-index phase rule, an off-axis MIMO static-plus-static-plus-moving-plus-moving four-target joint contract that adds per-channel dual-moving peak parity on top of the same sample-index phase rule, an off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target joint contract that splits one moving peak into positive and one into negative Doppler while keeping the same phase rule, an off-axis MIMO static-plus-static-plus-moving-plus-moving multi-frame joint contract that adds exact frame-block flattening on top of that four-target rule, signed off-axis MIMO static-plus-static-plus-moving multi-frame contracts that add exact frame-block flattening on top of that joint rule, and both signed off-axis MIMO multi-frame two-target mixed-scene contracts that add exact frame-block flattening, and mesh-target parity is tracked by timestamp/shape/dtype agreement plus waveform-correlation checks.
- `radarsimpy_whitebox.sim_lidar` now provides a structured-array reference path for mesh point clouds, with per-hit `positions`, `origins`, `directions`, `distance`, scan angles, and target indices; the vendor-backed LiDAR baseline now covers baseline, moving, and multi-hit scans, and the `directions` field follows the captured reflected-ray convention.
- `radarsimpy_whitebox.sim_rcs` provides a geometry-backed far-field triangular facet reference path for mesh RCS sweeps; current vendor-backed parity covers scalar plate cases exactly enough for the accepted tolerance, a qualitative angle-sweep main-lobe contract, and a polarization-discrimination contract where co-polar returns dominate cross-polar cases.
- `radarsimpy_whitebox.modulation` provides TDM/BPM-style pulse modulation helpers that generate `pulse_amp` and `pulse_phs` arrays for transmitter channel dictionaries.
- `radarsimpy_whitebox` now also implements the top-level utility and compatibility surface for metadata, diagnostics, and licensing: `get_version`, `get_info`, `print_info`, `check_installation`, `hello`, `set_license`, `is_licensed`, and `get_license_info`.
- `oracle_capture.run_oracle_capture` can now target canonical vendor package names on macOS, and the stored vendor bundle at `oracle_capture_output/macos_arm_py311/` now includes single-channel positive- and negative-radial-speed Doppler, phase-tagged, off-axis MIMO, positive- and negative-speed off-axis moving MIMO, MIMO, MIMO multi-frame, positive- and negative-speed off-axis moving MIMO multi-frame, static-static two-target point, static-plus-moving two-target point, static-plus-negative-moving two-target point, signed static-plus-static-plus-moving three-target point, signed static-plus-static-plus-moving-plus-moving four-target points in both Doppler directions, off-axis MIMO static-plus-moving two-target point, off-axis MIMO static-plus-negative-moving two-target point, signed off-axis MIMO static-plus-static-plus-moving three-target point, off-axis MIMO static-plus-static-plus-moving-plus-moving four-target point, off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target point, off-axis MIMO static-plus-static-plus-moving-plus-moving four-target multi-frame point, signed off-axis MIMO static-plus-static-plus-moving multi-frame three-target point, off-axis MIMO static-plus-moving multi-frame two-target point, off-axis MIMO static-plus-negative-moving multi-frame two-target point, mesh-target `sim_radar`, baseline, moving, and multi-hit `sim_lidar`, scalar and polarization `sim_rcs` angle cases, and an `RCS-002` observation-angle sweep artifact.
- `oracle_capture.run_oracle_capture` now also writes a machine-readable `scenario_index` into `manifest.json`, linking captured artifacts directly to scenario IDs such as `SIM-PT-004`, `SIM-MESH-002`, and `LIDAR-003`.
- `oracle_capture.run_oracle_compare` now compares `radarsimpy_whitebox` against the captured macOS arm oracle, writes a machine-readable parity report at `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json`, and includes per-scenario `scenario_results`.
- The current simulator scope is still intentionally limited: interference radar input and oracle-verified mesh or ray-tracing parity remain pending later phases.
- `tests/test_phase1_source_parity.py` validates full signature parity for the phase-1 APIs, checks that `processing` and `tools` are no longer source-backed, and smoke-tests representative numerical behavior against independently loaded origin source modules.
- `tests/test_phase2_model_parity.py` validates constructor signatures, model dict contracts, timestamp behavior, phase noise, and motion validation against independently loaded origin source modules.
- `tests/test_phase3_point_target_simulator.py` validates the `sim_radar` contract, dry-run behavior, seed reproducibility, requested point-target phase rotation, positive- and negative-radial-speed Doppler-bin behavior, off-axis MIMO channel-phase behavior, positive- and negative-speed off-axis moving MIMO joint phase-Doppler behavior, positive- and negative-speed off-axis moving MIMO multi-frame behavior, MIMO channel-count shape rules, MIMO multi-frame flattening, a two-target range-separation baseline, static-plus-moving, static-plus-negative-moving, signed static-plus-static-plus-moving, and signed static-plus-static-plus-moving-plus-moving range-Doppler baselines, off-axis MIMO static-plus-moving and static-plus-negative-moving mixed-scene baselines, signed off-axis MIMO static-plus-static-plus-moving three-target mixed-scene baselines, an off-axis MIMO static-plus-static-plus-moving-plus-moving four-target mixed-scene baseline, an off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target mixed-scene baseline, an off-axis MIMO static-plus-static-plus-moving-plus-moving four-target multi-frame baseline, an off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target multi-frame baseline, an accepted-band off-axis MIMO static-plus-static-plus-negative-moving-plus-negative-moving four-target multi-frame baseline, signed off-axis MIMO static-plus-static-plus-moving multi-frame baselines, signed off-axis MIMO multi-frame mixed-scene baselines, a range-peak sanity check, and static mesh-target acceptance plus aspect sensitivity.
- `tests/test_phase4_mesh_rcs.py` validates builtin mesh loading, scalar `sim_rcs` behavior, cross-polar quietness, and the current whitebox observation-angle sweep shape contract.
- `tests/test_phase5_lidar.py` validates `sim_lidar` signature, structured-array hit fields, hit filtering, `frame_time`-driven target motion, and multi-hit reflected-direction ordering.
- `tests/test_phase6_package_surface.py` validates metadata constants plus `get_version`, `get_info`, `print_info`, `check_installation`, and `hello`.
- `tests/test_phase7_vendor_oracle_parity.py` validates single-channel point-target, positive- and negative-radial-speed moving-point Doppler, phase-tagged point-target, off-axis MIMO spatial phase, positive- and negative-speed off-axis moving MIMO joint phase-Doppler, MIMO point-target, MIMO multi-frame point-target, positive- and negative-speed off-axis moving MIMO multi-frame point-target, static-static, static-plus-moving, static-plus-negative-moving, signed static-plus-static-plus-moving, signed static-plus-static-plus-moving-plus-moving, off-axis MIMO static-plus-moving, off-axis MIMO static-plus-negative-moving, signed off-axis MIMO static-plus-static-plus-moving, an off-axis MIMO static-plus-static-plus-moving-plus-moving four-target joint contract, an off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target joint contract, an off-axis MIMO static-plus-static-plus-moving-plus-moving four-target multi-frame joint contract, an off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target multi-frame joint contract, an accepted-band off-axis MIMO static-plus-static-plus-negative-moving-plus-negative-moving four-target multi-frame contract, signed off-axis MIMO static-plus-static-plus-moving multi-frame contracts, and signed off-axis MIMO multi-frame two-target point contracts, mesh-target, and mesh-aspect `sim_radar`, baseline, moving, and multi-hit `sim_lidar`, the captured scalar and polarization `sim_rcs` angle cases, and the accepted `RCS-002` sweep-shape contract against the macOS arm vendor oracle.
- `tests/test_phase8_license_surface.py` validates the local license compatibility surface, including default initialization, explicit license-file loading, and missing-path behavior.
- `tests/test_phase9_modulation_helpers.py` validates TDM/BPM code generation, channel-dict application, package exports, and the expected `sim_radar` pulse-phase effect.
- `tests/test_oracle_compare_suite.py` validates the oracle-vs-whitebox comparison harness and now confirms a full match across the currently captured extended macOS arm scenarios, including scenario-indexed results.
- `tests/test_oracle_capture_suite.py` validates that the oracle-capture harness writes a stable manifest, scenario index, and loadable artifacts against `radarsimpy_whitebox`.
- Full cross-platform simulator parity is still not claimed; current runtime verification is limited to the captured macOS arm vendor bundle at `oracle_capture_output/macos_arm_py311/`, but the latest compare report now shows a full match across that captured 40-scenario set.

## One-Line Priority Order

Specify and match `processing/tools` first, then `Transmitter/Receiver/Radar`, then `sim_radar` point targets, then mesh and RCS, then LiDAR, and only then optimize backends.
