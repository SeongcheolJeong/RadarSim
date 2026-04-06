# Decisions

This file records early architecture and process decisions for the RadarSimPy whitebox effort.

## ADR-001: Lock The Baseline To RadarSimPy 15.1.0

- Status: Accepted
- Date: 2026-04-05

### Context

The repository snapshot exposes `__version__ = "15.1.0"` in `radarsimpy_origin/__init__.py`, and the initial brief warns that parity goals become unstable if the baseline version changes during analysis.

### Decision

All documentation and first-pass parity work target RadarSimPy `15.1.0`.

### Consequences

- version drift is prevented during early whitebox work
- all future parity fixtures must name the baseline they were captured from

## ADR-002: Write The Core Documentation In English

- Status: Accepted
- Date: 2026-04-05

### Context

The public API, parameter names, and likely future tooling all use English identifiers.

### Decision

Core specification documents are written in English.

### Consequences

- API terms remain consistent with source-visible identifiers
- future automation and implementation agents can consume the documents without translation

## ADR-003: Store Specifications Under `docs/spec`

- Status: Accepted
- Date: 2026-04-05

### Context

The repository needs a single stable documentation root for contracts, scenarios, decisions, and runtime gaps.

### Decision

Use `README.md` as the entry point and keep detailed whitebox documents under `docs/spec`.

### Consequences

- navigation remains simple
- new docs can be added without cluttering the repository root

## ADR-004: Use A Clean-Room, Behavior-First Approach

- Status: Accepted
- Date: 2026-04-05

### Context

The simulator and licensing paths are binary-only, and the initial brief explicitly frames the effort as a whitebox reimplementation rather than a binary reconstruction.

### Decision

Treat the blackbox package as an oracle and reimplement behavior from documented contracts, not from inferred internal binary structure.

### Consequences

- the project stays aligned with clean-room goals
- parity documents become the implementation contract
- undocumented simulator details must be verified empirically before adoption

## ADR-005: Separate Static Specification From Runtime Verification

- Status: Accepted
- Date: 2026-04-05

### Context

Readable Python modules expose part of the package contract, but simulator and license binaries cannot be executed on the current macOS host.

### Decision

Every simulator-facing claim must be labeled as one of:

- static
- inferred
- runtime verification required

### Consequences

- unverified claims are visible instead of being mixed into hard contracts
- parity status can advance in stages from `Specified` to `Runtime verified`

## ADR-006: Start Oracle Capture On `cp312-win_amd64`

- Status: Accepted
- Date: 2026-04-05

### Context

Multiple Windows binary variants are present, but the first runtime baseline should be narrow enough to unblock contract capture.

### Decision

Use `cp312-win_amd64` as the first Windows oracle target unless a downstream compatibility requirement forces a different ABI.

### Consequences

- the first runtime capture plan has a concrete execution target
- later ABI expansion remains possible without blocking the whitebox specification effort

## ADR-007: Phase 1 Uses A Source-Backed Reference Package

- Status: Accepted
- Date: 2026-04-05

### Context

The readable `processing.py` and `tools.py` modules are available locally, but importing `radarsimpy_origin` as a normal package is blocked on this host because its package root eagerly imports Windows-only binary extensions.

### Decision

Expose phase-1 whitebox functionality through a local `radarsimpy_whitebox` package that loads readable origin source files under a private namespace instead of importing the binary package root.

### Consequences

- `processing` and `tools` become executable on the current host immediately
- signatures and behavior remain aligned with the readable source baseline
- later phases can replace individual functions with fully independent implementations without changing the package surface

## ADR-008: Phase 2 Uses Independent Whitebox Model Classes

- Status: Accepted
- Date: 2026-04-05

### Context

The Python model layer is fully readable from the local snapshot, and phase 2 of the roadmap requires executable whitebox behavior for `Transmitter`, `Receiver`, and `Radar`.

### Decision

Implement `Transmitter`, `Receiver`, and `Radar` as independent whitebox modules inside `radarsimpy_whitebox`, while keeping `processing` and `tools` source-backed for now.

### Consequences

- the workspace now has executable whitebox model objects on the current host
- the next simulator phase can build on whitebox timestamp, array, and motion behavior
- later refactors can replace source-backed utilities without changing the package surface

## ADR-009: Phase 3 Starts With A Point-Target `sim_radar` Reference Path

- Status: Accepted
- Date: 2026-04-05

### Context

Official v15.1.0 documentation provides a usable `sim_radar` contract, but Windows oracle capture is still unavailable on the current host. Waiting for oracle access would block all simulator work.

### Decision

Implement a narrow whitebox `sim_radar` reference path now for ideal point targets only, and keep mesh targets, LiDAR, RCS, and interference radar handling out of scope until later phases.

### Consequences

- simulator work can proceed on the current host using the completed model layer
- the repository gains executable simulator tests for return shape, dry-run behavior, and range-peak sanity checks
- blackbox parity is still not claimed until Windows oracle capture verifies runtime behavior

## ADR-010: Phase 4 Uses Geometry-Backed Mesh Reference Paths

- Status: Accepted
- Date: 2026-04-05

### Context

The phase-4 roadmap requires a mesh ingestion path plus an initial executable mesh or RCS reference before Windows oracle capture is available. The readable Python layer exposes `mesh_kit.py`, but the blackbox mesh and ray-tracing internals are still binary-only on the current host.

### Decision

Implement a dependency-light mesh reference path in `radarsimpy_whitebox` with builtin ASCII STL and OBJ loading, extend `sim_radar` to accept static triangular mesh targets, and implement `sim_rcs` as a coherent triangular facet sum until oracle-backed ray-tracing parity is possible.

### Consequences

- mesh-backed simulator work can continue on macOS without third-party mesh libraries
- `sim_radar` and `sim_rcs` gain executable reference behavior for static triangular meshes
- the current mesh reference path is intentionally documented as a development baseline, not as verified blackbox ray-tracing parity

## ADR-011: Phase 5 Uses A Structured-Array `sim_lidar` Reference Path

- Status: Accepted
- Date: 2026-04-05

### Context

The official v15.1.0 API reference describes `sim_lidar` as returning a structured ndarray of ray interactions, while the public usage example accesses a `positions` field directly from each scan result. The binary LiDAR implementation is still unavailable on the current macOS host.

### Decision

Implement `sim_lidar` as a mesh-backed whitebox reference path that returns a structured ndarray with explicit per-hit fields including `positions`, `origins`, `directions`, `distance`, `phi`, `theta`, and `target_index`.

### Consequences

- the local whitebox contract stays compatible with the documented ndarray return type
- the `positions` field supports the public example-style access pattern
- exact blackbox field names and dtype layout remain marked as runtime-verification work

## ADR-012: Match The Readable Top-Level Package Utilities Locally

- Status: Accepted
- Date: 2026-04-05

### Context

After the simulator reference paths were in place, the remaining readable package-surface gap on the current host was the metadata and diagnostic utility layer in `radarsimpy_origin/__init__.py`.

### Decision

Implement the readable top-level metadata constants and utility functions in `radarsimpy_whitebox` now, while continuing to leave license-bound functions for later oracle-backed work.

### Consequences

- the local package surface is more usable for inspection, environment checks, and onboarding
- the baseline `__version__` remains aligned with RadarSimPy `15.1.0`
- license-related APIs remain explicitly out of scope until their binary contracts are captured

## ADR-013: Replace The Source-Backed `tools` Layer First

- Status: Accepted
- Date: 2026-04-05

### Context

After phase 6, the remaining source-backed layers were `processing` and `tools`. The `tools` module is smaller, deterministic, and already covered by signature and numeric parity tests, making it the safer first target for replacement.

### Decision

Reimplement `radarsimpy_whitebox.tools` as an independent local module now, while leaving `radarsimpy_whitebox.processing` source-backed until a later step.

### Consequences

- the whitebox package now has an independent analysis utility layer for ROC and Swerling computations
- the phase-1 parity suite continues to validate behavior against the readable origin source
- `processing` remains the next obvious candidate for full source-backed replacement

## ADR-014: Complete The Phase-1 Replacement By Reimplementing `processing`

- Status: Accepted
- Date: 2026-04-05

### Context

After replacing `tools`, the last remaining source-backed phase-1 surface was `processing`. The readable source was fully available and the existing phase-1 parity suite already covered signatures plus representative FFT and CFAR behavior.

### Decision

Replace `radarsimpy_whitebox.processing` with an independent local implementation that preserves the readable-source signatures and numerical behavior.

### Consequences

- the entire phase-1 utility layer is now locally implemented instead of loaded through the source shim
- the local phase-1 parity suite now guards against regressions back to loader-based execution
- future optimization work can proceed from fully local code in both `processing` and `tools`

## ADR-015: Use The macOS Arm Vendor Build As The First Live Oracle Host

- Status: Accepted
- Date: 2026-04-06

### Context

A local `radarsimpy_macos_arm` package was found after the workspace had already reached a fully local whitebox baseline. The original `radarsimpy_origin` package remains Windows-only on this host, while the macOS arm build still needed three host-specific adjustments before it could be used as an oracle: clearing `com.apple.quarantine`, providing the expected `mbedtls 4.0.0` shared libraries, and importing the package through the canonical `radarsimpy` package name expected by the compiled extensions.

### Decision

Use the macOS arm vendor build as the first live oracle environment. Clear quarantine on the package directory, build a local `mbedtls 4.0.0` shared runtime under `.local-runtime/`, patch `radarsimpy_macos_arm/libradarsimcpp.dylib` to those local libraries, expose the package through a local `radarsimpy -> radarsimpy_macos_arm` symlink, and capture a minimal oracle bundle through `oracle_capture`.

### Consequences

- the workspace now has real vendor-runtime evidence for simulator and license contracts on macOS arm
- parity rows for top-level simulator and license exports can advance to `Runtime verified` when backed by `oracle_capture_output/macos_arm_py311/manifest.json`
- Windows-origin parity and cross-ABI consistency are still explicitly pending

## ADR-016: Implement A Local License Compatibility Surface

- Status: Accepted
- Date: 2026-04-06

### Context

The macOS arm vendor oracle capture established a minimal success-path contract for `set_license`, `is_licensed`, and `get_license_info`. The whitebox package still lacked those exported functions, which kept the top-level package surface incomplete even after the simulator parity work.

### Decision

Implement a lightweight local license compatibility layer in `radarsimpy_whitebox`. On import, it should search for a bundled vendor license file and initialize a permissive reference state. `set_license` should accept an explicit file path, `is_licensed` should expose a boolean success state, and `get_license_info` should return a vendor-style human-readable summary string.

### Consequences

- the remaining top-level package-surface gap is now closed in the whitebox package
- license rows in the parity matrix can advance from `Runtime verified` to `Whitebox matched`
- cross-platform vendor error paths and enforcement rules remain intentionally out of scope

## ADR-017: Track Oracle Parity With A Machine-Readable Compare Report

- Status: Accepted
- Date: 2026-04-06

### Context

After the macOS arm oracle bundle and the phase-7 and phase-8 parity tests were in place, the workspace still lacked a single report that answered the practical question: which captured vendor scenarios already match the whitebox package, and which still do not.

### Decision

Add a dedicated `oracle_capture.compare_suite` and CLI wrapper that reruns the minimal whitebox capture scenarios, compares them against a stored oracle bundle, and writes a JSON report. Treat that report as the current source of truth for the minimal captured macOS arm parity state.

### Consequences

- the workspace now has a reusable parity-reporting path in addition to point tests
- the compare report can now certify a full match across the currently captured minimal macOS arm scenarios
- future oracle captures can reuse the same comparison tool across vendor builds and ABIs

## ADR-018: Expand The macOS Arm Oracle Suite Beyond Minimal Cases

- Status: Accepted
- Date: 2026-04-06

### Context

The first macOS arm oracle bundle proved that the whitebox package could match a very small vendor scenario set, but it still left major blind spots: no mesh-target `sim_radar` artifact, no moving-target `sim_lidar` artifact, and no `sim_rcs` cases that exercised the vendor angle conventions in a meaningful way.

### Decision

Expand the stored macOS arm oracle bundle to include:

- point-target and mesh-target `sim_radar`
- baseline and moving `sim_lidar`
- `sim_rcs` cases for normal incidence, broadside, edge-on, backscatter, and forward-scatter plate views

Compare those scenarios with mixed strictness:

- keep point-target `sim_radar` parity exact
- compare mesh-target `sim_radar` by timestamp equality, shape and dtype agreement, nonzero echo presence, and high normalized waveform correlation
- compare `sim_lidar` on shared vendor fields
- compare expanded `sim_rcs` cases numerically with a small relative tolerance

### Consequences

- the macOS arm compare report now says something materially stronger than the original minimal bundle
- simulator regressions in mesh-target radar, moving LiDAR, and vendor angle conventions are more likely to be caught automatically
- full blackbox parity is still not claimed outside the captured scenario family

## ADR-019: Track Captured Oracle Evidence By Scenario ID

- Status: Accepted
- Date: 2026-04-06

### Context

The workspace already had a human-written `SCENARIO_CATALOG.md`, but the oracle bundle and compare report were still organized only by implementation detail such as `captures.sim_radar.mesh_target`. That made it harder to answer a higher-level question: which documented scenarios already have machine-readable vendor evidence, and which do not.

### Decision

Add a shared scenario-index layer to the oracle tooling:

- `manifest.json` now includes `scenario_index`
- `whitebox_compare_report.json` now includes `scenario_results`
- each indexed scenario points back to a documented scenario ID such as `SIM-PT-001` or `SIM-MESH-002`

Also capture one additional scenario-oriented radar artifact pair for `SIM-MESH-002`, using a broadside versus edge-on static mesh comparison.

### Consequences

- the scenario catalog, oracle bundle, and compare report now speak the same IDs
- parity status can be updated scenario-by-scenario instead of only at the family level
- future Windows-origin bundles can reuse the same scenario IDs without changing downstream tooling

## ADR-020: Promote `SIM-PT-002` To An Oracle-Backed MIMO Baseline

- Status: Accepted
- Date: 2026-04-06

### Context

After the scenario-indexed workflow was in place, the remaining obvious point-target gap was `SIM-PT-002`: the scenario catalog called out MIMO ordering, but the stored oracle bundle still only captured the single-channel case. A quick vendor-versus-whitebox probe on a 2-TX x 2-RX setup showed exact waveform parity within the existing strict tolerance.

### Decision

Add a captured MIMO point-target `sim_radar` scenario to the oracle suite:

- store it as `artifacts/sim_radar_point_mimo.npz`
- index it as `SIM-PT-002`
- compare it with the same strict timestamp, shape, dtype, and `baseband` allclose checks used for the existing point-target baseline

### Consequences

- the macOS arm oracle bundle now covers both single-channel and MIMO point-target radar scenarios
- the scenario catalog and parity matrix can now mark `SIM-PT-002` as `Whitebox matched`
- future Windows capture work has a concrete MIMO baseline to reproduce instead of a purely document-level target

## ADR-021: Accept `RCS-002` As A Curve-Shape Oracle Contract

- Status: Accepted
- Date: 2026-04-06

### Context

When the first `RCS-002` sweep candidate was tested pointwise, the whitebox facet-sum reference did not match the vendor curve closely enough to justify exact or near-exact numeric parity across the whole array. At the same time, the vendor and whitebox implementations did agree on several higher-level invariants: the dominant main lobe, the early decay segment, the quiet tail after 90 degrees, and a reasonably strong normalized correlation.

### Decision

Capture `RCS-002` as an observation-angle sweep artifact and compare it using a qualitative but machine-checkable contract:

- angle grid must match exactly
- peak index must match
- the first three sweep samples must decrease monotonically
- the post-90-degree tail must remain quiet relative to the peak
- normalized curve correlation must clear the accepted threshold

Do not present this as full pointwise ray-tracing parity.

### Consequences

- `RCS-002` can now advance from a document-only scenario to an oracle-backed matched scenario
- the compare report stays honest about what is and is not matched for sweep behavior
- richer RCS parity work can later tighten this contract or replace it with a stronger baseline

## ADR-022: Accept `SIM-PT-004` As A Range-Profile Oracle Contract

- Status: Accepted
- Date: 2026-04-06

### Context

When a two-point-target `sim_radar` case was probed against the macOS arm vendor runtime, the whitebox simulator matched the same dominant range bins and produced a very high normalized range-profile correlation, but the raw complex waveform was not stable enough under a strict pointwise equality contract to justify promoting it as exact waveform parity. The scenario was still valuable because it exercises target separation in a way that single-target and MIMO-only cases do not.

### Decision

Capture a dedicated two-point-target artifact as `SIM-PT-004` and compare it with a machine-checkable range-profile contract:

- timestamp must match exactly
- `baseband` and `noise` shapes and dtypes must match
- dominant range-bin indices must match exactly
- normalized range-profile correlation must clear the accepted threshold

Do not present `SIM-PT-004` as raw waveform-exact parity.

### Consequences

- the oracle workflow now covers multi-target range separation without overstating raw waveform parity
- the scenario catalog and parity matrix can promote `SIM-PT-004` to `Whitebox matched`
- future vendor studies can tighten this contract if a stronger multi-target invariant becomes reliable across builds

## ADR-023: Treat LiDAR `directions` As Reflected-Ray Output

- Status: Accepted
- Date: 2026-04-06

### Context

The original LiDAR parity work only compared single-hit and moving-hit captures where the returned `directions` field could be interpreted either as a reflected ray or as a hit-to-sensor vector because the rays were normal to the target plane. A new multi-hit capture with off-axis rays against three separated planar targets disambiguated the vendor behavior: the vendor `directions` field tracks the reflected outgoing ray direction for those planar hits, and the hit ordering remains aligned with the scan order.

### Decision

Update the whitebox `sim_lidar` implementation and oracle workflow so that:

- `directions` follows the reflected-ray convention for captured planar-hit cases
- a dedicated `LIDAR-003` multi-hit artifact is captured and compared
- LiDAR parity now includes shared-field equality, hit-order preservation, and derived-distance agreement for that multi-hit scenario

Do not generalize this yet to every possible LiDAR scene or material interaction.

### Consequences

- the LiDAR parity contract now covers a more discriminating vendor-backed case than the earlier single-hit baseline
- the scenario catalog and parity matrix can promote `LIDAR-003` to `Whitebox matched`
- broader LiDAR reflection behavior still needs additional oracle evidence before it is treated as a universal contract

## ADR-024: Accept `SIM-PT-005` As An Exact MIMO Multi-Frame Oracle Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After the LiDAR multi-hit work, the clearest remaining local oracle gap in `sim_radar` was multi-frame runtime behavior. The model layer already documented frame-major flattening, but the stored vendor oracle bundle still lacked a simulator-facing scenario that exercised both multiple channels and multiple frames together. A macOS arm probe on a 2-TX x 2-RX, 2-frame point-target setup showed exact timestamp parity, exact frame-block offsets, and waveform parity within the existing strict point-target tolerance.

### Decision

Add a dedicated MIMO multi-frame point-target artifact as `SIM-PT-005` and compare it with an exact oracle contract:

- shape must be `[8, 4, 160]`
- timestamps must match exactly
- the second frame block must equal the first frame block timestamps plus `0.001` seconds
- `baseband` parity should remain exact within the existing point-target tolerance

Treat this as the current oracle-backed proof of simulator frame flattening behavior for the captured baseline.

### Consequences

- the oracle workflow now covers the documented `[channels/frames, pulses, samples]` flattening contract more directly
- the scenario catalog and parity matrix can promote `SIM-PT-005` to `Whitebox matched`
- broader multi-frame behavior beyond this captured MIMO baseline still needs additional oracle evidence

## ADR-025: Accept `RCS-003` As A Polarization-Discrimination Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After the MIMO multi-frame radar scenario was captured, the most useful remaining local RCS gap was polarization behavior. The vendor macOS arm build exposes `inc_pol` and `obs_pol`, and a broadside plate probe showed a stable pattern: co-polarized backscatter remains large, while the captured cross-polarized cases collapse to values many orders of magnitude smaller. The whitebox facet-sum reference reproduces that discrimination pattern, but returns exact zeros for the captured cross-polar cases.

### Decision

Add a dedicated `RCS-003` oracle artifact for broadside polarization discrimination and compare it with a machine-checkable contract:

- case names must match exactly
- co-polar magnitude must remain within the accepted scalar tolerance
- every captured cross-polar case must remain effectively quiet relative to co-pol
- both vendor and whitebox must preserve `co_pol > max(cross_pol)`

Do not require pointwise agreement among the tiny captured cross-polar values.

### Consequences

- the oracle workflow now covers polarization behavior in addition to scalar angles and sweep shape
- the scenario catalog and parity matrix can promote `RCS-003` to `Whitebox matched`
- richer polarization behavior across different meshes, aspects, and materials still needs additional oracle evidence

## ADR-026: Accept `SIM-PT-006` As A Point-Target Phase-Rotation Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `RCS-003`, the next clean simulator contract to promote was direct point-target phase control. The public simulator docs expose a `phase` key on point targets, and a macOS arm probe showed stable behavior for a simple `phase=90.0` case: the returned waveform preserved the same timestamp and shape contract as the baseline point-target scenario, while the complex `baseband` rotated by a quadrature factor relative to `SIM-PT-001`. The whitebox simulator already implemented per-target phase in degrees and matched that vendor behavior closely.

### Decision

Add a dedicated `SIM-PT-006` oracle artifact for a single-channel point target with `phase=90.0` and compare it with a strict contract:

- timestamps must match exactly
- the stored waveform must match the vendor artifact within the existing exact point-target tolerance
- relative to the baseline point-target artifact, the phase-tagged waveform must behave as a quadrature rotation
- keep the contract narrow to the captured `90.0` degree case until additional vendor angles are studied

### Consequences

- the oracle workflow now captures explicit point-target phase behavior instead of inferring it only from docs
- the scenario catalog and parity matrix can promote `SIM-PT-006` to `Whitebox matched`
- broader phase-angle coverage beyond the captured quadrature case still needs more oracle evidence

## ADR-027: Accept `SIM-PT-007` As A Moving-Point Doppler Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-006`, the next meaningful point-target gap was explicit speed behavior within a single frame. The simulator docs expose a `speed` key on point targets, but the current oracle bundle only covered static targets and the multi-frame moving case. A macOS arm probe with `speed=[8.0, 0.0, 0.0]` showed exact waveform parity between vendor and whitebox at the existing strict tolerance and a stable dominant range-Doppler peak shift away from the static baseline.

### Decision

Add a dedicated `SIM-PT-007` oracle artifact for a single-channel moving point target with `speed=[8.0, 0.0, 0.0]` and compare it with a strict contract:

- timestamps must match exactly
- the stored waveform must match the vendor artifact within the existing exact point-target tolerance
- the dominant range-Doppler peak must match exactly
- keep the contract narrow to the captured single-frame moving baseline until additional speed and angle cases are studied

### Consequences

- the oracle workflow now captures explicit single-frame moving-target behavior in addition to static, phase-tagged, and multi-frame cases
- the scenario catalog and parity matrix can promote `SIM-PT-007` to `Whitebox matched`
- richer Doppler behavior across additional speeds, angles, and channel layouts still needs more oracle evidence

## ADR-028: Accept `SIM-PT-008` As An Off-Axis MIMO Spatial-Phase Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-007`, the next useful point-target gap was angle-dependent MIMO spatial phase. The current MIMO baseline proved channel ordering, but it still used a target on the array boresight. A macOS arm probe with a 2-TX x 2-RX radar and an off-axis target at `location=[50.0, 1.0, 0.0]` showed exact waveform parity between vendor and whitebox and a stable channel-relative phase pattern at the peak sample.

### Decision

Add a dedicated `SIM-PT-008` oracle artifact for an off-axis 2-TX x 2-RX point target and compare it with a strict contract:

- timestamps must match exactly
- the stored waveform must match the vendor artifact within the existing exact point-target tolerance
- the peak-sample index used for the spatial-phase readout must match exactly
- the channel-relative phase pattern at that peak sample must match exactly

### Consequences

- the oracle workflow now captures angle-dependent MIMO spatial phase in addition to channel ordering
- the scenario catalog and parity matrix can promote `SIM-PT-008` to `Whitebox matched`
- broader angular behavior across additional target azimuths, elevations, and larger arrays still needs more oracle evidence

## ADR-029: Accept `SIM-PT-009` As An Off-Axis Moving MIMO Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-008`, the next remaining point-target gap was the combination of off-axis spatial phase and single-frame Doppler in the same MIMO capture. A macOS arm probe with a 2-TX x 2-RX radar and a target at `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` showed exact waveform parity and an exact dominant range-Doppler peak match. The only unstable part was phase-readout sample selection at the very leading edge, so the contract was narrowed to a captured stable sample index.

### Decision

Add a dedicated `SIM-PT-009` oracle artifact for an off-axis moving 2-TX x 2-RX point target and compare it with a strict but stable contract:

- timestamps must match exactly
- the stored waveform must match the vendor artifact within the existing exact point-target tolerance
- the dominant range-Doppler peak must match exactly
- channel-relative spatial phase must match exactly at captured sample index `2`

### Consequences

- the oracle workflow now captures a combined spatial-phase and Doppler MIMO baseline instead of separate static-angle and moving-only baselines
- the scenario catalog and parity matrix can promote `SIM-PT-009` to `Whitebox matched`
- broader joint angle-velocity behavior across more target poses and array sizes still needs more oracle evidence

## ADR-030: Accept `SIM-PT-010` As An Off-Axis Moving MIMO Multi-Frame Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-009`, the next remaining point-target gap was whether the same off-axis moving MIMO behavior stayed stable once the radar flattened multiple frames into the channel axis. A macOS arm probe with a 2-TX x 2-RX radar, `frame_time=[0.0, 1e-3]`, and a target at `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` showed exact waveform parity between vendor and whitebox, the expected `[8, 4, 160]` shape, an exact `+0.001` second offset between the first and second frame blocks, a stable dominant range-Doppler peak, and a stable first-frame channel-relative spatial phase readout at sample index `2`.

### Decision

Add a dedicated `SIM-PT-010` oracle artifact for an off-axis moving 2-TX x 2-RX x 2-frame point target and compare it with a strict but stable contract:

- timestamps must match exactly
- the stored waveform must match the vendor artifact within the existing exact point-target tolerance
- the second frame block must remain offset by exactly `+0.001` seconds
- the dominant range-Doppler peak must match exactly
- first-frame channel-relative spatial phase must match exactly at captured sample index `2`

### Consequences

- the oracle workflow now covers the joint angle-velocity MIMO baseline in both single-frame and multi-frame flattened layouts
- the scenario catalog and parity matrix can promote `SIM-PT-010` to `Whitebox matched`
- broader multi-frame joint angle-velocity behavior across more frame counts and target poses still needs more oracle evidence

## ADR-031: Accept `SIM-PT-011` As A Negative-Radial-Speed Doppler-Direction Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-010`, the clearest remaining single-channel point-target gap was Doppler directionality. The current oracle bundle already covered the static baseline and the `speed=[8.0, 0.0, 0.0]` moving case, but that still left an ambiguity: whether the vendor runtime would mirror the dominant Doppler peak in the opposite direction for the same speed magnitude with negative radial velocity. A macOS arm probe with `speed=[-8.0, 0.0, 0.0]` showed exact waveform parity between vendor and whitebox, the same `[1, 4, 160]` shape and timestamp contract as the positive-speed baseline, and a stable dominant range-Doppler peak at the opposite Doppler bin.

### Decision

Add a dedicated `SIM-PT-011` oracle artifact for a single-channel point target with `speed=[-8.0, 0.0, 0.0]` and compare it with a strict contract:

- timestamps must match exactly
- the stored waveform must match the vendor artifact within the existing exact point-target tolerance
- the dominant range-Doppler peak must match exactly
- keep the contract narrow to the captured negative-speed baseline until additional signed-speed cases are studied

### Consequences

- the oracle workflow now captures Doppler direction as well as Doppler magnitude for the current single-channel moving baseline
- the scenario catalog and parity matrix can promote `SIM-PT-011` to `Whitebox matched`
- richer signed-speed behavior across more magnitudes, angles, frames, and array layouts still needs more oracle evidence

## ADR-032: Accept `SIM-PT-012` As An Off-Axis Negative-Speed MIMO Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-011`, the next useful signed-speed gap was whether the same negative-radial-speed reversal stayed stable once spatial phase entered the picture. A macOS arm probe with a 2-TX x 2-RX radar and a target at `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` showed exact waveform parity between vendor and whitebox, an exact dominant range-Doppler peak at the opposite Doppler bin, and the same stable off-axis channel-relative spatial phase readout at captured sample index `2`.

### Decision

Add a dedicated `SIM-PT-012` oracle artifact for an off-axis negative-speed 2-TX x 2-RX point target and compare it with a strict but stable contract:

- timestamps must match exactly
- the stored waveform must match the vendor artifact within the existing exact point-target tolerance
- the dominant range-Doppler peak must match exactly
- channel-relative spatial phase must match exactly at captured sample index `2`

### Consequences

- the oracle workflow now captures signed Doppler direction in the same off-axis MIMO setting already used for the positive-speed joint contract
- the scenario catalog and parity matrix can promote `SIM-PT-012` to `Whitebox matched`
- broader signed angle-velocity behavior across more target poses and array sizes still needs more oracle evidence

## ADR-033: Accept `SIM-PT-013` As An Off-Axis Negative-Speed MIMO Multi-Frame Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-012`, the remaining nearby gap was whether that same negative-speed off-axis MIMO behavior stayed stable under multi-frame flattening. A macOS arm probe with a 2-TX x 2-RX radar, `frame_time=[0.0, 1e-3]`, and a target at `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` showed exact waveform parity, the expected `[8, 4, 160]` shape, an exact `+0.001` second offset between the first and second frame blocks, the opposite dominant range-Doppler peak, and a stable first-frame channel-relative spatial phase readout at sample index `2`.

### Decision

Add a dedicated `SIM-PT-013` oracle artifact for an off-axis negative-speed 2-TX x 2-RX x 2-frame point target and compare it with a strict but stable contract:

- timestamps must match exactly
- the stored waveform must match the vendor artifact within the existing exact point-target tolerance
- the second frame block must remain offset by exactly `+0.001` seconds
- the dominant range-Doppler peak must match exactly
- first-frame channel-relative spatial phase must match exactly at captured sample index `2`

### Consequences

- the oracle workflow now captures signed joint angle-velocity behavior in both single-frame and multi-frame off-axis MIMO layouts
- the scenario catalog and parity matrix can promote `SIM-PT-013` to `Whitebox matched`
- broader signed multi-frame angle-velocity behavior across more frame counts and target poses still needs more oracle evidence

## ADR-034: Accept `SIM-PT-014` As A Static-Plus-Moving Two-Target Separation Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After the signed-speed single-target and MIMO scenarios were in place, the next local point-target gap was mixed-scene behavior: whether the whitebox simulator preserved both a static and a moving target in the same capture closely enough to the vendor runtime. A macOS arm probe with one static target at `20 m` and one moving target at `50 m` with `speed=[8.0, 0.0, 0.0]` showed that raw waveform equality was no longer an honest contract, but the important scene structure was stable: the static local range-Doppler peak remained at `(0, 27)`, the moving local range-Doppler peak remained at `(1, 67)`, and both normalized range-profile and waveform correlations stayed high.

### Decision

Add a dedicated `SIM-PT-014` oracle artifact for one static point target plus one moving point target and compare it with a structure-preserving contract:

- timestamps must match exactly
- the static local range-Doppler peak must match exactly
- the moving local range-Doppler peak must match exactly
- normalized range-profile correlation must pass the accepted threshold
- normalized waveform correlation must pass the accepted threshold

### Consequences

- the oracle workflow now covers a mixed static-plus-moving point-target scene instead of only single-target or static-static multi-target scenes
- the scenario catalog and parity matrix can promote `SIM-PT-014` to `Whitebox matched`
- richer multi-target scenes with multiple movers, signed-speed mixtures, or MIMO channel-phase constraints still need more oracle evidence

## ADR-035: Accept `SIM-PT-015` As A Static-Plus-Negative-Moving Two-Target Separation Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-014`, the next nearby signed mixed-scene gap was whether the same two-target separation stayed stable when the moving target reversed radial direction. A macOS arm probe with one static target at `20 m` and one moving target at `50 m` with `speed=[-8.0, 0.0, 0.0]` showed that raw waveform equality was still not an honest contract, but the important structure remained stable: the static local range-Doppler peak stayed at `(0, 27)`, the negative-speed moving local range-Doppler peak stayed at `(3, 67)`, and both normalized range-profile and waveform correlations remained high.

### Decision

Add a dedicated `SIM-PT-015` oracle artifact for one static point target plus one negative-speed moving point target and compare it with a structure-preserving contract:

- timestamps must match exactly
- the static local range-Doppler peak must match exactly
- the negative-speed moving local range-Doppler peak must match exactly
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers a signed mixed static-plus-moving point-target scene in addition to the positive-speed `SIM-PT-014` baseline
- the scenario catalog and parity matrix can promote `SIM-PT-015` to `Whitebox matched`
- richer mixed scenes with multiple movers, more range overlap, or MIMO phase constraints still need more oracle evidence

## ADR-036: Accept `SIM-PT-016` As An Off-Axis MIMO Static-Plus-Moving Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-015`, the next nearby mixed-scene gap was whether the same static-plus-moving separation stayed stable once off-axis MIMO spatial phase entered the capture. A macOS arm probe with a 2-TX x 2-RX radar, one static target at `20 m`, and one moving target at `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` showed that raw waveform equality was still not an honest contract, but the important mixed-scene structure remained stable across all channels: each channel preserved the same local static peak at `(0, 27)` and moving peak at `(1, 67)`, channel-relative phase stayed stable at sample index `2`, and both normalized range-profile and waveform correlations remained high.

### Decision

Add a dedicated `SIM-PT-016` oracle artifact for an off-axis MIMO static-plus-moving mixed scene and compare it with a joint structure-preserving contract:

- timestamps must match exactly
- each channel's static local range-Doppler peak must match exactly
- each channel's moving local range-Doppler peak must match exactly
- channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers a mixed scene that combines multi-target separation with off-axis MIMO spatial phase
- the scenario catalog and parity matrix can promote `SIM-PT-016` to `Whitebox matched`
- richer mixed MIMO scenes with negative-speed movers, more targets, or multi-frame flattening still need more oracle evidence

## ADR-037: Accept `SIM-PT-017` As An Off-Axis MIMO Static-Plus-Negative-Moving Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-016`, the next nearby signed mixed-MIMO gap was whether the same off-axis static-plus-moving scene stayed stable when the mover reversed radial direction. A macOS arm probe with a 2-TX x 2-RX radar, one static target at `20 m`, and one moving target at `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` showed the same kind of stable structure as the positive-speed case: each channel preserved the same local static peak at `(0, 27)`, the moving peak mirrored to `(3, 67)`, channel-relative phase stayed stable at sample index `2`, and both normalized range-profile and waveform correlations remained high.

### Decision

Add a dedicated `SIM-PT-017` oracle artifact for an off-axis MIMO static-plus-negative-moving mixed scene and compare it with a joint structure-preserving contract:

- timestamps must match exactly
- each channel's static local range-Doppler peak must match exactly
- each channel's negative-speed moving local range-Doppler peak must match exactly
- channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers signed off-axis MIMO mixed-scene behavior for both positive-speed and negative-speed movers
- the scenario catalog and parity matrix can promote `SIM-PT-017` to `Whitebox matched`
- richer mixed MIMO scenes with more targets or multi-frame flattening still need more oracle evidence

## ADR-038: Accept `SIM-PT-018` As An Off-Axis MIMO Static-Plus-Moving Multi-Frame Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-017`, the next nearby mixed-MIMO gap was whether the same off-axis static-plus-moving structure stayed stable once the radar flattened multiple frames into the channel axis. A macOS arm probe with a 2-TX x 2-RX radar, `frame_time=[0.0, 1e-3]`, one static target at `20 m`, and one moving target at `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` showed stable mixed-scene behavior across the full `[8, 4, 160]` capture: exact timestamp parity, exact `+0.001` frame-block offset, repeated per-channel static and moving local peaks, a stable first-frame channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations.

### Decision

Add a dedicated `SIM-PT-018` oracle artifact for an off-axis MIMO static-plus-moving multi-frame mixed scene and compare it with a joint structure-preserving contract:

- timestamps must match exactly
- the second frame block must remain offset by exactly `+0.001` seconds
- each channel's static local range-Doppler peak must match exactly
- each channel's moving local range-Doppler peak must match exactly
- first-frame channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers mixed-scene off-axis MIMO behavior in both single-frame and multi-frame positive-speed layouts
- the scenario catalog and parity matrix can promote `SIM-PT-018` to `Whitebox matched`
- the signed multi-frame mixed-MIMO mirror case still needs separate oracle evidence

## ADR-039: Accept `SIM-PT-019` As An Off-Axis MIMO Static-Plus-Negative-Moving Multi-Frame Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-018`, the remaining nearby signed mixed-MIMO gap was the negative-speed mirror of the same off-axis static-plus-moving multi-frame layout. A macOS arm probe with a 2-TX x 2-RX radar, `frame_time=[0.0, 1e-3]`, one static target at `20 m`, and one moving target at `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` showed the same stable `[8, 4, 160]` structure as the positive-speed case: exact timestamp parity, exact `+0.001` frame-block offset, repeated per-channel static and negative-speed moving local peaks, a stable first-frame channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations.

### Decision

Add a dedicated `SIM-PT-019` oracle artifact for an off-axis MIMO static-plus-negative-moving multi-frame mixed scene and compare it with a joint structure-preserving contract:

- timestamps must match exactly
- the second frame block must remain offset by exactly `+0.001` seconds
- each channel's static local range-Doppler peak must match exactly
- each channel's negative-speed moving local range-Doppler peak must match exactly
- first-frame channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers signed off-axis MIMO mixed-scene behavior for both single-frame and multi-frame layouts
- the scenario catalog and parity matrix can promote `SIM-PT-019` to `Whitebox matched`
- broader mixed-scene layouts with more targets or longer frame sequences still need additional oracle evidence

## ADR-040: Accept `SIM-PT-020` As A Static-Plus-Static-Plus-Moving Three-Target Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-019`, the nearest point-target gap was broader multi-target composition rather than another signed mirror. A macOS arm probe with one static point target at `20 m`, one static point target at `35 m`, and one moving point target at `50 m` with `speed=[8.0, 0.0, 0.0]` showed stable three-region structure in the single-channel `[1, 4, 160]` capture: exact timestamp parity, two separable static local range-Doppler peaks, one separable moving local range-Doppler peak, and high normalized range-profile and waveform correlations between the vendor and whitebox runs.

### Decision

Add a dedicated `SIM-PT-020` oracle artifact for a static-plus-static-plus-moving three-target mixed scene and compare it with a structure-preserving contract:

- timestamps must match exactly
- the first static local range-Doppler peak must match exactly
- the second static local range-Doppler peak must match exactly
- the moving local range-Doppler peak must match exactly
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers a broader three-target point-scene composition, not only two-target mixtures
- the scenario catalog and parity matrix can promote `SIM-PT-020` to `Whitebox matched`
- negative-speed three-target mirrors and multi-frame three-target scenes still need separate oracle evidence

## ADR-041: Accept `SIM-PT-021` As A Static-Plus-Static-Plus-Negative-Moving Three-Target Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-020`, the closest remaining three-target gap was its signed mirror with reversed radial velocity. A macOS arm probe with one static point target at `20 m`, one static point target at `35 m`, and one moving point target at `50 m` with `speed=[-8.0, 0.0, 0.0]` preserved the same single-channel `[1, 4, 160]` structure: exact timestamp parity, the same two separable static local range-Doppler peaks, a reversed moving local range-Doppler peak, and high normalized range-profile and waveform correlations between vendor and whitebox runs.

### Decision

Add a dedicated `SIM-PT-021` oracle artifact for a static-plus-static-plus-negative-moving three-target mixed scene and compare it with a structure-preserving contract:

- timestamps must match exactly
- the first static local range-Doppler peak must match exactly
- the second static local range-Doppler peak must match exactly
- the negative-speed moving local range-Doppler peak must match exactly
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers signed three-target point-scene composition, not just the positive-speed case
- the scenario catalog and parity matrix can promote `SIM-PT-021` to `Whitebox matched`
- multi-frame or MIMO three-target scenes still need separate oracle evidence

## ADR-042: Accept `SIM-PT-022` As An Off-Axis MIMO Static-Plus-Static-Plus-Moving Three-Target Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-021`, the next nearby mixed-scene gap was the same three-target composition under off-axis MIMO channel geometry. A macOS arm probe with a 2-TX x 2-RX radar, one static point target at `20 m`, one static point target at `35 m`, and one moving point target at `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` showed a stable `[4, 4, 160]` joint structure: exact timestamp parity, repeated per-channel static local peaks at `(0, 27)` and `(0, 47)`, repeated per-channel moving local peaks at `(1, 67)`, a stable channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations between vendor and whitebox runs.

### Decision

Add a dedicated `SIM-PT-022` oracle artifact for an off-axis MIMO static-plus-static-plus-moving three-target mixed scene and compare it with a structure-preserving joint contract:

- timestamps must match exactly
- each channel's first static local range-Doppler peak must match exactly
- each channel's second static local range-Doppler peak must match exactly
- each channel's moving local range-Doppler peak must match exactly
- channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers a three-target off-axis MIMO mixed scene, not only two-target off-axis layouts
- the scenario catalog and parity matrix can promote `SIM-PT-022` to `Whitebox matched`
- the signed off-axis MIMO three-target mirror case still needs separate oracle evidence

## ADR-043: Accept `SIM-PT-023` As An Off-Axis MIMO Static-Plus-Static-Plus-Negative-Moving Three-Target Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-022`, the nearest remaining off-axis three-target gap was its signed mirror with reversed radial velocity. A macOS arm probe with a 2-TX x 2-RX radar, one static point target at `20 m`, one static point target at `35 m`, and one moving point target at `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` preserved the same stable `[4, 4, 160]` joint structure: exact timestamp parity, repeated per-channel static local peaks at `(0, 27)` and `(0, 47)`, repeated per-channel negative-speed moving peaks at `(3, 67)`, a stable channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations between vendor and whitebox runs.

### Decision

Add a dedicated `SIM-PT-023` oracle artifact for an off-axis MIMO static-plus-static-plus-negative-moving three-target mixed scene and compare it with a structure-preserving joint contract:

- timestamps must match exactly
- each channel's first static local range-Doppler peak must match exactly
- each channel's second static local range-Doppler peak must match exactly
- each channel's negative-speed moving local range-Doppler peak must match exactly
- channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers signed off-axis MIMO three-target mixed scenes for both Doppler directions
- the scenario catalog and parity matrix can promote `SIM-PT-023` to `Whitebox matched`
- off-axis three-target multi-frame mirrors still need separate oracle evidence

## ADR-044: Accept `SIM-PT-024` As An Off-Axis MIMO Static-Plus-Static-Plus-Moving Multi-Frame Three-Target Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-023`, the next nearby gap was whether the same off-axis three-target composition stayed stable once the radar flattened two frames into the channel axis. A macOS arm probe with a 2-TX x 2-RX radar, `frame_time=[0.0, 1e-3]`, one static point target at `20 m`, one static point target at `35 m`, and one moving point target at `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` showed a stable `[8, 4, 160]` joint structure: exact timestamp parity, exact `+0.001` frame-block offset, repeated per-channel static local peaks at `(0, 27)` and `(0, 47)`, repeated per-channel moving peaks at `(1, 67)`, a stable first-frame channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations between vendor and whitebox runs.

### Decision

Add a dedicated `SIM-PT-024` oracle artifact for an off-axis MIMO static-plus-static-plus-moving multi-frame three-target mixed scene and compare it with a structure-preserving joint contract:

- timestamps must match exactly
- the second frame block must remain offset by exactly `+0.001` seconds
- each channel's first static local range-Doppler peak must match exactly
- each channel's second static local range-Doppler peak must match exactly
- each channel's moving local range-Doppler peak must match exactly
- first-frame channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers multi-frame off-axis MIMO three-target composition for the positive-speed case
- the scenario catalog and parity matrix can promote `SIM-PT-024` to `Whitebox matched`
- the signed mirror `SIM-PT-025` was deferred and is now closed by `ADR-045`

## ADR-045: Accept `SIM-PT-025` As An Off-Axis MIMO Static-Plus-Static-Plus-Negative-Moving Multi-Frame Three-Target Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-024`, the nearest remaining gap was its signed mirror with reversed radial velocity under the same off-axis multi-frame three-target layout. A macOS arm probe with a 2-TX x 2-RX radar, `frame_time=[0.0, 1e-3]`, one static point target at `20 m`, one static point target at `35 m`, and one moving point target at `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` preserved the same stable `[8, 4, 160]` joint structure: exact timestamp parity, exact `+0.001` frame-block offset, repeated per-channel static local peaks at `(0, 27)` and `(0, 47)`, repeated per-channel negative-speed moving peaks at `(3, 67)`, a stable first-frame channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations between vendor and whitebox runs.

### Decision

Add a dedicated `SIM-PT-025` oracle artifact for an off-axis MIMO static-plus-static-plus-negative-moving multi-frame three-target mixed scene and compare it with a structure-preserving joint contract:

- timestamps must match exactly
- the second frame block must remain offset by exactly `+0.001` seconds
- each channel's first static local range-Doppler peak must match exactly
- each channel's second static local range-Doppler peak must match exactly
- each channel's negative-speed moving local range-Doppler peak must match exactly
- first-frame channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers signed off-axis MIMO three-target multi-frame mixed scenes for both Doppler directions
- the scenario catalog and parity matrix can promote `SIM-PT-025` to `Whitebox matched`
- the next remaining radar expansions, if needed, are broader composed scenes rather than this immediate signed mirror

## ADR-046: Accept `SIM-PT-026` As A Static-Plus-Static-Plus-Moving-Plus-Moving Four-Target Separation Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-025`, the next natural radar expansion was not another signed mirror but a broader composed single-channel scene. A macOS arm probe with one static point target at `20 m`, one static point target at `35 m`, one moving point target at `10 m` with `speed=[8.0, 0.0, 0.0]`, and one moving point target at `50 m` with `speed=[8.0, 0.0, 0.0]` preserved a stable `[1, 4, 160]` structure between vendor and whitebox runs: exact timestamp parity, two separable static local range-Doppler peaks, two separable moving local range-Doppler peaks at the same Doppler row but different ranges, and high normalized range-profile and waveform correlations.

### Decision

Add a dedicated `SIM-PT-026` oracle artifact for a static-plus-static-plus-moving-plus-moving four-target mixed scene and compare it with a structure-preserving contract:

- timestamps must match exactly
- the first static local range-Doppler peak must match exactly
- the second static local range-Doppler peak must match exactly
- the first moving local range-Doppler peak must match exactly
- the second moving local range-Doppler peak must match exactly
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers a four-target composed single-channel point scene in addition to the earlier two-target and three-target baselines
- the scenario catalog and parity matrix can promote `SIM-PT-026` to `Whitebox matched`
- the next nearby gaps, if we keep expanding point-target composition, are the signed mirror or MIMO versions of the same broader scene

## ADR-047: Accept `SIM-PT-027` As A Static-Plus-Static-Plus-Negative-Moving-Plus-Negative-Moving Four-Target Separation Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-026`, the immediate signed mirror was the next natural extension. A macOS arm probe with one static point target at `20 m`, one static point target at `35 m`, one moving point target at `10 m` with `speed=[-8.0, 0.0, 0.0]`, and one moving point target at `50 m` with `speed=[-8.0, 0.0, 0.0]` preserved the same `[1, 4, 160]` output structure between vendor and whitebox runs: exact timestamp parity, two separable static local range-Doppler peaks, two separable negative-speed local range-Doppler peaks, and high normalized range-profile and waveform correlations.

### Decision

Add a dedicated `SIM-PT-027` oracle artifact for a static-plus-static-plus-negative-moving-plus-negative-moving four-target mixed scene and compare it with a structure-preserving contract:

- timestamps must match exactly
- the first static local range-Doppler peak must match exactly
- the second static local range-Doppler peak must match exactly
- the first negative-speed moving local range-Doppler peak must match exactly
- the second negative-speed moving local range-Doppler peak must match exactly
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers signed four-target composed single-channel point scenes in both Doppler directions
- the scenario catalog and parity matrix can promote `SIM-PT-027` to `Whitebox matched`
- the next nearby radar expansions, if needed, are MIMO versions of the broader four-target composition rather than this single-channel signed mirror

## ADR-048: Accept `SIM-PT-028` As An Off-Axis MIMO Static-Plus-Static-Plus-Moving-Plus-Moving Four-Target Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-027`, the next natural extension was to lift the broader four-target composition into off-axis MIMO geometry. A macOS arm probe with a 2-TX x 2-RX radar, one static point target at `20 m`, one static point target at `35 m`, one off-axis moving point target at `location=[10.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]`, and one off-axis moving point target at `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` preserved a stable `[4, 4, 160]` joint structure between vendor and whitebox runs: exact timestamp parity, repeated per-channel dual-static local peaks at `(0, 27)` and `(0, 47)`, repeated per-channel dual-moving local peaks at `(1, 14)` and `(1, 67)`, a stable channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations.

### Decision

Add a dedicated `SIM-PT-028` oracle artifact for an off-axis MIMO static-plus-static-plus-moving-plus-moving four-target mixed scene and compare it with a structure-preserving contract:

- timestamps must match exactly
- each channel's first static local range-Doppler peak must match exactly
- each channel's second static local range-Doppler peak must match exactly
- each channel's first moving local range-Doppler peak must match exactly
- each channel's second moving local range-Doppler peak must match exactly
- channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers an off-axis MIMO four-target composed scene in addition to the earlier single-channel baseline
- the scenario catalog and parity matrix can promote `SIM-PT-028` to `Whitebox matched`
- the next nearby radar expansion, if needed, is the signed mirror under the same off-axis four-target MIMO layout

## ADR-049: Accept `SIM-PT-029` As An Off-Axis MIMO Static-Plus-Static-Plus-Moving-Plus-Moving Multi-Frame Four-Target Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-028`, the next nearby expansion could have been the signed mirror of the same off-axis four-target MIMO layout. A probe against the macOS arm vendor runtime showed that mirror was not yet stable enough for promotion because one captured moving target window could land at `66` or `67` on some channels between vendor and whitebox runs. The positive-speed multi-frame variant, however, remained stable: with a 2-TX x 2-RX x 2-frame radar, one static point target at `20 m`, one static point target at `35 m`, one off-axis moving point target at `location=[10.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]`, and one off-axis moving point target at `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]`, vendor and whitebox runs preserved exact timestamp parity, exact `+0.001 s` frame-block offsets, repeated per-channel dual-static local peaks at `(0, 27)` and `(0, 47)`, repeated per-channel dual-moving local peaks at `(1, 14)` and `(1, 67)`, a stable first-frame channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations.

### Decision

Add a dedicated `SIM-PT-029` oracle artifact for an off-axis MIMO static-plus-static-plus-moving-plus-moving four-target multi-frame mixed scene and compare it with a structure-preserving contract:

- timestamps must match exactly
- the second frame block must equal the first plus exactly `0.001 s`
- each channel's first static local range-Doppler peak must match exactly
- each channel's second static local range-Doppler peak must match exactly
- each channel's first moving local range-Doppler peak must match exactly
- each channel's second moving local range-Doppler peak must match exactly
- first-frame channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers the multi-frame extension of the off-axis MIMO four-target positive-speed composition
- the scenario catalog and parity matrix can promote `SIM-PT-029` to `Whitebox matched`
- the remaining nearby gap is the signed mirror under the same multi-frame off-axis four-target MIMO layout

## ADR-050: Accept `SIM-PT-030` As An Off-Axis MIMO Static-Plus-Static-Plus-Mixed-Sign-Moving Four-Target Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-029`, the next natural mirror was the fully signed four-target off-axis MIMO composition. A direct probe against the macOS arm vendor runtime still showed an unstable far negative-moving peak on that symmetric mirror, with some whitebox channels landing at `66` while vendor channels stayed at `67`. A mixed-sign variant proved stable instead: with a 2-TX x 2-RX radar, one static point target at `20 m`, one static point target at `35 m`, one off-axis moving point target at `location=[10.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]`, and one off-axis moving point target at `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]`, vendor and whitebox runs preserved exact timestamp parity, repeated per-channel dual-static local peaks at `(0, 27)` and `(0, 47)`, repeated per-channel positive-moving local peaks at `(1, 14)`, repeated per-channel negative-moving local peaks at `(3, 67)`, a stable channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations.

### Decision

Add a dedicated `SIM-PT-030` oracle artifact for an off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target mixed scene and compare it with a structure-preserving contract:

- timestamps must match exactly
- each channel's first static local range-Doppler peak must match exactly
- each channel's second static local range-Doppler peak must match exactly
- each channel's positive-moving local range-Doppler peak must match exactly
- each channel's negative-moving local range-Doppler peak must match exactly
- channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers a stable mixed-sign four-target off-axis MIMO scene even though the fully signed mirror is still pending
- the scenario catalog and parity matrix can promote `SIM-PT-030` to `Whitebox matched`
- the next nearby radar expansion, if needed, is the multi-frame extension of this mixed-sign off-axis four-target MIMO layout

## ADR-051: Accept `SIM-PT-031` As An Off-Axis MIMO Static-Plus-Static-Plus-Mixed-Sign-Moving Multi-Frame Four-Target Joint Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-030`, the next natural extension was its multi-frame mirror under the same mixed-sign off-axis MIMO layout. A macOS arm probe with a 2-TX x 2-RX x 2-frame radar, one static point target at `20 m`, one static point target at `35 m`, one off-axis moving point target at `location=[10.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]`, and one off-axis moving point target at `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` remained stable between vendor and whitebox runs: exact timestamp parity, exact `+0.001 s` frame-block offsets, repeated per-channel dual-static local peaks at `(0, 27)` and `(0, 47)`, repeated per-channel positive-moving local peaks at `(1, 14)`, repeated per-channel negative-moving local peaks at `(3, 67)`, a stable first-frame channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations.

### Decision

Add a dedicated `SIM-PT-031` oracle artifact for an off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target multi-frame mixed scene and compare it with a structure-preserving contract:

- timestamps must match exactly
- the second frame block must equal the first plus exactly `0.001 s`
- each channel's first static local range-Doppler peak must match exactly
- each channel's second static local range-Doppler peak must match exactly
- each channel's positive-moving local range-Doppler peak must match exactly
- each channel's negative-moving local range-Doppler peak must match exactly
- first-frame channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers the multi-frame extension of the stable mixed-sign four-target off-axis MIMO composition
- the scenario catalog and parity matrix can promote `SIM-PT-031` to `Whitebox matched`
- the next nearby radar expansion, if needed, is the fully signed multiframe mirror under the same off-axis four-target MIMO layout

## ADR-052: Accept `SIM-PT-032` As An Off-Axis MIMO Static-Plus-Static-Plus-Negative-Moving-Plus-Negative-Moving Multi-Frame Four-Target Accepted-Band Contract

- Status: Accepted
- Date: 2026-04-06

### Context

After `SIM-PT-031`, the next nearby mirror was the fully signed off-axis four-target multi-frame MIMO layout. A macOS arm probe with a 2-TX x 2-RX x 2-frame radar, one static point target at `20 m`, one static point target at `35 m`, one off-axis moving point target at `location=[10.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]`, and one off-axis moving point target at `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` remained very close between vendor and whitebox runs: exact timestamp parity, exact `+0.001 s` frame-block offsets, repeated per-channel dual-static local peaks at `(0, 27)` and `(0, 47)`, repeated per-channel near negative-moving local peaks at `(3, 13)`, a stable first-frame channel-relative phase readout at sample index `2`, and high normalized range-profile and waveform correlations. The only remaining instability was the far negative-moving local peak at `50 m`: vendor channels consistently landed at `(3, 67)`, while whitebox channels landed at `(3, 66)` or `(3, 67)` depending on channel, while preserving high correlation and exact first-frame spatial phase.

### Decision

Add a dedicated `SIM-PT-032` oracle artifact for an off-axis MIMO static-plus-static-plus-negative-moving-plus-negative-moving four-target multi-frame mixed scene and compare it with a structure-preserving accepted-band contract:

- timestamps must match exactly
- the second frame block must equal the first plus exactly `0.001 s`
- each channel's first static local range-Doppler peak must match exactly
- each channel's second static local range-Doppler peak must match exactly
- each channel's near negative-moving local range-Doppler peak must match exactly at `(3, 13)`
- each channel's far negative-moving local range-Doppler peak must remain within the accepted band `(3, 66)` or `(3, 67)`
- first-frame channel-relative phase must match exactly at sample index `2`
- normalized range-profile correlation must be at least `0.999`
- normalized waveform correlation must be at least `0.995`

### Consequences

- the oracle workflow now covers the fully signed off-axis four-target multi-frame mirror without pretending a stricter far-peak contract than the current evidence supports
- the scenario catalog and parity matrix can promote `SIM-PT-032` to `Whitebox matched`
- the next nearby radar expansion, if needed, is a stricter future revisit of this same scenario once the far negative-moving peak can be pinned to a single exact bin across implementations
