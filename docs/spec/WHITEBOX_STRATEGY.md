# Whitebox Strategy

## Purpose

This document defines how the RadarSimPy whitebox effort should proceed without relying on proprietary internal source. The goal is behavioral compatibility, not binary reconstruction.

## Goal

Build a clean-room implementation that preserves the public behavior of RadarSimPy `15.1.0` well enough to replace the packaged blackbox backend over time.

## Non-Goals

- Reverse engineering binary internals into source-equivalent code
- Claiming simulator semantics that are not supported by static evidence or runtime oracle capture
- Optimizing early before contracts and parity are stable

## Evidence Baseline

- Strategic brief: `Initial_plan.rtf`
- Top-level package surface: `radarsimpy_origin/__init__.py`
- Readable model layer: `radarsimpy_origin/transmitter.py`, `receiver.py`, `radar.py`
- Readable processing layer: `radarsimpy_origin/processing.py`
- Readable analysis layer: `radarsimpy_origin/tools.py`
- Readable mesh utility layer: `radarsimpy_origin/mesh_kit.py`
- Opaque simulator and license layers: `simulator*.pyd`, `license*.pyd`, `radarsimcpp.dll`

## Whitebox Operating Model

- Keep the shipped blackbox package as an oracle for later parity checks.
- Write contracts from readable Python code first.
- Mark every simulator-facing claim as either:
  - `Specified from static analysis`
  - `Inferred from package examples or initial brief`
  - `Runtime verification required`
- Treat parity as the merge criterion for implementation, not just test pass/fail.

## Migration Sequence

### Phase 1: Contract Lock

- Freeze the target package version at `15.1.0`.
- Record package exports, constructor defaults, object property dictionaries, and array ordering.
- Record which claims are proven and which still require Windows oracle capture.

### Phase 2: Processing And Tools

- Reimplement `processing.*` and `tools.*` first.
- Favor deterministic parity with very tight tolerances.
- Use these functions as early whitebox wins and as support utilities for simulator validation.

### Phase 3: Radar Model Layer

- Reimplement `Transmitter`, `Receiver`, and `Radar`.
- Preserve constructor validation, default expansion, property dictionary keys, and channel ordering.
- Lock `timestamp`, `virtual_array`, and motion semantics before touching simulator behavior.

### Phase 4: `sim_radar` Point-Target Parity

- Start with point targets only.
- Match output keys, channel ordering, shape, timestamp alignment, seed behavior, and basic noise/interference semantics.
- Delay mesh and ray-tracing fidelity until point-target parity is stable.

### Phase 5: Mesh, Ray Tracing, And RCS

- Build a reference whitebox path before optimized backends.
- Reuse scene and intersection concepts across radar and LiDAR wherever possible.
- Accept looser statistical parity where the blackbox path likely involves sampling or path-pruning heuristics.

### Phase 6: LiDAR

- Add LiDAR after shared geometry and hit-processing pieces exist.
- Prefer shared scene and intersection infrastructure over a separate engine.

### Phase 7: Optimization

- Only optimize after parity contracts are stable.
- Preserve a `numpy` reference path as the correctness baseline.

## Backend Strategy

- `numpy_ref`: first correctness reference implementation
- `numba_opt` or equivalent CPU optimization: second-stage acceleration
- `cpp_cuda` or another compiled backend: final acceleration stage after parity lock

## Required Artifacts Per Phase

- Contract update in `API_INVENTORY.md` or `DATA_CONTRACTS.md`
- Parity row updates in `PARITY_MATRIX.md`
- Scenario coverage in `SCENARIO_CATALOG.md`
- Decision log update in `DECISIONS.md` if scope or assumptions change

## Exit Criteria For The Strategy

The strategy is working when:

- public APIs are specified
- model dictionary contracts are stable
- runtime gaps are isolated
- parity scenarios exist for each subsystem
- the implementation order is constrained enough that another engineer can build against the documents without inventing major policy decisions
