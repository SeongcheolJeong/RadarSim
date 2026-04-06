# Implementation Roadmap

## Roadmap Summary

Implementation order is fixed to reduce ambiguity and isolate parity failures:

1. `processing` and `tools`
2. `Transmitter`, `Receiver`, and `Radar`
3. `sim_radar` point-target behavior
4. mesh, ray tracing, and `sim_rcs`
5. `sim_lidar`
6. optimized backends

Each phase depends on the contracts and scenarios already recorded in this repository.

## Phase 1: `processing` And `tools`

### Entry Conditions

- `API_INVENTORY.md` covers every public function in `processing.py` and `tools.py`
- `SCENARIO_CATALOG.md` contains at least `PROC-*` and `TOOL-*` targets

### Deliverables

- Whitebox implementations for deterministic FFT, CFAR, DoA, and ROC utilities
- Unit and parity tests for all `PROC-*` and `TOOL-*` scenarios
- `PARITY_MATRIX.md` rows moved from `Specified` to `Whitebox matched`

### Completion Criteria

- signatures match exactly
- default values match exactly
- invalid-input behavior matches documented static behavior
- numerical parity is tight enough that these utilities can serve as a later simulator validation layer

## Phase 2: Model Layer

### Entry Conditions

- `DATA_CONTRACTS.md` for `Transmitter`, `Receiver`, and `Radar` is complete
- scenarios `MODEL-001` through `MODEL-006` are ready to implement

### Deliverables

- Whitebox `Transmitter`, `Receiver`, and `Radar`
- constructor validation parity
- property dictionary parity
- timestamp, virtual-array, and motion handling parity

### Completion Criteria

- all documented dict keys exist and have the expected shape
- channel ordering is stable
- single-frame and multi-frame timestamp behavior matches contract
- motion validation and error conditions match the documented rules

## Phase 3: `sim_radar` Point Targets

### Entry Conditions

- model layer is `Whitebox matched`
- Windows oracle capture exists for a minimal `sim_radar` run

### Deliverables

- reference whitebox path for point targets only
- parity fixtures for `SIM-PT-001` through `SIM-PT-003`
- documented result-key contract promoted from inferred to verified where possible

### Completion Criteria

- output dictionary keys are verified
- `baseband` shape and channel ordering match
- seeded runs reproduce stable outputs on a fixed baseline
- phase, amplitude, timestamp, and basic noise behavior match within agreed tolerance

## Phase 4: Mesh, Ray Tracing, And `sim_rcs`

### Entry Conditions

- mesh utilities are already matched
- point-target `sim_radar` baseline is stable
- Windows oracle capture exists for at least one mesh and one RCS case

### Deliverables

- reference geometry and intersection layer
- scene handoff from mesh loading to simulator-facing structures
- first whitebox RCS implementation path

### Completion Criteria

- mesh ingestion contracts remain exact
- simulator-facing geometry conventions are stable
- RCS acceptance is defined by scenario-specific numeric or curve-based parity

## Phase 5: `sim_lidar`

### Entry Conditions

- shared geometry path exists
- at least one LiDAR oracle scenario has been captured on Windows

### Deliverables

- LiDAR result contract
- reference LiDAR whitebox path
- parity scenarios for basic point-cloud generation

### Completion Criteria

- output container shape and key set are verified
- hit geometry is stable within agreed tolerance
- shared scene primitives are reused rather than duplicated

## Phase 6: Optimized Backends

### Entry Conditions

- reference whitebox behavior is already stable
- parity tests exist for every optimized subsystem

### Deliverables

- `numpy` reference backend retained as correctness baseline
- optional CPU-optimized backend
- optional compiled or GPU backend

### Completion Criteria

- optimized paths do not change public APIs
- optimized paths pass the same parity suite as the reference backend
- performance work does not weaken observability or testability

## Cross-Phase Rules

- Do not advance a subsystem before its contract and scenario are recorded.
- Do not optimize a subsystem before it is matched in a reference implementation.
- Update `DECISIONS.md` whenever a new compatibility policy or tolerance rule is introduced.
