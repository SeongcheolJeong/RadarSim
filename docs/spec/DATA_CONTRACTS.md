# Data Contracts

## Contract Evidence Levels

- `Static`: directly visible from readable Python code
- `Official docs`: documented in the official RadarSimPy v15.1.0 site, but not yet confirmed against the local blackbox runtime
- `Inferred`: supported by package examples or `Initial_plan.rtf`, but not executable on the current host
- `Runtime verified`: confirmed against a packaged vendor runtime on the current host
- `Runtime verification required`: cannot be confirmed without additional vendor-oracle runs

## Global Conventions

| Contract | Value | Evidence |
| --- | --- | --- |
| Version baseline | `15.1.0` | Static, `radarsimpy_origin/__init__.py:76` |
| Package-level result example | `result['baseband']` exists for `sim_radar` | Static, `radarsimpy_origin/__init__.py:342` |
| Angle input units | degrees for user-facing rotation, phase, azimuth, elevation inputs | Static, model docstrings |
| Internal stored rotation units in `Radar` | radians | Static, `radarsimpy_origin/radar.py:856-859`, `:879-880` |
| Frequency units | Hz | Static, model docstrings |
| Time units | seconds | Static, model docstrings |
| Power units | dBm for RF gain and TX power, dB/Hz for phase noise PSD | Static, model docstrings |

## `Transmitter` Contract

### `rf_prop`

| Key | Type | Contract | Evidence |
| --- | --- | --- | --- |
| `tx_power` | scalar number | stored as constructor input | Static, `transmitter.py:236` |
| `pn_f` | ndarray or `None` | must be paired with `pn_power`; lengths must match when present | Static, `transmitter.py:237`, `:317-334` |
| `pn_power` | ndarray or `None` | must be paired with `pn_f`; lengths must match when present | Static, `transmitter.py:238`, `:317-334` |

### `waveform_prop`

| Key | Type | Contract | Evidence |
| --- | --- | --- | --- |
| `f` | `numpy.ndarray` | waveform frequency samples; scalar input expands to two identical points | Static, `transmitter.py:244-252`, `:306-315` |
| `t` | `numpy.ndarray` | time vector; list-like input is shifted so first sample is zero | Static, `transmitter.py:246-250`, `:252` |
| `bandwidth` | `float` | `max(f) - min(f)` | Static, `transmitter.py:253` |
| `pulse_length` | `float` | final value of normalized `t` | Static, `transmitter.py:254` |
| `pulses` | `int` | must be at least `1` | Static, `transmitter.py:229`, `:255` |
| `f_offset` | `numpy.ndarray[pulses]` | scalar expands across pulses; length must match `pulses` | Static, `transmitter.py:258-272`, `:347-351` |
| `prp` | `numpy.ndarray[pulses]` | defaults to `pulse_length` per pulse; each element must be `>= pulse_length` | Static, `transmitter.py:275-283`, `:353-362` |
| `pulse_start_time` | `numpy.ndarray[pulses]` | `cumsum(prp) - prp[0]` | Static, `transmitter.py:286` |

### `txchannel_prop`

| Key | Type | Contract | Evidence |
| --- | --- | --- | --- |
| `size` | `int` | number of TX channel dictionaries | Static, `transmitter.py:450` |
| `delay` | `numpy.ndarray[size]` | default zero | Static, `transmitter.py:453`, `:481` |
| `grid` | `numpy.ndarray[size]` | default `1.0` degree | Static, `transmitter.py:454`, `:482` |
| `locations` | `numpy.ndarray[size, 3]` | required `location` per channel | Static, `transmitter.py:455`, `:484` |
| `polarization` | `numpy.ndarray[size, 3]` | default `[0, 0, 1]` | Static, `transmitter.py:456`, `:485-487` |
| `waveform_mod` | list of dicts | each dict is `{'enabled', 'var', 't'}` | Static, `transmitter.py:459`, `:489-495`, `:365-402` |
| `pulse_mod` | `numpy.ndarray[size, pulses]` complex | per-channel pulse modulation | Static, `transmitter.py:462-465`, `:497-500`, `:404-431` |
| `az_angles` | list of ndarrays | defaults `[-90, 90]`; length must match `az_patterns` | Static, `transmitter.py:468-469`, `:503-514` |
| `az_patterns` | list of ndarrays | normalized by subtracting per-channel peak gain | Static, `transmitter.py:468-469`, `:503-514` |
| `el_angles` | list of ndarrays | defaults `[-90, 90]`; length must match `el_patterns` | Static, `transmitter.py:472-473`, `:517-530` |
| `el_patterns` | list of ndarrays | normalized by subtracting their own max | Static, `transmitter.py:472-473`, `:517-530` |
| `antenna_gains` | `numpy.ndarray[size]` | peak of original azimuth pattern | Static, `transmitter.py:476`, `:510-512` |

## `Receiver` Contract

### `rf_prop`

| Key | Type | Contract | Evidence |
| --- | --- | --- | --- |
| `rf_gain` | scalar number | stored as constructor input | Static, `receiver.py:181` |
| `noise_figure` | scalar number | stored as constructor input | Static, `receiver.py:182` |

### `bb_prop`

| Key | Type | Contract | Evidence |
| --- | --- | --- | --- |
| `fs` | scalar number | sampling rate, must be positive | Static, `receiver.py:160`, `:184`, `:235-236` |
| `load_resistor` | scalar number | must be positive | Static, `receiver.py:166`, `:185`, `:238-239` |
| `baseband_gain` | scalar number | stored as constructor input | Static, `receiver.py:186` |
| `bb_type` | string | one of `complex` or `real` | Static, `receiver.py:169`, `:187`, `:229-233` |
| `noise_bandwidth` | scalar number | equals `fs` for `complex`, `fs/2` for `real` | Static, `receiver.py:188-191` |

### `rxchannel_prop`

| Key | Type | Contract | Evidence |
| --- | --- | --- | --- |
| `size` | `int` | number of RX channel dictionaries | Static, `receiver.py:255` |
| `locations` | `numpy.ndarray[size, 3]` | required `location` per channel | Static, `receiver.py:257`, `:267` |
| `polarization` | `numpy.ndarray[size, 3]` | default `[0, 0, 1]` | Static, `receiver.py:258`, `:268-270` |
| `az_angles` | list of ndarrays | defaults `[-90, 90]`; validated against `az_patterns` | Static, `receiver.py:260-261`, `:273-280` |
| `az_patterns` | list of ndarrays | normalized by subtracting per-channel peak gain | Static, `receiver.py:260-261`, `:273-282` |
| `el_angles` | list of ndarrays | defaults `[-90, 90]`; validated against `el_patterns` | Static, `receiver.py:263-264`, `:285-294` |
| `el_patterns` | list of ndarrays | normalized by subtracting their own max | Static, `receiver.py:263-264`, `:285-294` |
| `antenna_gains` | `numpy.ndarray[size]` | peak of original azimuth pattern | Static, `receiver.py:265`, `:278-279` |

## `Radar` Contract

### Derived Sizes

| Contract | Value | Evidence |
| --- | --- | --- |
| `samples_per_pulse` | `int(transmitter.waveform_prop['pulse_length'] * receiver.bb_prop['fs'])` | Static, `radar.py:418-430` |
| `array_prop['size']` | `tx_size * rx_size` | Static, `radar.py:434-437` |
| `array_prop['virtual_array']` | `repeat(tx_locations, rx_size) + tile(rx_locations, tx_size)` | Static, `radar.py:438-446` |

### `time_prop`

| Key | Type | Contract | Evidence |
| --- | --- | --- | --- |
| `origin_timestamp` | `numpy.ndarray[ch, pulses, samples]` | built from TX delay, pulse start time, and sample time | Static, `radar.py:454`, `:556-608` |
| `origin_timestamp_shape` | tuple | shape of `origin_timestamp` | Static, `radar.py:455-456` |
| `frame_start_time` | `numpy.ndarray` | scalar or 1-D frame starts in seconds | Static, `radar.py:458` |
| `timestamp` | `numpy.ndarray` | single-frame shape equals `origin_timestamp`; multi-frame shape flattens to `[frames * channels, pulses, samples]` | Static, `radar.py:461`, `:620-643` |
| `timestamp_shape` | tuple | shape of `timestamp` | Static, `radar.py:463` |

### Timestamp Axis Ordering

| Axis | Meaning | Evidence |
| --- | --- | --- |
| axis 0 | virtual channel order, TX-major and RX-minor within each frame | Static, `radar.py:349-354`, `:592-606` |
| axis 1 | pulse index | Static, `radar.py:597-606` |
| axis 2 | ADC sample index | Static, `radar.py:590-606` |

For multi-frame runs, the frame dimension is flattened into axis 0 in frame-major blocks. The order is inferred from `np.tile(origin_timestamp, (num_frames, 1, 1))` plus broadcasted frame offsets in `radar.py:624-639`.

### `sample_prop`

| Key | Type | Contract | Evidence |
| --- | --- | --- | --- |
| `samples_per_pulse` | `int` | must be `> 0`; otherwise constructor raises `ValueError` | Static, `radar.py:418-430` |
| `noise` | scalar number | receiver noise amplitude derived from Boltzmann noise, gains, bandwidth, and load resistor | Static, `radar.py:466`, `:647-673` |
| `phase_noise` | 1-D ndarray or `None` | generated only when `pn_f` and `pn_power` exist; flattened after generation | Static, `radar.py:468-497` |

### `radar_prop`

| Key | Type | Contract | Evidence |
| --- | --- | --- | --- |
| `transmitter` | `Transmitter` | stored instance reference | Static, `radar.py:449-452` |
| `receiver` | `Receiver` | stored instance reference | Static, `radar.py:449-452` |
| `speed` | `numpy.ndarray[3]` | static motion uses constant vector; time-varying mode still stores constant vector | Static, `radar.py:820`, `:878` |
| `rotation_rate` | `numpy.ndarray[3]` radians | stored in radians | Static, `radar.py:821`, `:881` |
| `location` | either `numpy.ndarray[3]` or `numpy.ndarray[timestamp_shape + (3,)]` | time-varying arrays must match `timestamp_shape` on each component | Static, `radar.py:739-744`, `:816-842`, `:879` |
| `rotation` | either `numpy.ndarray[3]` radians or `numpy.ndarray[timestamp_shape + (3,)]` radians | time-varying arrays must match `timestamp_shape` on each component | Static, `radar.py:747-752`, `:816-859`, `:880` |

### Motion Validation Rules

| Rule | Contract | Evidence |
| --- | --- | --- |
| location length | must have 3 elements | Static, `radar.py:694-695` |
| speed length | must have 3 elements | Static, `radar.py:696-697` |
| rotation length | must have 3 elements | Static, `radar.py:698-699` |
| rotation rate length | must have 3 elements | Static, `radar.py:700-703` |
| time-varying location or rotation | requires `speed == [0,0,0]` and `rotation_rate == [0,0,0]` | Static, `radar.py:706-729` |
| time-varying arrays | each component must match `timestamp_shape` | Static, `radar.py:733-752` |
| speed arrays | not supported | Static, `radar.py:756-761` |
| rotation rate arrays | not supported | Static, `radar.py:763-768` |

## Processing Contracts

| API | Input Shape | Output Shape | Notes | Evidence |
| --- | --- | --- | --- | --- |
| `range_fft` | `[channels, pulses, adc_samples]` | `[channels, pulses, range]` | FFT on axis 2; window is tiled across channels and pulses | Static, `processing.py:34-58` |
| `doppler_fft` | `[channels, pulses, adc_samples]` | `[channels, Doppler, range]` | FFT on axis 1; window is tiled across channels and range | Static, `processing.py:61-85` |
| `range_doppler_fft` | `[channels, pulses, adc_samples]` | `[channels, Doppler, range]` | `doppler_fft(range_fft(...))` | Static, `processing.py:88-117` |
| `cfar_ca_1d` | 1-D or 2-D real array | same shape as input | complex input rejected | Static, `processing.py:120-185` |
| `cfar_ca_2d` | 1-D or 2-D real array | same shape as input | complex input rejected | Static, `processing.py:188-260` |
| `cfar_os_1d` | 1-D or 2-D real array | same shape as input | rank-based thresholding | Static, `processing.py:320-425` |
| `cfar_os_2d` | 1-D or 2-D real array | same shape as input | rank-based thresholding | Static, `processing.py:428-536` |
| `doa_*` family | covariance or beamforming arrays | angle list or spectrum-like output | exact return container should be verified during whitebox tests | Static, `processing.py:539-823` |

## Mesh Contracts

| API | Contract | Evidence |
| --- | --- | --- |
| `import_mesh_module()` | tries `trimesh`, then `pyvista`, `pymeshlab`, `meshio` | Static, `mesh_kit.py:76-82` |
| `load_mesh()` return type | dict with keys `points` and `cells` | Static, `mesh_kit.py:118`, `:130`, `:136`, `:142` |
| `points` | vertex coordinates scaled by division with `scale` | Static, `mesh_kit.py:114`, `:124`, `:134`, `:140` |
| `cells` | face indices from backend-specific mesh representation | Static, `mesh_kit.py:115`, `:125`, `:135`, `:141` |

## Simulator Result Contract

### `sim_radar`

| Contract | State | Evidence |
| --- | --- | --- |
| callable exists at package root | Static | `radarsimpy_origin/__init__.py:51`, `:91` |
| example call uses `sim_radar(radar, targets)` | Static | `radarsimpy_origin/__init__.py:22`, `:342` |
| result contains `baseband` key | Static | `radarsimpy_origin/__init__.py:342` |
| documented signature includes `density`, `level`, `interf`, `ray_filter`, `back_propagating`, `device`, `log_path`, `dry_run` | Official docs | RadarSimPy v15.1.0 simulator docs |
| point target schema includes `location`, `rcs`, `speed`, `phase` | Official docs | RadarSimPy v15.1.0 simulator docs |
| documented result includes `baseband`, `noise`, `timestamp`, and `interference` when applicable | Official docs | RadarSimPy v15.1.0 simulator docs |
| documented `baseband` shape is `[channels/frames, pulses, samples]` | Official docs | RadarSimPy v15.1.0 simulator docs |
| captured macOS arm oracle runs return dict keys `baseband`, `noise`, `timestamp`, and `interference` for both point-target and mesh-target `sim_radar` cases | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json` |
| captured macOS arm oracle runs use `complex128` for `baseband` and `noise`, `float64` for `timestamp`, with shape `[1, 4, 160]` in the stored point-target and mesh-target cases | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json` |
| captured macOS arm oracle MIMO point-target run uses the same dict-key contract with shape `[4, 4, 160]` for `baseband`, `noise`, and `timestamp` | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo.npz` |
| captured macOS arm oracle off-axis MIMO point-target run uses the same dict-key contract with shape `[4, 4, 160]`, and the accepted parity contract is exact waveform equality plus exact channel-relative spatial phase at the captured peak sample for the `location=[50.0, 1.0, 0.0]` baseline | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis moving MIMO point-target run uses the same dict-key contract with shape `[4, 4, 160]`, and the accepted parity contract is exact waveform equality plus exact dominant range-Doppler peak parity and exact channel-relative spatial phase at captured sample index `2` for the `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` baseline | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_moving.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis negative-speed MIMO point-target run uses the same dict-key contract with shape `[4, 4, 160]`, and the accepted parity contract is exact waveform equality plus exact dominant range-Doppler peak parity and exact channel-relative spatial phase at captured sample index `2` for the `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` baseline | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_moving_negative.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle MIMO multi-frame point-target run uses the same dict-key contract with shape `[8, 4, 160]`, and frame-major timestamp flattening is runtime-verified by an exact `+0.001` second offset between the first and second frame blocks | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis moving MIMO multi-frame point-target run uses the same dict-key contract with shape `[8, 4, 160]`, and the accepted parity contract is exact frame-block timestamp flattening plus exact dominant range-Doppler peak parity and exact first-frame channel-relative spatial phase at captured sample index `2` for the `location=[50.0, 1.0, 0.0]`, `speed=[8.0, 0.0, 0.0]` baseline | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_moving_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis negative-speed MIMO multi-frame point-target run uses the same dict-key contract with shape `[8, 4, 160]`, and the accepted parity contract is exact frame-block timestamp flattening plus exact dominant range-Doppler peak parity and exact first-frame channel-relative spatial phase at captured sample index `2` for the `location=[50.0, 1.0, 0.0]`, `speed=[-8.0, 0.0, 0.0]` baseline | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_moving_negative_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle moving point-target run uses the same dict-key contract with shape `[1, 4, 160]`, and the accepted parity contract is exact waveform equality plus exact dominant range-Doppler peak parity for the captured `speed=[8.0, 0.0, 0.0]` baseline | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_moving_doppler.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle negative-radial-speed point-target run uses the same dict-key contract with shape `[1, 4, 160]`, and the accepted parity contract is exact waveform equality plus exact dominant range-Doppler peak parity for the captured `speed=[-8.0, 0.0, 0.0]` baseline | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_moving_doppler_negative.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle phase-tagged point-target run uses the same dict-key contract with shape `[1, 4, 160]`, and the accepted parity contract is exact waveform equality plus a quadrature `phase=90°` rotation relative to the baseline point-target artifact | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_phase.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle two-point-target run uses the same dict-key contract with shape `[1, 4, 160]`, but accepted parity is defined on timestamp equality, dominant range-bin equality, and normalized range-profile correlation rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_multi.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle static-plus-moving two-point-target run uses the same dict-key contract with shape `[1, 4, 160]`, but accepted parity is defined on timestamp equality, a static local range-Doppler peak at `(0, 27)`, a moving local range-Doppler peak at `(1, 67)`, high normalized range-profile correlation, and high waveform correlation rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_static_moving.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle static-plus-negative-moving two-point-target run uses the same dict-key contract with shape `[1, 4, 160]`, but accepted parity is defined on timestamp equality, a static local range-Doppler peak at `(0, 27)`, a negative-speed moving local range-Doppler peak at `(3, 67)`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_static_moving_negative.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle static-plus-static-plus-moving three-target run uses the same dict-key contract with shape `[1, 4, 160]`, but accepted parity is defined on timestamp equality, static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, a moving local range-Doppler peak at `(1, 67)`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_static_static_moving.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle static-plus-static-plus-negative-moving three-target run uses the same dict-key contract with shape `[1, 4, 160]`, but accepted parity is defined on timestamp equality, static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, a negative-speed moving local range-Doppler peak at `(3, 67)`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_static_static_moving_negative.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle static-plus-static-plus-moving-plus-moving four-target run uses the same dict-key contract with shape `[1, 4, 160]`, but accepted parity is defined on timestamp equality, static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, moving local range-Doppler peaks at `(1, 14)` and `(1, 67)`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_static_static_moving_moving.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle static-plus-static-plus-negative-moving-plus-negative-moving four-target run uses the same dict-key contract with shape `[1, 4, 160]`, but accepted parity is defined on timestamp equality, static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, negative-speed moving local range-Doppler peaks at `(3, 13)` and `(3, 66)`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_static_static_moving_moving_negative.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-static-plus-moving-plus-moving four-target run uses the same dict-key contract with shape `[4, 4, 160]`, but accepted parity is defined on timestamp equality, per-channel static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, per-channel moving local range-Doppler peaks at `(1, 14)` and `(1, 67)`, exact channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_static_moving_moving.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target run uses the same dict-key contract with shape `[4, 4, 160]`, but accepted parity is defined on timestamp equality, per-channel static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, per-channel positive-moving local range-Doppler peaks at `(1, 14)`, per-channel negative-moving local range-Doppler peaks at `(3, 67)`, exact channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_static_mixed_sign.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target multi-frame run uses the same dict-key contract with shape `[8, 4, 160]`, but accepted parity is defined on timestamp equality, exact `+0.001` frame-block offset, per-channel static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, per-channel positive-moving local range-Doppler peaks at `(1, 14)`, per-channel negative-moving local range-Doppler peaks at `(3, 67)`, exact first-frame channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_static_mixed_sign_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-static-plus-negative-moving-plus-negative-moving four-target multi-frame run uses the same dict-key contract with shape `[8, 4, 160]`, but accepted parity is defined on timestamp equality, exact `+0.001` frame-block offset, per-channel static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, per-channel near negative-moving local range-Doppler peaks at `(3, 13)`, an accepted far negative-moving local range-Doppler band of `(3, 66)` or `(3, 67)`, exact first-frame channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_static_moving_moving_negative_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-static-plus-moving-plus-moving four-target multi-frame run uses the same dict-key contract with shape `[8, 4, 160]`, but accepted parity is defined on timestamp equality, exact `+0.001` frame-block offset, per-channel static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, per-channel moving local range-Doppler peaks at `(1, 14)` and `(1, 67)`, exact first-frame channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_static_moving_moving_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-moving run uses the same dict-key contract with shape `[4, 4, 160]`, but accepted parity is defined on timestamp equality, per-channel static local range-Doppler peaks at `(0, 27)`, per-channel moving local range-Doppler peaks at `(1, 67)`, exact channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_moving.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-negative-moving run uses the same dict-key contract with shape `[4, 4, 160]`, but accepted parity is defined on timestamp equality, per-channel static local range-Doppler peaks at `(0, 27)`, per-channel negative-speed moving local range-Doppler peaks at `(3, 67)`, exact channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_moving_negative.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-static-plus-moving run uses the same dict-key contract with shape `[4, 4, 160]`, but accepted parity is defined on timestamp equality, per-channel static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, per-channel moving local range-Doppler peaks at `(1, 67)`, exact channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_static_moving.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-static-plus-negative-moving run uses the same dict-key contract with shape `[4, 4, 160]`, but accepted parity is defined on timestamp equality, per-channel static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, per-channel negative-speed moving local range-Doppler peaks at `(3, 67)`, exact channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_static_moving_negative.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-static-plus-moving multi-frame run uses the same dict-key contract with shape `[8, 4, 160]`, but accepted parity is defined on timestamp equality, exact `+0.001` frame-block offset, per-channel static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, per-channel moving local range-Doppler peaks at `(1, 67)`, exact first-frame channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_static_moving_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-static-plus-negative-moving multi-frame run uses the same dict-key contract with shape `[8, 4, 160]`, but accepted parity is defined on timestamp equality, exact `+0.001` frame-block offset, per-channel static local range-Doppler peaks at `(0, 27)` and `(0, 47)`, per-channel negative-speed moving local range-Doppler peaks at `(3, 67)`, exact first-frame channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_static_moving_negative_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-moving multi-frame run uses the same dict-key contract with shape `[8, 4, 160]`, but accepted parity is defined on timestamp equality, exact `+0.001` frame-block offset, per-channel static local range-Doppler peaks at `(0, 27)`, per-channel moving local range-Doppler peaks at `(1, 67)`, exact first-frame channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_moving_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| captured macOS arm oracle off-axis MIMO static-plus-negative-moving multi-frame run uses the same dict-key contract with shape `[8, 4, 160]`, but accepted parity is defined on timestamp equality, exact `+0.001` frame-block offset, per-channel static local range-Doppler peaks at `(0, 27)`, per-channel negative-speed moving local range-Doppler peaks at `(3, 67)`, exact first-frame channel-relative phase parity at sample index `2`, normalized range-profile correlation of at least `0.999`, and waveform correlation of at least `0.995` rather than raw waveform equality | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json`, `oracle_capture_output/macos_arm_py311/artifacts/sim_radar_point_mimo_offaxis_static_moving_negative_multiframe.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| seeded repeat preserves `baseband` but not `noise` in the captured vendor runtime | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json` |
| current whitebox implementation covers ideal point targets and static triangular mesh targets | Static | `radarsimpy_whitebox/simulator.py` |
| exact dict-key presence rules beyond the current captured scenario family and full runtime error text | Runtime verification required | additional vendor-oracle runs required |

### `sim_lidar`

| Contract | State | Evidence |
| --- | --- | --- |
| callable name exists at package root | Static | `radarsimpy_origin/__init__.py:51`, `:92` |
| `sim_lidar` is documented as `sim_lidar(lidar, targets, frame_time=0)` | Official docs | RadarSimPy v15.1.0 simulator docs |
| `sim_lidar` is described as LiDAR point cloud simulation | Static and official docs | `radarsimpy_origin/__init__.py:187` |
| lidar config requires `position`, `phi`, and `theta` | Official docs | RadarSimPy v15.1.0 simulator docs |
| `theta=90°` corresponds to horizontal rays in the official usage example | Official docs | RadarSimPy lidar example page |
| captured macOS arm oracle baseline, moving, and multi-hit runs return a structured ndarray with fields `positions` and `directions` | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json` |
| captured macOS arm oracle multi-hit artifact shows that `directions` follows the reflected-ray convention for planar hits, not a simple hit-to-sensor vector | Runtime verified | `oracle_capture_output/macos_arm_py311/artifacts/sim_lidar_multi_hits.npy`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| current whitebox implementation returns a structured ndarray with fields `positions`, `origins`, `directions`, `distance`, `phi`, `theta`, `target_index` | Static | `radarsimpy_whitebox/simulator.py` |
| current whitebox implementation excludes rays that miss all targets | Static and official docs | `radarsimpy_whitebox/simulator.py`, RadarSimPy lidar example page |
| exact blackbox structured-array field coverage and dtype details beyond the current captured scenario family | Runtime verification required | additional vendor-oracle runs required |

### `sim_rcs`

| Contract | State | Evidence |
| --- | --- | --- |
| callable name exists at package root | Static | `radarsimpy_origin/__init__.py:51`, `:93` |
| `sim_rcs` is documented as `sim_rcs(targets, f, inc_phi, inc_theta, ...)` | Official docs | RadarSimPy v15.1.0 simulator docs |
| `sim_rcs` is described as radar cross-section calculation | Static and official docs | `radarsimpy_origin/__init__.py:188` |
| captured macOS arm oracle runs return scalar results for normal-incidence, broadside, edge-on, and observation-angle plate cases | Runtime verified | `oracle_capture_output/macos_arm_py311/manifest.json` |
| captured macOS arm oracle observation-angle sweep artifact stores `obs_phi` and `rcs` arrays with shape `[7]` | Runtime verified | `oracle_capture_output/macos_arm_py311/artifacts/sim_rcs_obs_phi_sweep.npz` |
| captured macOS arm oracle polarization artifact stores `case_names` and `rcs` arrays with shape `[4]`, and co-polar response dominates all captured cross-polar cases by many orders of magnitude | Runtime verified | `oracle_capture_output/macos_arm_py311/artifacts/sim_rcs_polarization_cases.npz`, `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| accepted `RCS-002` comparator uses peak-index equality, early monotonic decay, quiet post-90-degree tail, and normalized curve correlation instead of pointwise equality | Runtime verified | `oracle_capture_output/macos_arm_py311/whitebox_compare_report.json` |
| current whitebox `sim_rcs` implementation is a coherent triangular facet sum, not blackbox ray tracing | Static | `radarsimpy_whitebox/simulator.py` |
| exact blackbox result containers and error behavior beyond the current captured scenario family | Runtime verification required | additional vendor-oracle runs required |

## Open Contract Gaps

- Windows-origin `sim_radar`, `sim_lidar`, and `sim_rcs` behavior still needs oracle introspection.
- Cross-platform license behavior and error paths still need additional oracle coverage.
- Exact `doa_*` return container shapes should be captured in executable tests.
- `sim_radar` raw multi-target waveform equality outside the accepted `SIM-PT-004`, `SIM-PT-014`, `SIM-PT-015`, `SIM-PT-016`, `SIM-PT-017`, `SIM-PT-018`, and `SIM-PT-019` structural contracts still needs stronger oracle evidence.
- `sim_radar` key presence rules outside the current captured scenario family still need runtime confirmation.
- `sim_lidar` reflected-direction behavior outside the current planar-hit capture family still needs stronger oracle evidence.
- `sim_lidar` field coverage beyond `positions` and `directions` still needs runtime confirmation.
- `sim_rcs` polarization behavior beyond the current broadside plate discrimination baseline still needs stronger oracle evidence.
