# API Inventory

## Scope

This inventory covers the package-visible interface needed for the whitebox effort:

- top-level exports from `radarsimpy_origin/__init__.py`
- public constructors, properties, and methods on `Transmitter`, `Receiver`, and `Radar`
- all public functions in `processing.py`, `tools.py`, and `mesh_kit.py`

Private helpers such as `_generate_timestamp` are intentionally excluded from the package-surface inventory, but their effects are captured in `DATA_CONTRACTS.md`.

## Status Legend

- `Specified`: visible from readable Python code
- `Partial`: name is visible, but signature or behavior still needs runtime verification

## Top-Level Package Exports

| API | Kind | Signature or Shape | Defaults | Inputs | Outputs | Exceptions or Warnings | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Radar` | class export | `Radar(transmitter, receiver, frame_time=0, location=(0,0,0), speed=(0,0,0), rotation=(0,0,0), rotation_rate=(0,0,0), seed=None)` | as shown | `Transmitter`, `Receiver`, scalar or array-like motion inputs | `Radar` instance | `ValueError` on invalid sample count or motion shape | Specified | `radarsimpy_origin/radar.py:401` |
| `Transmitter` | class export | `Transmitter(f, t, tx_power=0, pulses=1, prp=None, f_offset=None, pn_f=None, pn_power=None, channels=None)` | as shown | waveform args, RF options, channel dicts | `Transmitter` instance | `ValueError` on invalid waveform, PRP, or channel modulation input | Specified | `radarsimpy_origin/transmitter.py:214` |
| `Receiver` | class export | `Receiver(fs, noise_figure=10, rf_gain=0, load_resistor=500, baseband_gain=0, bb_type='complex', channels=None)` | as shown | sampling and channel args | `Receiver` instance | `ValueError` on invalid `fs`, `bb_type`, resistor, or channel arrays | Specified | `radarsimpy_origin/receiver.py:150` |
| `sim_radar` | binary function export | `sim_radar(radar, targets, density=1, level=None, interf=None, ray_filter=None, back_propagating=False, device='gpu', log_path=None, dry_run=False)` | as shown | `Radar`, ideal point targets or mesh targets, plus ray-tracing options | documented dict with `baseband`, `noise`, `timestamp`, and optional `interference`; macOS arm oracle capture observed `interference=None`, minimal, positive- and negative-radial-speed moving-point, phase-tagged, mixed static-plus-moving, mixed static-plus-negative-moving, signed static-plus-static-plus-moving, and signed static-plus-static-plus-moving-plus-moving multi-target shapes `[1, 4, 160]`, off-axis and on-axis MIMO shape `[4, 4, 160]`, off-axis MIMO static-plus-moving, static-plus-negative-moving, signed static-plus-static-plus-moving, static-plus-static-plus-moving-plus-moving, and static-plus-static-plus-mixed-sign-moving mixed-scene shapes `[4, 4, 160]`, and both on-axis and signed-speed off-axis MIMO multi-frame shapes `[8, 4, 160]`, including signed off-axis MIMO static-plus-static-plus-moving multi-frame mixed-scene shapes `[8, 4, 160]`, an off-axis MIMO static-plus-static-plus-moving-plus-moving multi-frame mixed-scene shape `[8, 4, 160]`, an off-axis MIMO static-plus-static-plus-mixed-sign-moving multi-frame mixed-scene shape `[8, 4, 160]`, an accepted-band off-axis MIMO static-plus-static-plus-negative-moving-plus-negative-moving multi-frame mixed-scene shape `[8, 4, 160]`, and both signed off-axis MIMO multi-frame two-target mixed-scene shapes `[8, 4, 160]` | `RuntimeError` and `ValueError` documented; macOS arm oracle capture confirms the minimal call path, `dry_run`, single-frame moving-point Doppler behavior for both `speed=[8.0, 0.0, 0.0]` and `speed=[-8.0, 0.0, 0.0]`, 90-degree point-target phase rotation, off-axis MIMO spatial phase, positive- and negative-speed off-axis moving MIMO joint phase-Doppler behavior, point-target MIMO, point-target MIMO multi-frame flattening, positive- and negative-speed off-axis moving MIMO multi-frame flattening plus first-frame spatial-phase parity, mixed static-plus-moving, static-plus-negative-moving, signed static-plus-static-plus-moving, and signed static-plus-static-plus-moving-plus-moving comparator contracts based on local range-Doppler peaks plus correlation, off-axis MIMO static-plus-moving and static-plus-negative-moving comparator contracts based on per-channel local range-Doppler peaks, channel-relative phase at sample index `2`, and correlation, signed off-axis MIMO static-plus-static-plus-moving comparator contracts based on per-channel dual-static peaks, per-channel moving peaks, channel-relative phase at sample index `2`, and correlation, an off-axis MIMO static-plus-static-plus-moving-plus-moving comparator contract based on per-channel dual-static peaks, dual-moving peaks, channel-relative phase at sample index `2`, and correlation, an off-axis MIMO static-plus-static-plus-mixed-sign-moving comparator contract based on per-channel dual-static peaks, one positive-moving peak, one negative-moving peak, channel-relative phase at sample index `2`, and correlation, an off-axis MIMO static-plus-static-plus-moving-plus-moving multi-frame comparator contract that adds exact frame-block flattening plus first-frame phase parity on top of those four-target peak checks, an off-axis MIMO static-plus-static-plus-mixed-sign-moving multi-frame comparator contract that adds exact frame-block flattening plus first-frame phase parity on top of mixed-sign four-target peak checks, an accepted-band off-axis MIMO static-plus-static-plus-negative-moving-plus-negative-moving multi-frame comparator contract that adds exact frame-block flattening plus first-frame phase parity while allowing the far negative-moving local peak to land at `(3, 66)` or `(3, 67)`, signed off-axis MIMO static-plus-static-plus-moving multi-frame comparator contracts that add exact frame-block flattening plus first-frame phase parity, and signed off-axis MIMO multi-frame comparator contracts that add exact frame-block flattening, while broader error behavior is still pending | Partial | `radarsimpy_origin/__init__.py:22`, `:51`, `:186`, `:342`; official v15.1.0 simulator docs; `oracle_capture_output/macos_arm_py311/manifest.json` |
| `sim_lidar` | binary function export | `sim_lidar(lidar, targets, frame_time=0)` | `frame_time=0` | lidar config dict and mesh targets | documented structured ndarray of ray interactions; macOS arm oracle capture observed fields `positions` and `directions` for the minimal case | minimal runtime verification now exists, but exact blackbox field coverage is still scenario-dependent; local whitebox reference intentionally exposes a broader hit record | Partial | `radarsimpy_origin/__init__.py:51`, `:187`; official v15.1.0 simulator docs; `oracle_capture_output/macos_arm_py311/manifest.json` |
| `sim_rcs` | binary function export | `sim_rcs(targets, f, inc_phi, inc_theta, inc_pol=[0, 0, 1], obs_phi=None, obs_theta=None, obs_pol=None, density=1.0)` | as shown | mesh targets and observation geometry | documented float or ndarray RCS result; macOS arm oracle capture observed scalar plate cases, an observation-angle sweep artifact, and a polarization-discrimination artifact | minimal runtime verification now exists; local whitebox reference still uses a coherent triangular facet sum rather than blackbox ray tracing | Partial | `radarsimpy_origin/__init__.py:51`, `:188`; official v15.1.0 simulator docs; `oracle_capture_output/macos_arm_py311/manifest.json` |
| `set_license` | binary function export | `set_license(license_file_path=None)` | `license_file_path=None` | optional license file path | `None` | auto-called during package import and has side effects | Partial | `radarsimpy_origin/__init__.py:54`, `:68`; `oracle_capture_output/macos_arm_py311/manifest.json` |
| `is_licensed` | binary function export | `is_licensed()` | none | none | `bool` | minimal success path captured on macOS arm | Partial | `radarsimpy_origin/__init__.py:54`, `:95-97`; `oracle_capture_output/macos_arm_py311/manifest.json` |
| `get_license_info` | binary function export | `get_license_info()` | none | none | human-readable `str` license summary | minimal success path captured on macOS arm | Partial | `radarsimpy_origin/__init__.py:54`, `:95-97`; `oracle_capture_output/macos_arm_py311/manifest.json` |
| `processing` | module namespace | namespace export | n/a | n/a | module object | none from static inspection | Specified | `radarsimpy_origin/__init__.py:57`, `:99` |
| `tools` | module namespace | namespace export | n/a | n/a | module object | none from static inspection | Specified | `radarsimpy_origin/__init__.py:58`, `:100` |
| `mesh_kit` | module namespace | namespace export | n/a | n/a | module object | import-time mesh dependency required later | Specified | `radarsimpy_origin/__init__.py:61`, `:101` |
| `__version__` | constant export | string constant | `15.1.0` | none | `str` | none | Specified | `radarsimpy_origin/__init__.py:76`, `:103` |
| `__author__` | constant export | string constant | `RadarSimX` | none | `str` | none | Specified | `radarsimpy_origin/__init__.py:77`, `:104` |
| `__email__` | constant export | string constant | `info@radarsimx.com` | none | `str` | none | Specified | `radarsimpy_origin/__init__.py:78`, `:105` |
| `__url__` | constant export | string constant | `https://radarsimx.com` | none | `str` | none | Specified | `radarsimpy_origin/__init__.py:79`, `:106` |
| `get_version` | function | `get_version()` | none | none | `str` | none | Specified | `radarsimpy_origin/__init__.py:120` |
| `get_info` | function | `get_info()` | none | none | `dict` package summary | optional dependency imports handled internally | Specified | `radarsimpy_origin/__init__.py:138` |
| `print_info` | function | `print_info()` | none | none | prints to stdout, returns `None` | none from static inspection | Specified | `radarsimpy_origin/__init__.py:227` |
| `check_installation` | function | `check_installation()` | none | none | `bool` | prints issues to stdout | Specified | `radarsimpy_origin/__init__.py:268` |
| `hello` | function | `hello()` | none | none | prints to stdout, returns `None` | none from static inspection | Specified | `radarsimpy_origin/__init__.py:314` |

## `Transmitter` Public Members

| API | Kind | Signature or Shape | Defaults | Inputs | Outputs | Exceptions or Warnings | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Transmitter.frequency` | property | `frequency` | n/a | none | `numpy.ndarray` | none | Specified | `radarsimpy_origin/transmitter.py:536` |
| `Transmitter.bandwidth` | property | `bandwidth` | n/a | none | `float` | none | Specified | `radarsimpy_origin/transmitter.py:541` |
| `Transmitter.pulse_length` | property | `pulse_length` | n/a | none | `float` | none | Specified | `radarsimpy_origin/transmitter.py:546` |
| `Transmitter.num_pulses` | property | `num_pulses` | n/a | none | `int` | none | Specified | `radarsimpy_origin/transmitter.py:551` |
| `Transmitter.num_channels` | property | `num_channels` | n/a | none | `int` | none | Specified | `radarsimpy_origin/transmitter.py:556` |
| `Transmitter.channel_locations` | property | `channel_locations` | n/a | none | `numpy.ndarray` | none | Specified | `radarsimpy_origin/transmitter.py:561` |
| `Transmitter.get_channel_info` | method | `get_channel_info(channel_idx)` | none | `int` channel index | `dict` with location, polarization, delay, grid, gain, angle and pattern arrays | `IndexError` if out of range | Specified | `radarsimpy_origin/transmitter.py:565` |
| `Transmitter.__str__` | method | `__str__()` | none | none | `str` | formatting only | Specified | `radarsimpy_origin/transmitter.py:591` |
| `Transmitter.__repr__` | method | `__repr__()` | none | none | `str` | formatting only | Specified | `radarsimpy_origin/transmitter.py:600` |

## `Receiver` Public Members

| API | Kind | Signature or Shape | Defaults | Inputs | Outputs | Exceptions or Warnings | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Receiver.sampling_rate` | property | `sampling_rate` | n/a | none | `float` | none | Specified | `radarsimpy_origin/receiver.py:308` |
| `Receiver.noise_bandwidth` | property | `noise_bandwidth` | n/a | none | `float` | none | Specified | `radarsimpy_origin/receiver.py:313` |
| `Receiver.num_channels` | property | `num_channels` | n/a | none | `int` | none | Specified | `radarsimpy_origin/receiver.py:318` |
| `Receiver.channel_locations` | property | `channel_locations` | n/a | none | `numpy.ndarray` | none | Specified | `radarsimpy_origin/receiver.py:323` |
| `Receiver.get_channel_info` | method | `get_channel_info(channel_idx)` | none | `int` channel index | `dict` with location, polarization, gain, angle and pattern arrays | `IndexError` if out of range | Specified | `radarsimpy_origin/receiver.py:327` |
| `Receiver.__str__` | method | `__str__()` | none | none | `str` | formatting only | Specified | `radarsimpy_origin/receiver.py:351` |
| `Receiver.__repr__` | method | `__repr__()` | none | none | `str` | formatting only | Specified | `radarsimpy_origin/receiver.py:360` |

## `Radar` Public Members

| API | Kind | Signature or Shape | Defaults | Inputs | Outputs | Exceptions or Warnings | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Radar.set_motion` | method | `set_motion(location=(0,0,0), speed=(0,0,0), rotation=(0,0,0), rotation_rate=(0,0,0))` | as shown | scalar or timestamp-shaped motion inputs | updates `radar_prop` in place | `ValueError` on invalid lengths, unsupported time-varying speed or rotation rate, or timestamp-shape mismatch | Specified | `radarsimpy_origin/radar.py:506` |
| `Radar.num_channels` | property | `num_channels` | n/a | none | `int` | none | Specified | `radarsimpy_origin/radar.py:883` |
| `Radar.samples_per_pulse` | property | `samples_per_pulse` | n/a | none | `int` | none | Specified | `radarsimpy_origin/radar.py:888` |
| `Radar.transmitter` | property | `transmitter` | n/a | none | `Transmitter` | none | Specified | `radarsimpy_origin/radar.py:895` |
| `Radar.receiver` | property | `receiver` | n/a | none | `Receiver` | none | Specified | `radarsimpy_origin/radar.py:900` |
| `Radar.virtual_array_locations` | property | `virtual_array_locations` | n/a | none | `numpy.ndarray` | none | Specified | `radarsimpy_origin/radar.py:905` |
| `Radar.__str__` | method | `__str__()` | none | none | `str` | formatting only | Specified | `radarsimpy_origin/radar.py:911` |
| `Radar.__repr__` | method | `__repr__()` | none | none | `str` | formatting only | Specified | `radarsimpy_origin/radar.py:919` |

## `processing.py`

| API | Kind | Signature or Shape | Defaults | Inputs | Outputs | Exceptions or Warnings | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `range_fft` | function | `range_fft(data, rwin=None, n=None)` | `rwin=None`, `n=None` | baseband `data[ch, pulses, adc_samples]`, optional range window | FFT result `data[ch, pulses, range]` | none from static inspection | Specified | `radarsimpy_origin/processing.py:34` |
| `doppler_fft` | function | `doppler_fft(data, dwin=None, n=None)` | `dwin=None`, `n=None` | range profiles `data[ch, pulses, adc_samples]`, optional Doppler window | FFT result `data[ch, Doppler, range]` | none from static inspection | Specified | `radarsimpy_origin/processing.py:61` |
| `range_doppler_fft` | function | `range_doppler_fft(data, rwin=None, dwin=None, rn=None, dn=None)` | as shown | baseband cube and optional windows | range-Doppler cube | none from static inspection | Specified | `radarsimpy_origin/processing.py:88` |
| `cfar_ca_1d` | function | `cfar_ca_1d(data, guard, trailing, pfa=1e-5, axis=0, detector='squarelaw', offset=None)` | as shown | real amplitude or power input | threshold array with same shape | `ValueError` for complex input or invalid detector | Specified | `radarsimpy_origin/processing.py:120` |
| `cfar_ca_2d` | function | `cfar_ca_2d(data, guard, trailing, pfa=1e-5, detector='squarelaw', offset=None)` | as shown | real amplitude or power input | threshold array with same shape | `ValueError` for complex input, invalid detector, or zero trailing bins | Specified | `radarsimpy_origin/processing.py:188` |
| `os_cfar_threshold` | function | `os_cfar_threshold(k, n, pfa)` | none | CFAR rank and window size | `float` threshold factor | no static exception summary captured yet | Specified | `radarsimpy_origin/processing.py:263` |
| `cfar_os_1d` | function | `cfar_os_1d(data, guard, trailing, k, pfa=1e-5, axis=0, detector='squarelaw', offset=None)` | as shown | real amplitude or power input | threshold array | `ValueError` for invalid detector and other rank/window cases | Specified | `radarsimpy_origin/processing.py:320` |
| `cfar_os_2d` | function | `cfar_os_2d(data, guard, trailing, k, pfa=1e-5, detector='squarelaw', offset=None)` | as shown | real amplitude or power input | threshold array | `ValueError` for invalid detector and invalid rank/window cases | Specified | `radarsimpy_origin/processing.py:428` |
| `doa_music` | function | `doa_music(covmat, nsig, spacing=0.5, scanangles=range(-90, 91))` | as shown | covariance matrix, signal count, scan geometry | angle list or estimator output tuple | runtime parity still needed for exact return shape | Specified | `radarsimpy_origin/processing.py:539` |
| `doa_root_music` | function | `doa_root_music(covmat, nsig, spacing=0.5)` | `spacing=0.5` | covariance matrix | angle list | runtime parity still needed for exact ordering rules | Specified | `radarsimpy_origin/processing.py:589` |
| `doa_esprit` | function | `doa_esprit(covmat, nsig, spacing=0.5)` | `spacing=0.5` | covariance matrix | angle list | runtime parity still needed for exact output ordering | Specified | `radarsimpy_origin/processing.py:638` |
| `doa_iaa` | function | `doa_iaa(beam_vect, steering_vect, num_it=15, p_init=None)` | as shown | beam vector, steering vector, iteration count | spectrum-like estimate | no static exception summary captured yet | Specified | `radarsimpy_origin/processing.py:668` |
| `doa_bartlett` | function | `doa_bartlett(covmat, spacing=0.5, scanangles=range(-90, 91))` | as shown | covariance matrix | spectrum-like output | no static exception summary captured yet | Specified | `radarsimpy_origin/processing.py:730` |
| `doa_capon` | function | `doa_capon(covmat, spacing=0.5, scanangles=range(-90, 91))` | as shown | covariance matrix | spectrum-like output | no static exception summary captured yet | Specified | `radarsimpy_origin/processing.py:767` |

## `tools.py`

| API | Kind | Signature or Shape | Defaults | Inputs | Outputs | Exceptions or Warnings | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `marcumq` | function | `marcumq(a, x, m=1)` | `m=1` | float inputs | `float` | delegates to SciPy `ncx2.cdf` | Specified | `radarsimpy_origin/tools.py:47` |
| `log_factorial` | function | `log_factorial(n)` | none | `int` or ndarray | `float` or ndarray | none from static inspection | Specified | `radarsimpy_origin/tools.py:68` |
| `threshold` | function | `threshold(pfa, npulses)` | none | false alarm probability and pulse count | `float` | none from static inspection | Specified | `radarsimpy_origin/tools.py:90` |
| `pd_swerling0` | function | `pd_swerling0(npulses, snr, thred)` | none | pulse count, SNR, threshold | scalar or array probability of detection | ignores some `RuntimeWarning` values internally | Specified | `radarsimpy_origin/tools.py:111` |
| `pd_swerling1` | function | `pd_swerling1(npulses, snr, thred)` | none | pulse count, SNR, threshold | `float` | none from static inspection | Specified | `radarsimpy_origin/tools.py:184` |
| `pd_swerling2` | function | `pd_swerling2(npulses, snr, thred)` | none | pulse count, SNR, threshold | `float` | none from static inspection | Specified | `radarsimpy_origin/tools.py:221` |
| `pd_swerling3` | function | `pd_swerling3(npulses, snr, thred)` | none | pulse count, SNR, threshold | `float` | none from static inspection | Specified | `radarsimpy_origin/tools.py:246` |
| `pd_swerling4` | function | `pd_swerling4(npulses, snr, thred)` | none | pulse count, SNR, threshold | `float` | none from static inspection | Specified | `radarsimpy_origin/tools.py:294` |
| `roc_pd` | function | `roc_pd(pfa, snr, npulses=1, stype='Coherent')` | as shown | Pfa, SNR, pulse count, signal type | probability of detection scalar or array | behavior depends on `stype` dispatch | Specified | `radarsimpy_origin/tools.py:361` |
| `roc_snr` | function | `roc_snr(pfa, pd, npulses=1, stype='Coherent')` | as shown | Pfa, Pd, pulse count, signal type | required SNR scalar or array | behavior depends on `stype` dispatch | Specified | `radarsimpy_origin/tools.py:454` |

## `mesh_kit.py`

| API | Kind | Signature or Shape | Defaults | Inputs | Outputs | Exceptions or Warnings | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `check_module_installed` | function | `check_module_installed(module_name)` | none | module name string | `bool` | swallows import discovery exceptions internally | Specified | `radarsimpy_origin/mesh_kit.py:28` |
| `safe_import` | function | `safe_import(module_name)` | none | module name string | module object or `None` | returns `None` instead of raising `ImportError` | Specified | `radarsimpy_origin/mesh_kit.py:53` |
| `import_mesh_module` | function | `import_mesh_module()` | module priority is fixed internally | none | first available mesh module object | `ImportError` with installation guidance if none found | Specified | `radarsimpy_origin/mesh_kit.py:68` |
| `load_mesh` | function | `load_mesh(mesh_file_name, scale, mesh_module)` | none | mesh path, scale factor, imported mesh module | `dict` with `points` and `cells` arrays | `ImportError` if module is unsupported | Specified | `radarsimpy_origin/mesh_kit.py:101` |

## Notes For Whitebox Implementation

- Constructors and package utility functions are fully visible from static analysis.
- `processing.py`, `tools.py`, and `mesh_kit.py` are good first whitebox targets because their interfaces are readable and mostly deterministic.
- An expanded vendor oracle capture now exists at `oracle_capture_output/macos_arm_py311/manifest.json` for the macOS arm build.
- Windows-origin capture is still required before claiming cross-platform blackbox parity.
- The current whitebox workspace now implements both `processing` and `tools` independently, along with a reference `sim_radar` path for point targets and static mesh targets, a structured-array `sim_lidar` point-cloud path, a reference triangular-facet `sim_rcs` path, the readable top-level metadata and diagnostic utility functions, and a compatible local license surface.
