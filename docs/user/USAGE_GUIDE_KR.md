# RadarSimPy Whitebox Usage Guide

## 기본 import

whitebox 사용의 기본 import는 아래입니다.

```python
import radarsimpy_whitebox as rsp
```

필요하면 simulator 함수만 별도 import할 수도 있습니다.

```python
from radarsimpy_whitebox import sim_lidar, sim_radar, sim_rcs
```

## 1. 가장 기본적인 radar workflow

whitebox의 기본 흐름은 공식 RadarSimPy와 거의 같습니다.

1. `Transmitter` 생성
2. `Receiver` 생성
3. `Radar` 생성
4. `sim_radar()` 실행
5. `processing`으로 후처리

예제:

```python
import numpy as np
import radarsimpy_whitebox as rsp

tx = rsp.Transmitter(
    f=[76.0e9, 76.2e9],
    t=[0.0, 40e-6],
    tx_power=13,
    pulses=4,
    prp=60e-6,
    channels=[{"location": [0.0, 0.0, 0.0]}],
)

rx = rsp.Receiver(
    fs=4e6,
    noise_figure=11,
    rf_gain=0,
    load_resistor=500,
    baseband_gain=0,
    bb_type="complex",
    channels=[{"location": [0.0, 0.0, 0.0]}],
)

radar = rsp.Radar(tx, rx, seed=2026)

targets = [
    {
        "location": [50.0, 0.0, 0.0],
        "rcs": 10.0,
        "speed": [0.0, 0.0, 0.0],
    }
]

result = rsp.sim_radar(radar, targets)

print(result.keys())
print(result["baseband"].shape)
print(result["timestamp"].shape)
print(result["interference"])
```

`sim_radar()` 결과는 현재 whitebox 기준으로 아래 키를 기대하면 됩니다.

- `baseband`
- `noise`
- `timestamp`
- `interference`

간섭 레이더를 넣지 않은 경우 `interference`는 보통 `None`입니다.

## 2. range / Doppler 처리

`sim_radar()` 결과는 `processing` 함수와 바로 연결할 수 있습니다.

```python
range_cube = rsp.processing.range_fft(result["baseband"])
doppler_cube = rsp.processing.doppler_fft(range_cube)
rd_map = rsp.processing.range_doppler_fft(result["baseband"])

print(range_cube.shape)
print(doppler_cube.shape)
print(rd_map.shape)
```

자주 쓰는 함수는 아래와 같습니다.

- `range_fft`
- `doppler_fft`
- `range_doppler_fft`
- `cfar_ca_1d`, `cfar_ca_2d`
- `cfar_os_1d`, `cfar_os_2d`
- `doa_bartlett`, `doa_capon`, `doa_music`, `doa_esprit`

## 3. LiDAR 사용법

`sim_lidar()`는 lidar 설정 dict와 mesh target 목록을 받습니다.

```python
import numpy as np
import radarsimpy_whitebox as rsp

lidar = {
    "position": [0.0, 0.1, 0.1],
    "phi": np.array([0.0]),
    "theta": np.array([90.0]),
}

targets = [
    {
        "model": "models/plate.stl",
        "location": [10.0, 0.0, 0.0],
    }
]

hits = rsp.sim_lidar(lidar, targets, frame_time=0.0)

print(hits.dtype.names)
print(hits["positions"])
print(hits["directions"])
```

현재 whitebox `sim_lidar()`는 structured ndarray를 반환하며, 아래 필드를 포함합니다.

- `positions`
- `origins`
- `directions`
- `distance`
- `phi`
- `theta`
- `target_index`

주의할 점:

- ray가 아무 target에도 맞지 않으면 해당 ray는 결과에서 제외됩니다
- 현재 vendor oracle로 강하게 확인된 공통 필드는 `positions`, `directions` 중심입니다
- whitebox는 그보다 넓은 hit record를 제공합니다

## 4. RCS 사용법

`sim_rcs()`는 mesh 기반 RCS 계산 함수입니다.

```python
import radarsimpy_whitebox as rsp

targets = [
    {
        "model": "models/plate.stl",
        "location": [0.0, 0.0, 0.0],
    }
]

rcs = rsp.sim_rcs(
    targets,
    f=77e9,
    inc_phi=0.0,
    inc_theta=90.0,
)

print(rcs)
```

스윕도 가능합니다.

```python
import numpy as np
import radarsimpy_whitebox as rsp

obs_phi = np.linspace(-90.0, 90.0, 7)

rcs_curve = rsp.sim_rcs(
    [{"model": "models/plate.stl"}],
    f=77e9,
    inc_phi=0.0,
    inc_theta=90.0,
    obs_phi=obs_phi,
    obs_theta=90.0,
)

print(rcs_curve.shape)
```

주의할 점:

- 현재 whitebox `sim_rcs()`는 공식 blackbox ray tracing / SBR 엔진의 내부 구현과 동일하지 않습니다
- 현재 구현은 geometry-backed reference path입니다
- 따라서 사용 목적은 “같은 public API로 reference calculation 수행”에 가깝습니다

## 5. mesh target 사용법

`sim_radar`, `sim_lidar`, `sim_rcs` 모두 mesh target dict를 사용할 수 있습니다.

예:

```python
target = {
    "model": "models/plate.stl",
    "location": [10.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0],
    "origin": [0.0, 0.0, 0.0],
    "unit": "m",
}
```

현재 whitebox에서 기억하면 좋은 점:

- 기본 내장 mesh loader는 ASCII STL, OBJ 지원
- 다른 포맷은 `trimesh`, `pyvista`, `pymeshlab`, `meshio` 설치 시 더 잘 처리될 수 있음
- `sim_radar`에서 mesh target motion은 아직 구현 범위가 좁습니다

## 6. `sim_radar()`에서 현재 조심할 옵션

시그니처는 공식과 비슷하지만, 현재 whitebox에서 고급 옵션 일부는 제한이 있습니다.

```python
rsp.sim_radar(
    radar,
    targets,
    density=1,
    level=None,
    interf=None,
    ray_filter=None,
    back_propagating=False,
    device="gpu",
    log_path=None,
    dry_run=False,
)
```

현재 문서 기준:

- `interf`는 미지원
- `ray_filter`는 현재 무시됨
- `back_propagating`은 현재 무시됨
- `log_path`는 현재 무시됨
- `device`는 `"gpu"`와 `"cpu"`를 받지만, vendor binary 문서의 성능 의미를 그대로 기대하면 안 됨

즉, 현재 whitebox에서는 point target, MIMO, multi-frame, static mesh target, 기본 range-Doppler workflow 중심으로 사용하는 것이 가장 안전합니다.

## 7. `tools` 사용법

`tools`에는 ROC와 탐지 성능 계산 함수가 들어 있습니다.

```python
import radarsimpy_whitebox as rsp

th = rsp.tools.threshold(pfa=1e-6, npulses=1)
pd = rsp.tools.roc_pd(pfa=1e-6, snr=13.0, npulses=1, stype="Coherent")
snr = rsp.tools.roc_snr(pfa=1e-6, pd=0.9, npulses=1, stype="Coherent")

print(th, pd, snr)
```

## 8. package utility surface

whitebox는 metadata / diagnostics 함수도 제공합니다.

```python
import radarsimpy_whitebox as rsp

print(rsp.get_version())
print(rsp.get_info())
rsp.print_info()
print(rsp.check_installation())
```

## 9. license compatibility surface

사용은 가능하지만, 공식 vendor license 시스템과 동일하다고 가정하면 안 됩니다.

```python
import radarsimpy_whitebox as rsp

rsp.set_license()
print(rsp.is_licensed())
print(rsp.get_license_info())
```

현재 whitebox에서 이 API는 “호환 표면” 역할에 가깝습니다.

## 10. 어떤 예제를 먼저 따라 하면 좋은가

추천 순서는 아래입니다.

1. point target `sim_radar`
2. `range_doppler_fft`
3. MIMO radar shape 확인
4. `sim_lidar` structured array 읽기
5. `sim_rcs` scalar / sweep 계산

공식 예제를 가져와서 수정할 때는 반드시 [`OFFICIAL_DOCS_COMPATIBILITY_KR.md`](./OFFICIAL_DOCS_COMPATIBILITY_KR.md)를 같이 참고하세요.
