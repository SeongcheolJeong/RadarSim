# 공식 RadarSimPy 문서와 Whitebox의 호환성

## 한 줄 결론

공식 RadarSimPy 웹페이지는 참고 자료로는 매우 유용하지만, 그대로 whitebox 문서로 쓰면 안 됩니다.

가장 큰 이유는 아래와 같습니다.

- 공식 문서는 vendor binary 패키지 `radarsimpy` 기준입니다
- 현재 whitebox는 repo-local 패키지 `radarsimpy_whitebox` 기준입니다
- simulator와 license의 일부 동작은 공식판과 범위가 다릅니다

공식 문서 메인: [https://radarsimx.github.io/radarsimpy/](https://radarsimx.github.io/radarsimpy/)

## 항목별 재사용 가능성

| 주제 | 그대로 참고 가능 | 수정 후 참고 가능 | 그대로 쓰면 안 되는 이유 |
| --- | --- | --- | --- |
| 모델 개념 (`Transmitter`, `Receiver`, `Radar`) | 높음 | 매우 쉬움 | 기본 개념과 생성자 감각이 유사함 |
| `processing`, `tools` API | 높음 | 매우 쉬움 | 함수명과 용도가 거의 동일함 |
| 좌표계 / 도플러 부호 개념 | 높음 | 낮음 | 개념 문서라서 whitebox에도 그대로 유효함 |
| point target 예제 | 중간 | 높음 | import 경로만 바꾸면 상당수 구조를 재사용 가능 |
| basic MIMO 예제 | 중간 | 높음 | 모델링과 processing 흐름은 거의 동일함 |
| `sim_lidar` 기본 사용법 | 중간 | 높음 | config dict와 target dict 구조는 비슷하지만 반환 필드 설명을 whitebox 기준으로 바꿔야 함 |
| `sim_rcs` 기본 사용법 | 중간 | 중간 | 호출 형태는 비슷하지만 내부 물리 모델 설명은 다시 써야 함 |
| 설치 문서 | 낮음 | 낮음 | 공식 문서는 vendor 배포본 복사 방식이고 whitebox는 repo-local import 방식임 |
| license 문서 | 낮음 | 낮음 | whitebox는 vendor enforcement가 아니라 compatibility surface에 가까움 |
| ray tracing / interference / GPU 성능 문서 | 낮음 | 낮음 | whitebox는 일부 고급 branch가 미구현 또는 축소됨 |
| build from source 문서 | 낮음 | 낮음 | 현재 저장소는 그 문서 구조를 따르지 않음 |

## 공식 설치 문서를 그대로 쓰면 안 되는 이유

공식 설치 문서: [Installation](https://radarsimx.github.io/radarsimpy/user_guide/installation.html)

공식 설치 문서는 아래를 전제로 합니다.

- prebuilt `radarsimpy/` 폴더를 프로젝트에 복사
- platform-specific compiled library 사용
- license file placement도 vendor package layout 기준

현재 whitebox는 아래가 기준입니다.

- 저장소 안의 `radarsimpy_whitebox/`를 import
- `numpy`, `scipy`를 먼저 설치
- 저장소 루트 또는 `PYTHONPATH` 기준으로 사용
- `radarsimpy` 이름은 현재 repo에서 vendor capture alias일 수 있음

따라서 설치 섹션은 반드시 whitebox 전용으로 다시 써야 합니다.

## 공식 사용 예제를 재활용하는 방법

아래 규칙으로 보면 됩니다.

### 거의 그대로 가져와도 되는 경우

- `Transmitter`, `Receiver`, `Radar` 생성
- `processing.range_fft`, `doppler_fft`, `range_doppler_fft`
- `tools.threshold`, `roc_pd`, `roc_snr`
- point target 기반 기본 radar workflow

이때 보통 필요한 수정은 아래 정도입니다.

```python
import radarsimpy_whitebox as rsp
```

그리고 기존 `radarsimpy.xxx`를 `rsp.xxx`로 바꾸면 됩니다.

### 내용을 다시 읽고 가져와야 하는 경우

- `sim_lidar()` 예제
- `sim_rcs()` 예제
- mesh target 예제

이 경우에는 아래를 꼭 다시 확인해야 합니다.

- 반환 데이터 구조
- 지원 mesh 포맷
- 현재 whitebox에서 지원하는 target key 범위

### 그대로 가져오면 위험한 경우

- `interf`를 쓰는 `sim_radar`
- `ray_filter`, `back_propagating`, `log_path`에 의미를 기대하는 예제
- vendor free-tier / license 제한 설명
- blackbox ray tracing / SBR 내부 동작 설명
- GPU/CPU 성능, 빌드 플래그, vendor binary 배포 구조 설명

## Whitebox 기준으로 다시 써야 하는 차이

### `sim_radar`

공식 문서의 시그니처는 거의 같지만 현재 whitebox는 아래 차이가 있습니다.

- `interf` 미지원
- `ray_filter` 무시
- `back_propagating` 무시
- `log_path` 무시
- mesh target motion은 아직 제한적

즉, 공식의 “고급 ray tracing simulator” 문장을 whitebox 설명으로 그대로 쓰면 안 됩니다.

### `sim_lidar`

공식 문서는 `positions` 중심으로 설명해도 되지만, whitebox는 현재 더 넓은 structured array를 반환합니다.

whitebox 필드:

- `positions`
- `origins`
- `directions`
- `distance`
- `phi`
- `theta`
- `target_index`

따라서 공식 LiDAR 문서는 개념 참고용으로 쓰고, 반환 필드 설명은 whitebox 전용으로 다시 써야 합니다.

공식 예제 페이지: [LIDAR point cloud example](https://radarsimx.com/wp-content/uploads/html/lidar.html)

### `sim_rcs`

공식 API 문서는 호출 인자 확인용으로 유용합니다.

공식 API: [Simulator API](https://radarsimx.github.io/radarsimpy/api/sim.html)

하지만 현재 whitebox `sim_rcs()`는 공식 blackbox ray tracing / SBR 구현과 내부적으로 동일하지 않습니다. 현재는 geometry-backed reference path로 보는 것이 맞습니다.

따라서 “SBR 엔진 그대로 사용”처럼 쓰면 안 됩니다.

### license

공식 문서는 free tier, commercial license, automatic validation을 설명합니다.

현재 whitebox는 아래 정도로 이해하는 것이 안전합니다.

- top-level API 호환
- 기본적인 상태 보고 가능
- vendor licensing policy 전체를 whitebox가 그대로 재현한다고 보지는 않음

## 추천 사용 원칙

whitebox 사용자에게는 아래 원칙을 추천합니다.

1. 설치는 반드시 whitebox 전용 가이드를 따른다.
2. 모델/processing/tools는 공식 문서를 강하게 참고해도 된다.
3. simulator는 공식 예제를 참고하되, whitebox 현재 범위를 먼저 확인한다.
4. 고급 ray-tracing, interference, license 정책은 공식 문서를 그대로 옮기지 않는다.

## 함께 보면 좋은 문서

- [`INSTALL_GUIDE_KR.md`](./INSTALL_GUIDE_KR.md)
- [`USAGE_GUIDE_KR.md`](./USAGE_GUIDE_KR.md)
- [`../spec/PARITY_MATRIX.md`](../spec/PARITY_MATRIX.md)
- [`../spec/RUNTIME_GAPS.md`](../spec/RUNTIME_GAPS.md)
