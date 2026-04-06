# RadarSimPy Whitebox 사용자 문서

이 문서는 현재 저장소의 whitebox 구현인 `radarsimpy_whitebox`를 실제로 사용하는 사람을 위한 안내서입니다.

중요한 전제는 두 가지입니다.

- 이 문서는 공식 vendor 패키지 `radarsimpy` 설치 문서가 아닙니다.
- 이 문서는 현재 저장소 루트(`/Users/seongcheoljeong/Documents/RadarSimPy`)에서 `radarsimpy_whitebox`를 사용하는 흐름을 기준으로 작성되었습니다.

## 먼저 알아둘 점

- 현재 이 저장소에는 실행 가능한 whitebox 패키지 [`radarsimpy_whitebox`](/Users/seongcheoljeong/Documents/RadarSimPy/radarsimpy_whitebox)가 있습니다.
- 같은 저장소의 [`radarsimpy`](/Users/seongcheoljeong/Documents/RadarSimPy/radarsimpy)는 현재 macOS arm vendor runtime capture를 위한 심볼릭 링크입니다.
- 따라서 whitebox 사용자는 기본적으로 `import radarsimpy_whitebox as rsp`를 사용해야 합니다.

## 문서 순서

1. [`INSTALL_GUIDE_KR.md`](./INSTALL_GUIDE_KR.md)
   현재 저장소 기준 설치, 의존성, import 확인 절차를 설명합니다.
2. [`USAGE_GUIDE_KR.md`](./USAGE_GUIDE_KR.md)
   `Transmitter`, `Receiver`, `Radar`, `sim_radar`, `sim_lidar`, `sim_rcs`, `processing`, `tools`의 기본 사용 흐름을 설명합니다.
3. [`OFFICIAL_DOCS_COMPATIBILITY_KR.md`](./OFFICIAL_DOCS_COMPATIBILITY_KR.md)
   공식 RadarSimPy 문서에서 무엇을 그대로 참고해도 되는지, 무엇은 whitebox 기준으로 다시 봐야 하는지 정리합니다.

## 빠른 요약

- 설치는 패키지 배포본 설치가 아니라 현재 repo 사용 방식입니다.
- 기본 import는 `radarsimpy_whitebox`입니다.
- 모델링과 `processing/tools`는 공식 RadarSimPy 감각과 매우 비슷합니다.
- `sim_radar`, `sim_lidar`, `sim_rcs`는 많이 구현되어 있지만, 공식 vendor simulator의 모든 ray-tracing branch와 완전히 동일하다고 보지는 않습니다.

## 관련 문서

- 공식 RadarSimPy 문서: [https://radarsimx.github.io/radarsimpy/](https://radarsimx.github.io/radarsimpy/)
- 내부 구현/검증 문서:
  - [`../spec/PARITY_MATRIX.md`](../spec/PARITY_MATRIX.md)
  - [`../spec/RUNTIME_GAPS.md`](../spec/RUNTIME_GAPS.md)
  - [`../spec/API_INVENTORY.md`](../spec/API_INVENTORY.md)
