# RadarSimPy Whitebox Install Guide

## 이 가이드의 대상

이 가이드는 현재 저장소에 들어 있는 whitebox 구현 [`radarsimpy_whitebox`](/Users/seongcheoljeong/Documents/RadarSimPy/radarsimpy_whitebox)를 사용하는 사람을 위한 설치 안내입니다.

공식 RadarSimPy 설치 문서와 가장 큰 차이는 아래와 같습니다.

- 공식 문서는 prebuilt vendor `radarsimpy/` 폴더를 프로젝트에 복사하는 방식입니다.
- 현재 whitebox는 이 저장소 안에서 직접 import해서 사용하는 방식입니다.
- 현재 저장소에는 `pyproject.toml`이나 `setup.py`가 없으므로 `pip install -e .` 기준 문서가 아닙니다.

## 권장 환경

- Python 3.11 권장
- macOS, Linux, Windows 모두 가능하지만 현재 vendor oracle 검증은 macOS arm 중심으로 진행되었습니다
- 필수 패키지:
  - `numpy`
  - `scipy`
- 선택 패키지:
  - `trimesh`
  - `pyvista`
  - `pymeshlab`
  - `meshio`

선택 패키지가 없어도 기본 ASCII STL, OBJ는 whitebox 내장 mesh loader로 처리할 수 있습니다.

## 설치 절차

저장소 루트에서 아래 순서로 진행합니다.

```bash
cd /Users/seongcheoljeong/Documents/RadarSimPy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy
```

mesh 파일 지원을 넓히고 싶다면 아래 중 하나 이상을 추가로 설치합니다.

```bash
python -m pip install trimesh
```

또는:

```bash
python -m pip install pyvista
python -m pip install pymeshlab
python -m pip install meshio
```

## import 방식

가장 단순한 방법은 저장소 루트에서 스크립트를 실행하는 것입니다.

```bash
cd /Users/seongcheoljeong/Documents/RadarSimPy
python your_script.py
```

파이썬에서 기본 import는 아래처럼 사용합니다.

```python
import radarsimpy_whitebox as rsp
```

저장소 바깥에서 실행해야 한다면 `PYTHONPATH`에 repo root를 추가합니다.

```bash
export PYTHONPATH="/Users/seongcheoljeong/Documents/RadarSimPy:${PYTHONPATH}"
```

## 설치 확인

아래 명령으로 import와 기본 의존성을 확인할 수 있습니다.

```bash
cd /Users/seongcheoljeong/Documents/RadarSimPy
python - <<'PY'
import radarsimpy_whitebox as rsp

print("version:", rsp.__version__)
print("installed:", rsp.check_installation())
print("info:", rsp.get_info())
PY
```

정상이라면 최소한 아래가 확인되어야 합니다.

- `rsp.__version__ == "15.1.0"`
- `rsp.check_installation()`이 `True`
- `rsp.get_info()`에 `processing`, `tools`, `sim_radar`, `sim_lidar`, `sim_rcs`가 포함됨

## 가장 흔한 실수

### 1. `radarsimpy`를 import한 경우

현재 저장소의 [`radarsimpy`](/Users/seongcheoljeong/Documents/RadarSimPy/radarsimpy)는 whitebox가 아니라 vendor runtime capture용 링크일 수 있습니다.

whitebox 사용자 문서 기준 기본 import는 아래입니다.

```python
import radarsimpy_whitebox as rsp
```

### 2. 저장소 바깥에서 실행해서 `ModuleNotFoundError`가 나는 경우

해결 방법은 둘 중 하나입니다.

- 저장소 루트에서 실행
- `PYTHONPATH`에 `/Users/seongcheoljeong/Documents/RadarSimPy` 추가

### 3. mesh 파일 로딩이 안 되는 경우

whitebox는 기본적으로 ASCII STL/OBJ를 지원합니다. 그 외 포맷은 mesh 관련 선택 패키지가 필요할 수 있습니다.

가장 무난한 추가 설치는 아래입니다.

```bash
python -m pip install trimesh
```

### 4. 공식 RadarSimPy 설치 가이드를 그대로 따른 경우

공식 설치 문서는 vendor `radarsimpy/` 배포본 기준입니다. 현재 whitebox는 배포 패키지가 아니라 repo-local import 기준이므로 절차가 다릅니다.

공식 문서: [RadarSimPy Installation](https://radarsimx.github.io/radarsimpy/user_guide/installation.html)

## 라이선스 관련 주의사항

whitebox는 `set_license`, `is_licensed`, `get_license_info`를 제공하지만, 이것은 현재 whitebox workspace에서의 compatibility surface입니다.

- vendor free-tier 제한 정책을 그대로 재현하는 문서는 아닙니다
- commercial license enforcement를 동일하게 제공한다고 가정하면 안 됩니다

자세한 차이는 [`OFFICIAL_DOCS_COMPATIBILITY_KR.md`](./OFFICIAL_DOCS_COMPATIBILITY_KR.md)를 참고하세요.
