# 작업 규약

이 저장소에서는 여러 작업(사람, Claude 세션)이 같은 저장소와 같은 로컬 폴더(`C:\workspace\Welding-Robot`)를 함께 씁니다.

## 산출물 경로

- 실행 산출물은 모두 `outputs/` 아래에 둡니다. 규약은 [docs/outputs.md](docs/outputs.md), 경로 계산은 `common/paths.py`.
- 새 실험은 `outputs/experiments/<YYYYMMDD>-<이름>/` 하나를 만들고 그 안에 `batch/`, `reach/`, `episodes/`,
  `episodes_isaac/<시편군>/`, `data/`, `runs/<정책>/`, `report.md`를 둡니다.
- 명령에 `--exp <ID>`를 주면 경로가 이 배치로 채워지고 `manifest.json`에 기록됩니다 (docs/outputs.md "쓰는 법").
- 저장소 루트나 소스 폴더에 산출물 폴더(`exp`, `episodes`, `runs` 등)를 새로 만들지 않습니다.
  경로를 주지 않은 단독 실행은 `outputs/scratch/`에 씁니다.
- 산출물(.npz, .pt, .parquet, 에피소드, 데이터셋)은 커밋하지 않습니다.

## 문서

- 실험 결과는 `docs/experiments/<실험 ID>.md` 새 파일에 적고 `docs/experiments/README.md` 표 맨 위에 한 줄을 더합니다.
  ID는 산출물 폴더 `outputs/experiments/<ID>/`와 같게 합니다. `learning/README.md`에는 사용법만 둡니다.
- 설계는 `docs/architecture.md`, 대안과 이유는 `docs/design-decisions.md`.

## 자산

- Isaac 장면은 `assets/isaac/opti3_welding_scene/` (USD끼리 상대경로로 참조하므로 폴더 구조를 바꾸지 않음).
  `sim_bridge/isaac_opti3_receiver.py`의 `--usd` 기본값이 이 장면입니다.
- 리더 URDF는 `weld_teleop/urdf/so102.urdf`.

## 하위 호환

- 패키지 이름(`weld_teleop`, `sim_errors`, `sim_bridge`, `learning`)과 실행 명령(`python -m sim_errors.make_scenarios`,
  `python sim_bridge/run_batch.py` 등), `--out`의 의미는 바꾸지 않습니다. 새 동작은 선택 옵션으로만 더합니다.

## 확인

- 린트: `ruff check .` / 테스트: `pytest` (설정은 `pyproject.toml`, ROS 없이 돌아감).
