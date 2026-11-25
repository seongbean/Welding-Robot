# 용접 로봇 (Welding-Robot)

사람이 시작 트리거를 누르면 계산된 이음매 경로 위에서 학습 정책이 보정값만 내고, 숙련공은 언제든 리더 암으로 보정을 더하는 용접 로봇입니다.
설계는 [아키텍처](docs/architecture.md), 설계마다 검토한 대안과 이유는 [설계 결정](docs/design-decisions.md)에 있습니다.

## 폴더

| 폴더 | 내용 |
| --- | --- |
| [`weld_teleop/`](weld_teleop/README.md) | ROS2 텔레옵 패키지: 리더 암(SO-101/102) → opti3 팔로워, IK와 안전 |
| [`sim_bridge/`](sim_bridge/README.md) | Isaac Sim 연결: 수신기, 리더·전문가 송신기, 시나리오 묶음 실행(`run_batch.py`) |
| [`sim_errors/`](sim_errors/README.md) | 모재 오차 모델, 시나리오 생성, 규칙 전문가, 에피소드 기록·점검 |
| [`learning/`](learning/README.md) | 에피소드 → 학습 데이터 변환, MLP/GRU 행동 복제, 닫힌 루프 평가 |
| `common/` | 공용 코드 (산출물 경로 규약 `paths.py`) |
| `tools/` | 하드웨어 점검 스크립트 (`leader_probe.py`, `so102_urdf_check.py`) |
| `assets/isaac/` | Isaac Sim 장면 (`opti3_welding_scene/posco_torch.usd`) |
| `docs/` | [아키텍처](docs/architecture.md), [설계 결정](docs/design-decisions.md), [산출물 규약](docs/outputs.md), [실험 기록](docs/experiments/README.md) |
| `outputs/` | 실행 산출물 (git에 넣지 않음, [산출물 규약](docs/outputs.md)) |

## 빠른 시작

저장소 루트에서. Isaac 없이 운동학으로 생성 → 변환 → 학습 → 평가를 한 바퀴 돌립니다 (`pip install numpy pyyaml pin torch`).

```bash
E=20251201-try
python -m sim_errors.make_scenarios --batch --scale 0.2 --exp $E
python sim_bridge/run_batch.py --exp $E --kinematic --jobs 4
python -m learning.run_experiment --exp $E --jobs 4      # → outputs/experiments/$E/report.md
```

Isaac Sim 실행, 시편군별 묶음, 비교 실험은 각 폴더 README에 있습니다.

## 확인

```bash
ruff check .
pytest            # ROS 없이 돌아감 (설정은 pyproject.toml)
```
