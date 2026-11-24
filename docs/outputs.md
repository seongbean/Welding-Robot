# 산출물 폴더 규약

실행 산출물(시나리오, 에피소드, 데이터셋, 모델, 평가)은 git에 넣지 않고 모두 `outputs/` 한 곳 아래에 둡니다.
루트는 환경변수 `WELD_OUTPUTS`로 바꿀 수 있습니다. 경로 계산은 [`common/paths.py`](../common/paths.py)가 맡습니다.

## 구조

```text
outputs/
├── experiments/<YYYYMMDD>-<이름>/    시나리오 묶음 하나 = 실험 하나
│   ├── manifest.json                 단계마다 git 커밋·명령·시각·설정을 덧붙임
│   ├── batch/                        시나리오 json           (sim_errors.make_scenarios)
│   ├── reach/                        reach_report.json       (sim_bridge/run_batch.py --reach-only)
│   ├── episodes/                     운동학 에피소드 .npz + batch_report.json (run_batch.py --kinematic)
│   ├── episodes_isaac/<시편군>/       Isaac 에피소드 .npz + batch_report.json (run_batch.py)
│   ├── data/                         frames.npz, splits.json, lerobot/ (learning.to_lerobot)
│   ├── runs/<정책>/                  policy.pt, train_log.json, eval.json (learning.train_bc, evaluate)
│   └── report.md                     표 (learning.run_experiment)
├── ci/<GitHub 실행 ID>/<정책>/        내려받은 실험 워크플로 결과 (eval.json, train_log.json)
└── scratch/                          단독 시나리오·디버그. 지워도 되는 것
```

- **실험 ID**: `YYYYMMDD-<이름>`, 이름은 소문자·숫자·`-`·`_`·`.` (예: `20251123-kin-4fam`, `20251124-isaac-4fam`).
  `docs/experiments/`의 결과 기록도 같은 ID를 씁니다.
- 하위 폴더 이름은 `learning/run_experiment.py`가 이미 쓰는 배치(`data/`, `runs/`, `report.md`)와 같습니다.
  한 실험에 변형이 여럿이면 `episodes_prof/`, `data_prof/`처럼 접미사를 붙입니다.

## 지금 쓰는 법

지금은 각 명령의 `--out`에 위 경로를 직접 줍니다 (`--exp` 옵션은 다음 단계에서 추가).

```bash
E=outputs/experiments/20251201-weave
python -m sim_errors.make_scenarios --batch --scale 0.2 --out $E/batch
python sim_bridge/run_batch.py $E/batch --out $E/episodes --kinematic --jobs 4
python -m learning.run_experiment --episodes $E/episodes --out $E --jobs 8
```

## git에 넣는 것과 넣지 않는 것

| 대상 | 둔 곳 |
| --- | --- |
| 코드, 테스트, config yaml, 워크플로, 문서 | git |
| 실험 결론(표, 실행 링크) | git, `docs/experiments/` |
| 로봇·장면 자산 (`weld_teleop/urdf/so102.urdf`, `assets/isaac/opti3_welding_scene/`) | git |
| 대표 정책 `policy.pt` | GitHub Release 첨부 |
| 시나리오, 에피소드, 데이터셋, 일반 runs | `outputs/` (gitignore) |

## 옛 경로 (임시)

2026-11-25에 로컬 산출물을 이 구조로 옮겼습니다. 옛 이름 `exp`, `exp_isaac`, `github-evals`는 새 위치를 가리키는
Windows 정션으로 남아 있어 예전 명령도 같은 곳에 씁니다. 대응표는 로컬의 `outputs/MIGRATION.md`에 있습니다.

| 옛 경로 | 새 경로 |
| --- | --- |
| `exp/` | `outputs/experiments/20251123-kin-4fam/` |
| `exp_isaac/` (`episodes/` → `episodes_isaac/`) | `outputs/experiments/20251124-isaac-4fam/` |
| `github-evals/` | `outputs/ci/` |
| `batches/t_small`, `episodes/t_small(_prof)`, `datasets/t_small(_prof)`, `runs/` | `outputs/experiments/20251118-t-small/` |
| `scenarios/`, `scenarios_single/`, `episodes/*.npz` | `outputs/scratch/20251117-single-checks/` |

새 실험은 옛 경로가 아니라 `outputs/experiments/<ID>/`에 만듭니다. 정션은 모든 작업이 새 경로로 옮겨 온 뒤 지웁니다.
