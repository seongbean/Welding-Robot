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

- **실험 ID**: `YYYYMMDD-<이름>`, 이름은 소문자·숫자·`-`·`_`·`.` (예: `20261123-kin-4fam`, `20261124-isaac-4fam`). 날짜는 실험을 시작한 날의 연월일입니다(여러 날에 걸쳐도 첫날 ID를 계속 씁니다).
  `docs/experiments/`의 결과 기록도 같은 ID를 씁니다.
- 하위 폴더 이름은 `learning/run_experiment.py`가 이미 쓰는 배치(`data/`, `runs/`, `report.md`)와 같습니다.
  한 실험에 변형이 여럿이면 `episodes_prof/`, `data_prof/`처럼 접미사를 붙입니다.

## 쓰는 법: --exp

명령마다 `--exp <실험 ID>`를 주면 입력·출력 경로를 위 배치에서 채우고, 실험 폴더의 `manifest.json`에
단계(시각, git 커밋·변경 여부, 명령줄, 설정 해시, 시드 등)를 덧붙입니다. 경로를 따로 주면(`--out`, `--data` 등)
그 경로가 우선합니다. `--exp`를 주지 않으면 예전과 똑같이 동작하고 manifest도 쓰지 않습니다.

```bash
E=20261201-weave
python -m sim_errors.make_scenarios --batch --scale 0.2 --exp $E          # → batch/
python sim_bridge/run_batch.py --exp $E --kinematic --jobs 4                # batch/ → episodes/
python sim_bridge/run_batch.py --exp $E --reach-only --follower-urdf ...   # batch/ → reach/
python sim_bridge/run_batch.py --exp $E --group butt --only butt_ --follower-urdf ...   # Isaac → episodes_isaac/butt/
python -m learning.to_lerobot --exp $E --no-lerobot                         # episodes/ → data/
python -m learning.train_bc --exp $E --run mlp_all                          # data/ → runs/mlp_all/
python -m learning.evaluate mlp_all rule --exp $E --jobs 8                  # runs/<이름>/eval.json
python -m learning.run_experiment --exp $E --jobs 8                         # 변환·학습·평가·report.md 한 번에
```

- `run_batch.py --exp`: 시나리오 폴더 기본값은 `batch/`. 출력은 `--kinematic`이면 `episodes/`, `--reach-only`면
  `reach/`, Isaac이면 `episodes_isaac/<--group>/`(기본 `all`).
- `train_bc --exp`는 `--run <정책 이름>`이 있어야 출력 폴더를 정합니다.
- `evaluate --exp`에서 경로가 아닌 정책 이름(`mlp_all`)은 `runs/<이름>/policy.pt`, 규칙 기준선(`rule`)은 `runs/rule/`.
- `run_experiment --exp`는 하위 단계에도 `--exp`를 넘겨 같은 manifest에 기록합니다.

## git에 넣는 것과 넣지 않는 것

| 대상 | 둔 곳 |
| --- | --- |
| 코드, 테스트, config yaml, 워크플로, 문서 | git |
| 실험 결론(표, 실행 링크) | git, `docs/experiments/` |
| 로봇·장면 자산 (`weld_teleop/urdf/so102.urdf`, `assets/isaac/opti3_welding_scene/`) | git |
| 대표 정책 `policy.pt` | GitHub Release 첨부 |
| 시나리오, 에피소드, 데이터셋, 일반 runs | `outputs/` (gitignore) |

## 기본 출력

`--exp`도 `--out`도 주지 않은 단독 실행은 `outputs/scratch/` 아래에 씁니다:
`make_scenarios` → `scratch/scenarios/`, `expert_to_sim.py` → `scratch/episodes/`, `evaluate rule`의 `--rule-dir` → `scratch/runs/`.
저장소 루트나 현재 폴더에 산출물 폴더를 만들지 않습니다.

## 옛 경로

2026-11-25에 로컬 산출물을 이 구조로 옮겼습니다. 대응표는 로컬의 `outputs/MIGRATION.md`에 있습니다.

| 옛 경로 | 새 경로 |
| --- | --- |
| `exp/` | `outputs/experiments/20261123-kin-4fam/` |
| `exp_isaac/` (`episodes/` → `episodes_isaac/`) | `outputs/experiments/20261124-isaac-4fam/` |
| `github-evals/` | `outputs/ci/` |
| `batches/t_small`, `episodes/t_small(_prof)`, `datasets/t_small(_prof)`, `runs/` | `outputs/experiments/20261118-t-small/` |
| `scenarios/`, `scenarios_single/`, `episodes/*.npz` | `outputs/scratch/20261117-single-checks/` |

옛 이름은 `.gitignore`에 남아 있어, 예전 명령으로 다시 생겨도 커밋되지 않습니다.

처음에는 실험 ID 연도를 2025로 잘못 붙였습니다(`20251123-kin-4fam` 등). 실험은 모두 2026년 11월에 한 것이라 2026-11-25에 연도만 2026으로 바꿨습니다. 옛 ID가 보이면 앞 네 자리만 2026으로 읽으면 됩니다.
