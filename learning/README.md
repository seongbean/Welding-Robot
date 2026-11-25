# 보정 정책 학습 (learning)

시뮬 규칙 전문가 에피소드 묶음으로 기준 보정 정책(작은 MLP 또는 GRU 행동 복제)을 학습하고, 보정 0 기준선 대비 정답 오차로 평가한다. 설계 근거는 [아키텍처](../docs/architecture.md), Claude Doc "에피소드 생성 사양"의 "기록 형식"과 "구현 순서".

| 파일 | 하는 일 |
| --- | --- |
| `features.py` | 관측·행동 정의, 단면 거리 버퍼. 변환과 평가가 같은 코드를 쓴다 |
| `to_lerobot.py` | 에피소드 .npz(100 Hz) → `frames.npz`(10 Hz, 학습용)와 LeRobotDataset |
| `train_bc.py` | MLP/GRU 행동 복제, 입력 묶음 선택, `policy.pt` 저장 |
| `evaluate.py` | 열린 루프(기록 관측으로 예측) + 닫힌 루프(운동학, 정책이 직접 토치를 몸) 평가. 정책 자리에 `rule`/`rule-noprescan`을 주면 학습 없는 규칙 기준선 |
| `report_evals.py` | 여러 `eval.json`을 표 하나로 (실험 워크플로 요약) |
| `policy_driver.py` | Isaac 정책 실행기(`sim_bridge/policy_to_sim.py`)의 정책 구동과 실측 기준 닫힌 루프 오차 |

## 관측과 행동

관측 61개 (단위 mm, deg). 기본 입력은 `section,profile,progress,mode` (토치 자세는 뺌).

| 묶음 (`--obs`) | 개수 | 내용 |
| --- | --- | --- |
| `section` | 36 | 단면 거리 버퍼: 토치 뒤 10 mm ~ 앞 30 mm, 5 mm 간격 9점 × (좌우, 높이, 틈새 폭) + 9점 유효 표시. 단면은 토치 앞 30 mm에서만 재므로 잰 값을 이음매 위치와 함께 쌓아 꺼낸다. 값은 추정 경로 기준 편차(토치 기준 측정 + 그때 토치 자세) |
| `profile` | 18 | 같은 버퍼 9점 × (작업각, 진행각): 단면 패치의 두 판 면 평면 맞춤으로 구한 이음매 좌표계 회전 ([sim_errors/README.md](../sim_errors/README.md) "단면 센서 모델") |
| `torch` | 4 | 추정 경로 기준 토치 자세 어긋남 (좌우, 높이, 작업각, 진행각). 위빙·섭동·시작 어긋남 포함. 직전 정답을 따라 하는 지름길이 생겨 기본 입력에서 뺌 |
| `progress` | 1 | 진행도 s |
| `mode` | 2 | 위빙/직선 원핫 |

행동 7개 (정답 = 규칙 전문가): 좌우, 높이(CTWD, 가접 올림 포함) [mm], 작업각, 진행각 [deg], 진행 속도 [mm/s], 위빙 진폭 [mm], 위빙 주파수 [Hz].

## 실행 순서

저장소 루트에서. 학습·변환은 `pip install torch "lerobot[dataset]"` (lerobot은 Python 3.12 이상, conda `weld` 환경에서 됨). lerobot이 없으면 `frames.npz`만 만든다.

```bat
python -m sim_errors.make_scenarios --batch --scale 0.2 --out batches\t_small
python sim_bridge\run_batch.py batches\t_small --out episodes\t_small --follower-urdf C:\...\opti3.urdf
python -m learning.to_lerobot episodes\t_small --out datasets\t_small
python -m learning.train_bc datasets\t_small --model mlp --out runs\mlp
python -m learning.train_bc datasets\t_small --model mlp --obs section,progress,mode --out runs\mlp_noprofile
python -m learning.train_bc datasets\t_small --model gru --out runs\gru
python -m learning.evaluate runs\mlp\policy.pt runs\mlp_noprofile\policy.pt runs\gru\policy.pt --data datasets\t_small --episodes episodes\t_small
```

Isaac 없이 전체 흐름만 볼 때는 두 번째 줄을 `--kinematic --jobs 4`(완벽 추종 가정)로 바꾼다.

경로 대신 실험 ID를 주면(`--exp 20261201-weave`) 모든 단계가 `outputs/experiments/<ID>/` 아래에 쓰고 `manifest.json`에 기록한다 ([산출물 규약](../docs/outputs.md)). 예: `python -m learning.train_bc --exp 20261201-weave --run mlp`, `python -m learning.evaluate mlp rule --exp 20261201-weave`.

비교용 명령: 시작 구간 스캔 없이 변환(`--no-prescan`), 학습 없는 규칙 기준선(`rule`, `rule-noprescan`), eval.json 여러 개를 표 하나로(`report_evals`).

```bat
python -m learning.to_lerobot episodes\t_small --out datasets\t_small_noscan --no-prescan
python -m learning.evaluate rule rule-noprescan --data datasets\t_small --episodes episodes\t_small --rule-dir runs
python -m learning.report_evals runs\mlp\eval.json runs\rule\eval.json runs\rule-noprescan\eval.json
```

### 시편군 (T 필릿 + ㄱ자 코너 필릿)

시편군마다 묶음을 만들고 에피소드 폴더 여러 개를 한 데이터로 합친다. 에피소드마다 시편군(`ep_family`)이 기록되어,
학습(`train_bc --family`)과 평가(`evaluate --family`)를 군별로 나눌 수 있다. 섞인 시험을 평가하면 표에 군별 줄
(`fam_fillet_t`, `fam_fillet_corner`, `fam_<군>_<강도>`)이 더해진다.

```bat
python -m sim_errors.make_scenarios --batch --family fillet_corner --out batches\corner
python sim_bridge\run_batch.py batches\corner --out episodes\corner --follower-urdf C:\...\opti3.urdf
python -m learning.to_lerobot episodes\t episodes\corner --out datasets\t_corner
python -m learning.train_bc datasets\t_corner --family fillet_t --out runs\mlp_t
python -m learning.train_bc datasets\t_corner --out runs\mlp_both
python -m learning.evaluate runs\mlp_t\policy.pt runs\mlp_both\policy.pt --data datasets\t_corner --episodes episodes\t episodes\corner
python -m learning.report_evals runs\*\eval.json --kind fam_fillet_corner
```

사양 수량 생성부터 평가까지는 GitHub Actions의 "Experiment" 워크플로(`.github/workflows/experiment.yml`)를 Run workflow로 돌린다. 지금 실험은 T 330 + 코너 335 운동학 에피소드로 MLP/GRU를 T만·코너만·둘 다로 학습해 두 군 시험에서 본다. 러너 시간을 많이 써서 push로는 돌지 않는다. 결과는 실행 요약의 표와 `eval-*` 아티팩트.

- 나눔은 시나리오의 `episode.split`(에피소드 단위 학습 80 · 검증 10 · 시험 10, 가혹은 모두 시험).
- 평가 결과는 `<정책 폴더>/eval.json`(`--name X`면 `eval_X.json`, 규칙 기준선은 `<rule-dir>/rule/eval.json`), 학습 곡선은 `train_log.json`.
- 높이 오차는 시작 30 mm와 나머지로도 나눠 나온다. 시작 구간 스캔으로 학습한 정책(`policy.pt`의 `prescan`)은 닫힌 루프에서도 스캔을 넣는다.
- 닫힌 루프 평가는 로봇이 명령을 바로 따라간다고 가정한다. Isaac 추종 오차는 들어가지 않는다.
  Isaac에서 정책이 직접 opti3를 모는 닫힌 루프는 `sim_bridge/policy_to_sim.py` ([sim_bridge/README.md](../sim_bridge/README.md) "학습 정책으로 몰기").

## 실험 결과

실험별 결과 기록은 [docs/experiments/](../docs/experiments/README.md)로 옮겼습니다 (2026-11-25).

## 다음 단계

- DAgger는 보류했다(사용자 결정). 지금 관측에서는 정책 출력이 다음 관측을 거의 바꾸지 않아([수량·섭동 비교](../docs/experiments/20261118-t-quantity-perturb.md)) 시뮬 DAgger로 얻을 것이 적다. 카메라 시야가 토치를 따라 움직이는 효과를 모델링하거나, 정책 출력에 따라 바뀌는 입력을 넣을 때 다시 본다.
- 실제 depth 카메라로 시연 시편 단면을 재서 점 노이즈·측정 구멍 값 맞추기. 시작 구간 스캔 자세의 노이즈도 따로 잰다.
- 이전 Isaac 기록은 `python -m sim_errors.resense`로 지금 센서 설정(스캔 포함)을 넣은 뒤 다시 변환한다.
- 실제 depth 카메라로 V홈 시편 단면을 재서 판 윗면 단차 측정이 되는지 확인.
- 코너·V홈·파이프 Isaac 1/5 묶음은 모두 통과했다(추종 95% 0.33–0.41 mm, sim_errors/README.md "Isaac 물리 검증"). 남은 것: 이 기록으로 학습·평가해 운동학 결과와 비교.
- Isaac 정책 실행기로 공용 MLP의 Isaac 닫힌 루프를 잰다 ([20261125-isaac-policy](../docs/experiments/20261125-isaac-policy.md)). 이 실행기가 공유 제어(정책 구동 중 리더 암 보정)와 개입 데이터 수집의 바탕이다.
- 파이프의 다른 자세(관 아래쪽까지 5G 전체, 세운 관 2G).
