# 보정 정책 학습 (learning)

시뮬 규칙 전문가 에피소드 묶음으로 기준 보정 정책(작은 MLP 또는 GRU 행동 복제)을 학습하고, 보정 0 기준선 대비 정답 오차로 평가한다. 설계 근거는 [아키텍처](../README.md), Claude Doc "에피소드 생성 사양"의 "기록 형식"과 "구현 순서".

| 파일 | 하는 일 |
| --- | --- |
| `features.py` | 관측·행동 정의, 단면 거리 버퍼. 변환과 평가가 같은 코드를 쓴다 |
| `to_lerobot.py` | 에피소드 .npz(100 Hz) → `frames.npz`(10 Hz, 학습용)와 LeRobotDataset |
| `train_bc.py` | MLP/GRU 행동 복제, 입력 묶음 선택, `policy.pt` 저장 |
| `evaluate.py` | 열린 루프(기록 관측으로 예측) + 닫힌 루프(운동학, 정책이 직접 토치를 몸) 평가 |

## 관측과 행동

관측 43개 (단위 mm, deg)

| 묶음 (`--obs`) | 개수 | 내용 |
| --- | --- | --- |
| `section` | 36 | 단면 거리 버퍼: 토치 뒤 10 mm ~ 앞 30 mm, 5 mm 간격 9점 × (좌우, 높이, 틈새 폭) + 9점 유효 표시. 단면은 토치 앞 30 mm에서만 재므로 잰 값을 이음매 위치와 함께 쌓아 꺼낸다. 값은 추정 경로 기준 편차(토치 기준 측정 + 그때 토치 자세) |
| `torch` | 4 | 추정 경로 기준 토치 자세 어긋남 (좌우, 높이, 작업각, 진행각). 위빙·섭동·시작 어긋남 포함 |
| `progress` | 1 | 진행도 s |
| `mode` | 2 | 위빙/직선 원핫 |

행동 7개 (정답 = 규칙 전문가): 좌우, 높이(CTWD, 가접 올림 포함) [mm], 작업각, 진행각 [deg], 진행 속도 [mm/s], 위빙 진폭 [mm], 위빙 주파수 [Hz].

## 실행 순서

저장소 루트에서. 학습·변환은 `pip install torch "lerobot[dataset]"` (lerobot은 Python 3.12 이상, conda `weld` 환경에서 됨). lerobot이 없으면 `frames.npz`만 만든다.

```bat
python -m sim_errors.make_scenarios --batch --scale 0.2 --out batches\t_small
python sim_bridge\run_batch.py batches\t_small --out episodes\t_small --follower-urdf C:\...\opti3.urdf
python -m learning.to_lerobot episodes\t_small --out datasets\t_small
python -m learning.train_bc datasets\t_small --model mlp --out runs\mlp_full
python -m learning.train_bc datasets\t_small --model mlp --obs section,progress,mode --out runs\mlp_notorch
python -m learning.train_bc datasets\t_small --model gru --out runs\gru_full
python -m learning.evaluate runs\mlp_full\policy.pt runs\mlp_notorch\policy.pt runs\gru_full\policy.pt --data datasets\t_small --episodes episodes\t_small
```

Isaac 없이 전체 흐름만 볼 때는 두 번째 줄을 `--kinematic --jobs 4`(완벽 추종 가정)로 바꾼다.

- 나눔은 시나리오의 `episode.split`(에피소드 단위 학습 80 · 검증 10 · 시험 10, 가혹은 모두 시험).
- 평가 결과는 `<정책 폴더>/eval.json`, 학습 곡선은 `train_log.json`.
- 닫힌 루프 평가는 로봇이 명령을 바로 따라간다고 가정한다. Isaac 추종 오차는 들어가지 않는다.

## 첫 한 바퀴 결과 (운동학 데이터, 2026-11-17)

Isaac 대신 운동학 에피소드 66개(1/5 수량: 깨끗함 6, 단독 13, 혼합 41, 가혹 6)로 흐름 전체를 돌렸다. 학습 48, 검증 6, 시험 12(가혹 6 포함). 학습은 CPU로 정책 하나에 10초 안쪽.

닫힌 루프 RMS, 시험 12개 전체 (정책 / 보정 0, 개선율)

| 정책 | 좌우 mm | 높이 mm | 작업각 deg | 진행각 deg |
| --- | --- | --- | --- | --- |
| MLP, 전체 입력 | 0.24 / 1.30 (82%) | 0.49 / 1.31 (62%) | 0.27 / 0.23 (−17%) | 0.67 / 0.68 (2%) |
| MLP, 토치 자세 뺌 | 0.21 / 1.30 (83%) | 0.41 / 1.31 (69%) | 0.26 / 0.23 (−15%) | 0.66 / 0.68 (3%) |
| GRU, 전체 입력 | 0.24 / 1.30 (82%) | 0.64 / 1.31 (51%) | 0.33 / 0.23 (−42%) | 0.73 / 0.68 (−7%) |
| GRU, 토치 자세 뺌 | 0.30 / 1.30 (77%) | 0.48 / 1.31 (63%) | 0.25 / 0.23 (−9%) | 0.68 / 0.68 (1%) |

이 결과로 본 것 (데이터가 작아 방향만 본다)

- 좌우·높이는 배운다. 보정 0 대비 좌우 오차 약 80%, 높이 60–70% 줄었다.
- 각도는 못 배운다. 단면 버퍼에는 좌우·높이·틈새만 있어 작업각·진행각을 알 단서가 거의 없다. 정답 각도도 0.1–0.9°로 작다. 단면 프로파일(두 판 면의 기울기)을 입력에 넣거나, 각도를 행동에서 빼고 규칙(이음매 좌표계 그대로)으로 두는 것 중 하나를 골라야 한다.
- 토치 자세 입력은 지름길을 만든다. 열린 루프에서는 토치 자세가 있는 쪽이 더 좋지만(직전 정답이 토치 자세에 그대로 들어 있어서), 닫힌 루프 높이 오차는 오히려 나빠진다. 기준 정책은 토치 자세를 빼고(또는 위빙 위상만 남기고) 시작하는 쪽을 추천한다.
- GRU가 MLP보다 낫지 않다. 지금 크기에서는 MLP로 시작한다.
- 수량: 학습 손실이 검증 손실보다 훨씬 낮아(과적합) 1/5 수량은 부족하다. 사양대로 군당 300개로 늘린 뒤 다시 비교한다.

## 다음 단계

- Isaac 에피소드로 같은 비교 (추종 오차와 처짐이 들어간 실제 기록)
- DAgger: 정책이 Isaac(또는 운동학)에서 몰고 전문가가 정답을 다는 실행기. 닫힌 루프 평가 코드(`evaluate.closed_loop`)를 기록하도록 넓히면 된다.
- 각도 입력(단면 프로파일) 또는 각도 행동 제거 결정
