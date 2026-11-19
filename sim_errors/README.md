# 모재 오차 시뮬레이션 (sim_errors)

Isaac Sim에서 모재 오차를 용접 전·중·후로 나눠 분포로 주입하고, 실제 경로를 아는 규칙 전문가가 에피소드를 만들며 매 주기 정답(보정값)을 기록한다. 설계 근거와 범위는 Claude Doc "모재 오차 모델링 계획"에 있다.

| 파일 | 하는 일 |
| --- | --- |
| `config/part_errors.yaml` | 오차 분포, 센서, 규칙 전문가 설정. 범위는 가정치이며 실측하면 이 파일만 고친다 |
| `part_errors.py` | 시나리오 샘플링, 실제·추정 경로, 정답, 모재 메시 변형, 이음매 단면 센서(코드 이름 `laser`, 실기에서는 손목 depth 카메라 단면, 판 면 평면 맞춤), 다음 패스 (numpy만 사용) |
| `resense.py` | 기록의 단면 열만 지금 센서 모델로 다시 계산 (Isaac 재실행 없이) |
| `expert.py` | 규칙 전문가: 접근 → 하강 → 용접(위빙) → 후퇴 토치 목표와 정답. 용접 구간만 모드, 직선 모드, 시작 어긋남, 실행 섭동 |
| `make_scenarios.py` | 시나리오 JSON N개 생성과 정답 크기 분포 요약, 오차별 단독 검증 시나리오(`--single`), 학습용 묶음(`--batch`) |
| `episode.py` | 기록(.npz) 저장 공통 부분, Isaac 없이 도는 운동학 에피소드(완벽 추종 가정) |
| `check_episode.py` | 에피소드 기록(.npz) 점검 8항목과 그래프 |
| `../sim_bridge/expert_to_sim.py` | 규칙 전문가 → opti3 IK → Isaac Sim, 에피소드 하나 기록(.npz) |
| `../sim_bridge/run_batch.py` | 묶음 일괄 실행, 에피소드별 자동 점검, 묶음 통계(`batch_report.json`) |
| `../sim_bridge/isaac_opti3_receiver.py` | Isaac Sim에 모재를 만들고 아크 진행도에 맞춰 변형을 그림. 패킷으로 시나리오 교체·바로 놓기 |
| `../learning/` | 묶음 → LeRobotDataset 변환, MLP/GRU 행동 복제, 보정 0 대비 평가 ([learning/README.md](../learning/README.md)) |

## 실행 순서

저장소 루트에서, `conda activate weld` 환경 기준.

1. 분포 확인: `python -m sim_errors.make_scenarios --n 500 --summary-only`
2. 시나리오 생성: `python -m sim_errors.make_scenarios --n 20 --out scenarios` (다층 용접은 `--passes 3`)
3. Isaac Sim (터미널 1):
   ```bat
   C:\isaacsim\python.bat sim_bridge\isaac_opti3_receiver.py --usd C:\...\posco_torch.usd --scenario scenarios\part0000_pass0.json
   ```
4. 규칙 전문가 (터미널 2):
   ```bat
   python sim_bridge\expert_to_sim.py --scenario scenarios\part0000_pass0.json --follower-urdf C:\workspace\indy-ros2-jazzy\indy_description\urdf_files\opti3.urdf
   ```
   시작 자세의 토치 팁 근처에 모재를 자동 배치하고, 이음매 전체에 IK가 닿는지 확인한 뒤 시작한다. 끝나면 `episodes/part0000_pass0.npz`에 기록한다.

5. 기록 점검: `python -m sim_errors.check_episode episodes\part0000_pass0.npz` (그래프는 `conda install matplotlib` 후 같은 폴더에 .png)

Isaac Sim 화면에서 빨간 선이 실제 이음매, 파란 선이 추정 경로다. 두 선의 차이가 정답이고, 토치는 빨간 선을 따라간다.

## 학습용 묶음 일괄 생성

에피소드 생성 사양의 묶음(깨끗함 30, 단독 65, 혼합 205, 가혹 30)을 만든다. 모든 에피소드는 용접 구간만 담고,
로봇을 이음매 시작점 + 시작 어긋남(좌우·높이 ±2 mm, 작업각·진행각 ±2°) 자세에 바로 놓고 시작한다.
직선/위빙 모드는 에피소드 시작 때 정해 끝까지 고정하고 묶음마다 반반이다(직선 = 위빙 0, 속도 규칙만).
혼합 묶음의 절반에는 실행 섭동(좌우·높이 σ 0.5 mm, 각도 σ 1°, 0.5–2초 반 코사인)을 넣고 정답은 섭동 없는 값으로 둔다. 토치 자세를 정책 입력에서 뺀 뒤로는 섭동이 정책 관측을 바꾸지 않아 학습 효과가 없다(learning/README.md "수량·섭동 비교"). Isaac 추종 상태를 다양하게 하려고 그대로 둔다.
개수·비율·어긋남·섭동은 `config/part_errors.yaml`의 `episode` 섹션.

1. 시나리오: `python -m sim_errors.make_scenarios --batch --scale 0.2 --out batches/t_small` (1/5 수량 66개. 전체는 `--scale` 빼기)
2. Isaac Sim (터미널 1, 한 번만 띄움. `--scenario` 없이):
   ```bat
   C:\isaacsim\python.bat sim_bridge\isaac_opti3_receiver.py --usd C:\...\posco_torch.usd --lockstep
   ```
3. 일괄 실행 (터미널 2):
   ```bat
   python sim_bridge\run_batch.py batches\t_small --out episodes\t_small --follower-urdf C:\workspace\indy-ros2-jazzy\indy_description\urdf_files\opti3.urdf
   ```
   중단해도 같은 명령으로 이어서 만든다(있는 .npz는 건너뜀). 끝나면 강도별 통계와 `episodes\t_small\batch_report.json`.
   점검에 실패한 에피소드는 보고서에 이유가 남고 학습 변환에서 빠진다. 끝나지 않은 실행은 `*.failed.npz`.
4. Isaac 없이 흐름만 시험: `python sim_bridge/run_batch.py batches/t_small --out episodes/t_small_kin --kinematic --jobs 4`

receiver 쪽 동작: 패킷의 `scene.scenario`가 바뀌면 모재를 다시 만들고, `reset_q`가 오면 로봇을 그 관절 자세로 순간 이동시킨다.
답장의 시뮬 시각(`t`)으로 전문가를 한 주기씩 진행하므로, Isaac이 멈추면 전문가도 멈춘다(single_place에서 0.34초 끊김으로 생긴
4.8 mm 오차 원인). `--lockstep`이면 receiver가 전문가의 응답을 받은 뒤 다음 물리 스텝을 밟아, `--headless`와 같이 쓸 때
실시간보다 빠르다. 물리 주기는 `--physics-hz`(기본 60, 이전과 같음)이고 전문가 주기도 같아진다.

## 단면 센서 모델 (평면 맞춤)

토치 앞 30 mm에서 카메라가 보는 작은 패치(이음매 방향 10 mm, 판마다 뿌리에서 15 mm, 1 mm 간격)의 두 판 면 위 점을 실제 변형된 모재에서 뽑고, 점마다 노이즈(σ 0.2 mm)와 측정 구멍(10%)을 넣은 뒤 판마다 평면을 맞춘다. 두 평면의 교선이 뿌리선이고, 그 위치가 좌우·높이, 두 평면이 만드는 이음매 좌표계(x = 교선, z = 법선 이등분)의 회전이 작업각·진행각 측정이다. 틈새 폭은 정답 + 노이즈(σ 0.1 mm). 패치 전체 실패는 5%. 설정은 `config/part_errors.yaml`의 `sensors`.

작업각(이음매 축 둘레 회전)은 뿌리 위치를 바꾸지 않으므로 이 각도 측정 없이는 정책이 배울 수 없다. 기본 설정에서 측정 노이즈는 좌우·높이 약 0.04 mm, 작업각 약 0.15°, 진행각 약 0.3°.

**용접 전 시작 구간 스캔 (`prescan_mm`, 기본 30):** 단면은 토치 앞 30 mm에서만 재므로 이음매 시작 30 mm는 용접 중에 한 번도 보이지 않는다. 첫 가접(s 0.02~0.08)이 이 구간에 있어 남은 높이 오차의 대부분이 여기서 나왔다(learning/README.md "남은 높이 오차의 원인"). 그래서 아크를 켜기 전에 손목 카메라로 시작 구간을 한 번 훑는다고 보고, `PartErrorModel.prescan()`이 s_arc = 0 상태의 단면을 0.5 mm 간격으로 같은 평면 맞춤 모델(노이즈·측정 실패 포함)로 잰다. 스캔은 시나리오만으로 정해지므로 에피소드 기록에는 없고, 변환(`learning.to_lerobot`)과 평가가 시나리오에서 다시 계산해 버퍼를 먼저 채운다. 이전 시나리오(설정에 `prescan`이 없음)는 스캔이 없고, `resense`로 지금 설정을 넣으면 생긴다.

센서 모델을 바꿨을 때 이미 만든 에피소드는 Isaac을 다시 돌리지 않고 단면 열만 다시 계산한다(단면 값은 시나리오와 진행도만으로 정해짐).

```bat
python -m sim_errors.resense episodes\t_small --out episodes\t_small_prof
python sim_bridge\run_batch.py batches\t_small --out episodes\t_small_prof --report-only
```

## 오차별 단독 검증

여러 오차가 섞인 에피소드로는 하나가 틀려도 묻히므로, 오차를 하나만 크게 과장해서 켠 시나리오로 따로 확인한다.

1. `python -m sim_errors.make_scenarios --single all --out scenarios_single` → 13개 `single_<오차>.json`과 기대 정답 표(진행도 0, 0.25, 0.5, 0.75, 1에서 좌우·높이·작업각·진행각·틈새·가접)를 출력
2. 오차마다 receiver(`--scenario scenarios_single\single_<오차>.json --kp 1e5 --kd 2e3`)와 `expert_to_sim.py`를 실행
3. `check_episode`로 8항목을 확인하고, 그래프가 1의 기대 정답 표와 같은 모양인지, Isaac 화면이 계획 문서 "오차별 단독 검증" 표의 화면 기대와 같은지 본다

단독 시나리오는 단면 센서(`laser`)의 노이즈와 측정 실패도 0이라 측정 점이 정답 선 위에 정확히 겹친다(토치 앞 30 mm에서 잰 값이라 그만큼 앞에 찍힘).

## 기록 점검 (check_episode.py)

| 항목 | 통과 조건 | 실패하면 |
| --- | --- | --- |
| 단계 순서 | approach → descend → weld → retract → 완료 (용접 구간만 담은 에피소드는 weld → 완료) | 중단됐거나 IK 실패로 멈춤 |
| IK 상태 | 모든 주기 `ok` (95% 넘으면 주의) | `approx`·`ik_failed`가 많으면 모재가 도달 범위 경계에 있음 |
| 진행도 | s가 줄지 않고 1까지 | 에피소드가 끝나지 않음 |
| 정답 재계산 | 기록된 정답 = 시나리오로 다시 계산한 값 | 시나리오 파일과 기록이 어긋남 (코드 버전 차이) |
| 목표 기하 | 전문가 목표 = 실제 이음매 + standoff(+가접) + 위빙, 토치 방향 = 이음매 쪽 + 미는 각, 보낸 목표 = 전문가 목표 + 기록된 어긋남 | 규칙 전문가 버그 |
| 추종 오차 | 용접 중 토치 팁(실측 관절의 URDF FK)과 목표 차이 95%가 2 mm 미만 (6 mm 미만은 주의) | Isaac 관절이 명령을 못 따라감: 게인, 속도 제한 확인 |
| 단면 센서 (`laser`) | 측정 실패율이 설정과 비슷, 좌우·높이·틈새 폭의 정답 대비 편차가 노이즈 수준 | 센서 모델 문제 |
| 용접 시간 | 길이 / 속도 범위 안 | 진행 속도 규칙 문제 |

추종 오차는 URDF FK로 계산하므로, Isaac 장면의 토치 팁이 URDF와 다르면 여기서는 드러나지 않는다. 그건 Isaac 화면에서 토치 끝이 빨간 선 위를 지나가는지로 확인한다.

## 모델 요약

- 모재: T이음 필릿. 모재 좌표계 원점 = 이음매 시작점, x = 이음매 방향, 바닥 판 윗면 z = 0, 웹 판 용접 쪽 면 y = 0.
- 추정 경로 = 명목 경로 × (배치 자세 × ICP 잔차). 용접 전에 정해지고 고정.
- 실제 경로 = 배치 자세 × (명목 + 형상 오차 + 열변형 + 밀림). 열변형과 밀림은 시간이 아니라 아크 진행도 s_arc의 함수.
- 정답 e = 추정 경로 좌표계에서 본 (실제 − 추정). y = 좌우, z = 높이. 각도 정답 = 두 좌표계의 회전 차이(x = 작업각, y = 진행각).
- 틈새는 이음매 위치를 바꾸지 않고, 규칙 전문가의 위빙 진폭·진행 속도에만 영향을 준다(`expert` 설정, WPS 값으로 바꿀 것).
- 다음 패스: 이번 패스의 잔류 각변형·휨을 다음 패스의 용접 전 형상으로 넘긴다. 틈새와 가접은 0.

## 기록 파일 (.npz) 필드

주기마다 한 행. 단위 m, rad.

| 필드 | 모양 | 뜻 |
| --- | --- | --- |
| `t`, `phase`, `arc_on`, `done` | (N,) | 시각, 그 주기 목표를 만든 단계(approach/descend/weld/retract), 아크 ON, 에피소드 완료 |
| `s`, `s_arc` | (N,) | 토치 진행도, 아크 진행도 |
| `e` | (N,3) | 정답 위치 보정 (추정 경로 좌표계: x 진행, y 좌우, z 높이) |
| `rotvec` | (N,3) | 정답 각도 보정 (x 작업각, y 진행각) |
| `gap`, `tack` | (N,) | 토치 위치의 틈새, 가접 높이 |
| `weave_amp`, `weave_hz`, `speed` | (N,) | 규칙 전문가의 위빙 진폭·주파수, 진행 속도 |
| `laser`, `laser_valid` | (N,5), (N,) | 이음매 단면 측정 (토치 앞 30 mm의 좌우·높이 편차 m, 틈새 폭 m, 작업각·진행각 rad). 실기에서는 손목 depth 카메라 단면. 측정 실패는 NaN이고 `laser_valid`가 False. 이전 기록은 (N,2), (N,3)이며 `resense`로 다시 계산 |
| `p_true`, `p_est` | (N,3) | 토치 진행도의 실제·추정 이음매 점 (link0 기준) |
| `T_target`, `T_expert`, `T_meas` | (N,4,4) | 보낸 토치 팁 목표(어긋남 포함), 섭동 없는 전문가 목표, 실측 관절로 구한 포즈 |
| `perturb` | (N,4) | 보낸 목표에 더한 어긋남 (이음매 좌표계 좌우·높이 m, 작업각·진행각 rad) = 시작 어긋남(지수 감소) + 실행 섭동 |
| `t_sim` | (N,) | receiver 시뮬 시각 (Isaac 실행만) |
| `q_cmd`, `q_meas`, `status` | (N,6), (N,6), (N,) | 관절 명령(안 보냈으면 NaN), 실측, FollowerStepper 상태 |
| `scenario_json` | () | 사용한 시나리오 (자동 배치된 모재 위치, `episode` 설정 포함) |
| `meta_json` | () | 시편군, 직선/위빙 모드, 강도, 나눔, git 커밋, 실행 결과 |

## 아직 없는 것

에피소드 생성 사양(Claude Doc "에피소드 생성 사양")과 지금 코드의 차이다. 구현 순서는 그 문서에 있다.

실제로 용접할 이음 형태는 아직 정해지지 않았다. 그래서 T 필릿 하나로 일괄 생성부터 학습·평가까지 먼저 한 바퀴 돌려 파이프라인을 확인하고(학습용 묶음 일괄 생성, [learning/README.md](../learning/README.md)), 그다음 사양 순서대로 코너 필릿, 맞대기 V홈, 파이프 원주를 더한다.

직선/위빙 모드는 정해져 구현했다: 작업 시작 때 WPS의 일부로 정하고 에피소드 안에서는 바꾸지 않음, 묶음마다 반반, 사인파 위빙, 직선 모드는 위빙 진폭 0이고 틈새가 커지면 속도만 줄임. 모드는 정책 조건 입력(원-핫)이다.

- 고정 배치: 지금은 홈 자세(`--home-q`)의 토치 팁 근처 고정 자리다. 도달 범위 지도로 여유가 가장 큰 자리를 고르는 일은 아직.
- 작업대·지그 메시와 충돌 검사: 모재는 표시 전용이다.
- 규칙 상태기계 검사용 전체 주기(대기~후퇴) 에피소드를 묶음에 섞기: `--batch`는 용접 구간만 만든다. 전체 주기는 `episode` 없는 시나리오(`--n`)로 만든다.
- 시편군 확장: 지금은 T 필릿 하나다. ㄱ자 코너 필릿, 맞대기 V홈, 파이프 원주를 더한다.
- 단독 묶음의 오차 서명 자동 비교 (지금은 `--single` 검증에서 눈으로 본다)
- 렌더링한 depth 점군으로 실제 ICP 돌리기 (지금은 ICP 잔차를 분포에서 뽑음)
- 다층 용접의 비드 형상
- 카메라 이미지 기록 (기록 전용, 정책 입력 아님)
- 정책을 Isaac에서 모는 실행기 (추종 오차가 들어간 닫힌 루프 평가). DAgger는 보류 (learning/README.md "다음 단계")
