# 모재 오차 시뮬레이션 (sim_errors)

Isaac Sim에서 모재 오차를 용접 전·중·후로 나눠 분포로 주입하고, 실제 경로를 아는 규칙 전문가가 에피소드를 만들며 매 주기 정답(보정값)을 기록한다. 설계 근거와 범위는 Claude Doc "모재 오차 모델링 계획"에 있다.

| 파일 | 하는 일 |
| --- | --- |
| `config/part_errors.yaml` | 오차 분포, 센서, 규칙 전문가 설정. 범위는 가정치이며 실측하면 이 파일만 고친다 |
| `part_errors.py` | 시나리오 샘플링, 실제·추정 경로, 정답, 모재 메시 변형, 레이저 라인 센서, 다음 패스 (numpy만 사용) |
| `expert.py` | 규칙 전문가: 접근 → 하강 → 용접(위빙) → 후퇴 토치 목표와 정답 |
| `make_scenarios.py` | 시나리오 JSON N개 생성과 정답 크기 분포 요약 |
| `../sim_bridge/expert_to_sim.py` | 규칙 전문가 → opti3 IK → Isaac Sim, 에피소드 기록(.npz) |
| `../sim_bridge/isaac_opti3_receiver.py --scenario` | Isaac Sim에 모재를 만들고 아크 진행도에 맞춰 변형을 그림 |

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

Isaac Sim 화면에서 빨간 선이 실제 이음매, 파란 선이 추정 경로다. 두 선의 차이가 정답이고, 토치는 빨간 선을 따라간다.

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
| `t`, `phase`, `arc_on` | (N,) | 시각, 단계(approach/descend/weld/retract/done), 아크 ON |
| `s`, `s_arc` | (N,) | 토치 진행도, 아크 진행도 |
| `e` | (N,3) | 정답 위치 보정 (추정 경로 좌표계: x 진행, y 좌우, z 높이) |
| `rotvec` | (N,3) | 정답 각도 보정 (x 작업각, y 진행각) |
| `gap`, `tack` | (N,) | 토치 위치의 틈새, 가접 높이 |
| `weave_amp`, `weave_hz`, `speed` | (N,) | 규칙 전문가의 위빙 진폭·주파수, 진행 속도 |
| `laser` | (N,2) | 레이저 라인 측정 (토치 앞, 좌우·높이 편차). 실패는 NaN |
| `p_true`, `p_est` | (N,3) | 토치 진행도의 실제·추정 이음매 점 (link0 기준) |
| `T_target`, `T_meas` | (N,4,4) | 토치 팁 목표 포즈, 실측 관절로 구한 포즈 |
| `q_cmd`, `q_meas`, `status` | (N,6), (N,6), (N,) | 관절 명령(안 보냈으면 NaN), 실측, FollowerStepper 상태 |
| `scenario_json` | () | 사용한 시나리오 (자동 배치된 모재 위치 포함) |

## 아직 없는 것

- 렌더링한 depth 점군으로 실제 ICP 돌리기 (지금은 ICP 잔차를 분포에서 뽑음)
- 맞대기 이음과 개선 각도, 다층 용접의 비드 형상
- 카메라 이미지 기록 (LeRobotDataset 형식 기록과 함께 추가)
