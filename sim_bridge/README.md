# 리더 암 → Isaac Sim opti3 (ROS 없이)

리더 암(SO-102)으로 Isaac Sim 안의 opti3 토치를 움직여 보는 구성. 프로세스 두 개가 localhost UDP로 연결된다.

```
[leader_to_sim.py]  리더 서보 → 리더 FK → 클러치·상대 매핑 → opti3 IK ──관절 목표(9870)──▶ [isaac_opti3_receiver.py]
   (conda 환경)                                                        ◀──실측 관절(9871)──   (Isaac Sim 파이썬)
```

계산은 ROS 노드와 같은 코드·config(`weld_teleop/config/*.yaml`)를 쓴다. 추종 오차·입력 끊김 시 명령 정지도 같다.

## 1. 환경 준비 (한 번만)

pinocchio는 pip용 Windows 패키지(`pin`)가 없어서 conda-forge로 설치한다.

1. [Miniforge](https://github.com/conda-forge/miniforge) 설치
2. Miniforge Prompt에서:
   ```bat
   conda create -n weld -c conda-forge python=3.12 pinocchio numpy pyyaml
   conda activate weld
   pip install feetech-servo-sdk
   ```

Isaac Sim 쪽은 Isaac Sim에 들어 있는 파이썬을 쓰므로 설치할 것이 없다.

## 2. 실행

터미널 1: Isaac Sim (경로는 설치 위치에 맞게)
```bat
C:\isaacsim\python.bat C:\workspace\Welding-Robot\sim_bridge\isaac_opti3_receiver.py   # 장면 기본값 assets\isaac\opti3_welding_scene\posco_torch.usd
```
장면이 열리면 opti3가 시작 자세(토치가 로봇 앞에서 아래를 향함, `--initial-q`로 변경 가능)로 선다.

터미널 2: 리더 (`conda activate weld` 후, 저장소 루트에서)
```bat
python sim_bridge\leader_to_sim.py --mock --follower-urdf C:\workspace\indy-ros2-jazzy\indy_description\urdf_files\opti3.urdf
```
먼저 `--mock`(사인파 입력)으로 연결을 확인한 뒤, `--mock` 대신 `--port COM3`로 실제 리더를 쓴다.

## 3. 조작

- **Enter** = 클러치 ON/OFF (시작은 OFF). ON 순간의 리더·opti3 자세를 기준으로 리더의 상대 이동만 따라간다.
  OFF 상태에서 리더를 편한 자세로 옮긴 뒤 다시 ON 하면 튀지 않는다.
- **범위 밖**: 리더가 opti3 도달 범위 밖(또는 특이점 쪽)으로 가면 토치는 갈 수 있는 경계까지만 가서 멈추고
  상태가 `out_of_reach`가 된다. 리더를 안쪽으로 되돌리면 클러치를 다시 누르지 않아도 그 자리에서 바로 따라간다.
  (범위 밖에서 움직인 만큼은 버려지므로 리더와 토치의 상대 위치가 그만큼 바뀐다)
- **목표 마커**: Isaac Sim에 리더가 가리키는 토치 팁 목표 포즈가 축 마커로 보인다(x 빨강, y 초록, z 파랑 = 토치 방향,
  노란 공 = 팁). 마커와 토치 팁이 같이 움직이면 매핑이 맞는 것이다. 끄려면 receiver에 `--no-marker`.
- 상태 줄: `ok` 정상, `out_of_reach`(위 설명), `clutch_off`, `waiting`(Isaac Sim 실측 대기),
  `joint_state_timeout`(Isaac Sim 끊김), `tracking_error`(명령과 실측 차이가 큼 → 재동기화),
  `out_of_limits`(실측 관절이 URDF 리밋 밖), `leader_lost`(리더 읽기 실패)
- 그리퍼 %는 표시만 한다(용도 미정).

## 4. 모재 오차 시나리오와 규칙 전문가

`--scenario`로 모재 오차 시나리오를 주면 receiver가 모재(바닥 판·웹 판·가접)를 만들고, 리더 대신 `expert_to_sim.py`가 실제 이음매를 따라 토치를 움직이며 정답을 기록한다. 실행 순서와 기록 형식은 [sim_errors/README.md](../sim_errors/README.md).

학습용 묶음은 receiver를 `--scenario` 없이(대량이면 `--headless --lockstep`) 한 번 띄워 두고 `run_batch.py`로 연달아 만든다. 시나리오 교체와 로봇 바로 놓기는 패킷으로 한다. receiver 관절 게인 기본값은 검증한 `kp 1e5 / kd 2e3`이다. 이전 값이 필요하면 `--kp 1e4 --kd 1e3`.

## 5. 학습 정책으로 몰기 (Isaac 정책 실행기)

`policy_to_sim.py`는 규칙 전문가 대신 학습한 보정 정책(`learning.train_bc`의 `policy.pt`, 또는 규칙 기준선 `rule`)이 매 주기
토치 목표를 내서 opti3를 몬다. `expert_to_sim.run_episode`를 그대로 쓰고 목표를 내는 부분만 `learning/policy_driver.py`로 바꿔서,
모재 자동 배치·이음매 IK 도달 확인·시작 자세 바로 놓기·lockstep·기록은 `run_batch.py`와 같다. 정책 관측의 토치 자세는 Isaac 실측이다.

```bat
set K=outputs\experiments\20261123-kin-4fam
python sim_bridge\policy_to_sim.py --policy %K%\runs\mlp_all\policy.pt --scenarios %K%\episodes ^
    --kin-eval %K%\runs\mlp_all\eval.json --exp 20261125-isaac-policy --follower-urdf C:\...\opti3.urdf
```

- 시나리오는 `*.json` 또는 에피소드 `*.npz` 폴더에서 `--split`(기본 test)만 쓴다. 운동학 실험의 에피소드 폴더를 주면 운동학 평가와 같은 시험 시나리오다.
  `--only`, `--per-family N`, `--limit N`으로 줄인다. 화면 확인은 receiver를 `--headless` 없이 띄우고 `--only pipe_clean --limit 1`처럼.
- 출력은 `<실험>/runs/<정책 이름>/isaac/`: 에피소드 `.npz`(정책 보정 `policy_c`, 행동 `policy_action`, 관절 실측 포함), receiver가 읽는 시나리오 사본
  `scenarios/`, 요약 `eval_isaac.json`. 도달 실패로 시작 못 한 시나리오는 `skipped.json`에 남고 다음 실행에서 건너뛴다.
- 오차는 **측정**(실측 토치 FK − 실제 이음매, 좌우는 명령한 위빙을 뺌)과 **명령**(정책 보정 − 실제 이음매, 운동학 평가와 같은 정의)
  두 가지다. `--kin-eval`을 주면 같은 에피소드의 운동학 오차를 옆에 놓는다. 추종 95%(실측 팁과 목표 거리)와 뒤짐(이음매 방향으로
  실측이 명령보다 늦은 평균)도 낸다. `--report-only`는 Isaac 없이 저장된 에피소드로 요약만 다시 낸다.
- 완벽 추종(실측 = 목표)이면 명령 오차가 운동학 닫힌 루프(`learning.evaluate`)와 똑같이 나온다 (`learning/test/test_policy_driver.py`).
- 위빙 모드는 실측이 위빙을 조금 늦게 따라가 측정 좌우 오차에 위빙 주파수 흔들림이 남는다. 위빙 한 주기 평균한 **위빙 중심 좌우**도 낸다.

**팔·토치–모재 간격 점검** (`clearance_check.py`): Isaac 장면의 모재는 표시 전용이라 충돌이 없다. 저장된 에피소드(`run_batch.py`,
`policy_to_sim.py`)의 관절 실측으로 토치·팔 표면 점에서 모재 판까지 최소 거리를 재고, 음수(뚫음)를 알려 준다.
팔·토치 모양은 장면의 표시 메시(`geometry/opti3_scene_points.npz`, `tools/extract_scene_points.py`로 `posco_torch.usd`에서 뽑음)이고,
장면의 링크·tcp·토치 팁 좌표계는 `opti3.urdf`와 같다(관절 0에서 확인). 모재는 명목 단면이고 변형(mm 단위)은 넣지 않는다.

```bat
python sim_bridge\clearance_check.py outputs\experiments\20261125-isaac-policy\runs\mlp_all\isaac --follower-urdf C:\...\opti3.urdf
```

## 6. 공유 제어: 기본 정책 + 리더 보정, 개입 기록

`shared_to_sim.py`는 정책 실행기와 같은 시나리오·배치·lockstep 위에서, 고정한 기본 정책(base-v1, [docs/policies.md](../docs/policies.md))이
용접하는 동안 SO-102 리더로 보정을 더하고 개입 에피소드를 기록한다. 규칙은 `learning/shared_control.py`.
로봇 보정 = 기본 정책 + 잔차 모델(아직 없음, 0) + 사람 보정이고, 기본 정책은 개입 데이터로 다시 학습하지 않는다.

```bat
:: 터미널 1 (화면을 보며 조작하므로 --headless 없이)
C:\isaacsim\python.bat sim_bridge\isaac_opti3_receiver.py --lockstep
:: 터미널 2
set K=outputs\experiments\20261123-kin-4fam
python sim_bridge\shared_to_sim.py --policy %K%\runs\mlp_all\policy.pt --base-tag base-v1 ^
    --scenarios %K%\episodes --per-family 1 --port COM3 --exp 20261125-shared ^
    --follower-urdf C:\workspace\indy-ros2-jazzy\indy_description\urdf_files\opti3.urdf
```

- 조작 (터미널 2에 포커스): 로봇이 시작 자세에 놓이면 **s = 시작 트리거**, n = 건너뜀. 용접 중 **리더 그리퍼를 쥐고 있는 동안 클러치 ON**
  (`--clutch key`면 c로 켜고 끔, `--mock`은 항상 key), **r = 오프셋 초기화**(1초 램프), q = 멈춤(대기 중이면 종료).
- 매핑: 클러치를 잡은 순간 대비 리더 끝 변화를 리더 베이스 축 그대로 이음매 좌표로 읽는다 (x 진행 = 버림, y 좌우, z 높이,
  x·y축 회전 = 작업각·진행각). 위치 0.25배, 회전 0.5배, 한도 ±5 mm, ±10°. 리더를 놓는 방향이 다르면 `--leader-to-seam-rpy`.
  `--gain 1,1,0,0`처럼 축별 권한을 끌 수 있다.
- 놓으면 그때 보정이 유지 오프셋으로 남는다. 리더 읽기가 끊겨도 유지하고, 다시 읽히면 그 자세에서 기준을 잡는다.
- 출력 `<실험>/runs/base-v1/shared/<시나리오>__<시각>.npz` (멈추거나 못 끝내면 `.failed.npz`). 기본 정책 `base_c`, 잔차 `residual_c`,
  사람 보정 `human_c`(잔차 정답, 유지 오프셋 포함), 합계 `policy_c`, 클러치·초기화 표시, 10 Hz 정책 관측 `obs`와 meta의 정책 경로·sha256·
  고정 이름·사람 보정 설정. 끝나면 `eval_isaac.json`(정책 실행기와 같은 오차, 명령 = 합계)과 `interventions.json`(클러치 횟수·시간, 초기화, 사람 보정 크기).
- 리더 없이 확인: `--port COM3` 대신 `--mock` (사인파 리더, c로 클러치).

## 확인 안 된 것

- `opti3.urdf` 경로: indy-ros2-jazzy 안의 실제 위치로 바꿔 쓴다(xacro만 있으면 알려 줄 것).
- Isaac Sim 장면의 관절 영점·방향이 `opti3.urdf`와 같은지. 다르면 토치가 리더와 다른 방향으로 움직인다.
- 위치 스케일·베이스 방향은 `mapping.yaml`(`position_scale`, `leader_to_follower_rpy`)에서 조정.
