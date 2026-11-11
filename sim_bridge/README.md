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
C:\isaacsim\python.bat C:\workspace\Welding-Robot\sim_bridge\isaac_opti3_receiver.py --usd C:\...\opti3_welding_scene\posco_torch.usd
```
`posco_torch.usd`는 로봇(`robot/opti3/opti3.usd`)과 토치(`assets/...`)를 상대 경로로 참조한다.
usd 파일만 다른 폴더로 복사하면 로봇이 안 불러와져 `/World/indy` articulation을 못 찾으므로,
`robot/`, `assets/`가 옆에 있는 원래 장면 폴더의 usd를 지정한다(수신기가 시작 시 끊긴 참조를 알려 준다).

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

## 확인 안 된 것

- `opti3.urdf` 경로: indy-ros2-jazzy 안의 실제 위치로 바꿔 쓴다(xacro만 있으면 알려 줄 것).
- Isaac Sim 장면의 관절 영점·방향이 `opti3.urdf`와 같은지. 다르면 토치가 리더와 다른 방향으로 움직인다.
- 위치 스케일·베이스 방향은 `mapping.yaml`(`position_scale`, `leader_to_follower_rpy`)에서 조정.
