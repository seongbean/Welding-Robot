# weld_teleop

SO-101(개조형) 리더 → Neuromeka Indy opti3 팔로워, **엔드이펙터(TCP) 기반** 텔레오퍼레이션 골격 (ROS2 Jazzy, ament_python).

## 노드 구성

```
[leader_state_node] ──/leader/joint_states──▶ [leader_fk_node] ──/leader/ee_pose──┐
        │                                                                           ▼
        └──/leader/aux (그리퍼 축, 0~1) ─────────────────────────────▶ [ee_mapping_node] ◀── /teleop/clutch (Bool)
                                                                            │      ▲
                                              /follower/aux_command ◀───────┤      │ /follower/ee_pose
                                                                            ▼      │
                                                    /follower/ee_target ──▶ [follower_ik_node] ◀── /joint_states (opti3)
                                                                                   │
                                                          /follower/ik_preview (dry_run) / 관절 명령 토픽
```

| 노드 | 역할 | 리더 축 수 변경 시 |
|---|---|---|
| leader_state_node | 서보 읽기 → 관절각(rad) | config만 수정 (ID, offset, sign) |
| leader_fk_node | URDF FK → 리더 TCP 포즈 | config만 수정 (URDF, joint_names) |
| ee_mapping_node | 클러치·상대 매핑·스케일·속도 제한·작업공간 제한 | `orientation_mode`만 확인 |
| follower_ik_node | opti3 DLS IK → 관절 명령 | 변경 없음 |

리더 축 수(5관절/6관절)는 `config/leader_so101_*.yaml` 프로필로 분리되어 있고, 코드는 공통입니다.

## 의존성

```bash
sudo apt install ros-jazzy-pinocchio      # 또는 pip install pin
pip install feetech-servo-sdk             # 실기 리더(Feetech)만 필요
```
opti3 URDF는 `indy-ros2`(jazzy-indyDCP3)의 `indy_description`을 같은 워크스페이스에서 빌드하면 자동으로 찾습니다.

## 실행

```bash
colcon build --packages-select weld_teleop && . install/setup.bash

# 1) 하드웨어 없이 파이프라인 확인: mock 리더 + 자동 클러치 + loopback 팔로워
ros2 launch weld_teleop teleop.launch.py leader_profile:=5dof test_mode:=true \
    leader_urdf:=/abs/path/so101_new_calib.urdf
ros2 topic echo /follower/ik_preview

# 2) 실기 (opti3 bringup을 먼저 띄운 뒤)
ros2 launch indy_driver indy_bringup.launch.py indy_type:=opti3 indy_ip:=192.168.x.x
ros2 launch weld_teleop teleop.launch.py leader_profile:=7axis   # so102.urdf 자동 사용
ros2 topic pub /teleop/clutch std_msgs/msg/Bool "{data: true}" --once
```

`follower_opti3.yaml`의 `dry_run: true`가 기본값입니다. RViz에서 `/follower/ik_preview`로 동작을 확인한 다음에 false로 바꾸세요.

## 테스트

ROS 없이 기구학(FK/IK)과 EE 매핑 계산을 검증합니다. GitHub Actions에서 푸시·PR마다 자동 실행됩니다.

```bash
pip install pin numpy pytest
pytest weld_teleop/test        # 저장소 루트에서
```

실측값으로 config나 URDF를 바꾼 뒤에도 이 테스트가 통과하는지 확인하세요.

## 확인/결정이 필요한 항목 (TODO)

- [x] 개조형 URDF 수령 (`urdf/so102.urdf`, wrist_yaw 추가된 팔 6관절 + 그리퍼)
- [x] **SO-102 서보 ID:** pan=1 … roll=6, gripper=7 (`leader_probe.py`로 확인). lerobot은 wrist_yaw를 읽지 못해 feetech 드라이버 사용
- [x] SO-102 URDF yaw→roll 오프셋 부호: -z로 확정 (2026-10-06 실측)
- [ ] **SO-102 URDF 임시 가정:** yaw 회전 방향·yaw 리밋 (`so102_urdf_check.py`)
- [ ] **SO-101 URDF의 joint/frame 이름:** 현재 config 값은 추정값이므로 실제 파일과 대조
- [x] 리더 캘리브레이션 (SO-102): `offsets`·`signs`·그리퍼 `aux_range` 실측 기록 (2026-10-06)
- [ ] **opti3 명령 인터페이스:** `ros2 control list_controllers`로 컨트롤러 이름을 확인한 뒤 `command_topic`/`output_type` 설정. 대안으로 indy_moveit `servo_mode`(MoveIt Servo)에 포즈 명령을 보내는 방식도 있음
- [ ] **토치 팁 오프셋:** Isaac Sim 장면의 CAD 값(`weld_tip`)을 `tool_xyz/rpy`에 넣음. 실물 와이어 돌출 길이와 같은지 실측 확인 필요
- [ ] **베이스 정렬:** `leader_to_follower_rpy`, `position_scale`, `workspace_min/max`
- [ ] **그리퍼(aux) 축 용도:** `ee_mapping_node._on_aux` (예: 아크 ON/OFF 트리거)
- [ ] **클러치 입력 장치:** 풋페달·버튼 등 → `/teleop/clutch`

## 모방학습 데이터 기록 권장 토픽

리더의 축 수와 무관한 형식으로 남기려면 다음 토픽을 기록하세요.

```bash
ros2 bag record /joint_states /follower/ee_pose /follower/ee_target /follower/aux_command /teleop/clutch
```

`/leader/joint_states`는 디버깅용으로만 쓰고, 학습 입력으로는 쓰지 않는 것을 권장합니다.
