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
opti3 URDF는 `indy-ros2`(jazzy-indyDCP3)의 `indy_description/urdf_files/opti3.urdf`를 `urdf/opti3.urdf`로 복사해 두었습니다
(`indy_description`은 `urdf_files/`를 설치하지 않음). 다른 파일을 쓰려면 `follower_urdf:=...`로 지정하세요.

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

## 확인/결정이 필요한 항목 (TODO)

- [x] 개조형 URDF 수령 (`urdf/so102.urdf`, wrist_yaw 추가된 팔 6관절 + 그리퍼)
- [ ] **SO-102 잔여 확인:** wrist_yaw 서보 ID, lerobot 모터 이름, URDF 임시 가정(yaw→roll 오프셋 부호·yaw 회전 방향·yaw 리밋)
- [ ] **SO-101 URDF의 joint/frame 이름:** 현재 config 값은 추정값이므로 실제 파일과 대조
- [ ] **리더 캘리브레이션:** URDF 영점 자세에서의 tick 값으로 `offsets`, 회전 방향은 `signs`에 기록
- [ ] **opti3 명령 인터페이스:** `ros2 control list_controllers`로 컨트롤러 이름을 확인한 뒤 `command_topic`/`output_type` 설정. 대안으로 indy_moveit `servo_mode`(MoveIt Servo)에 포즈 명령을 보내는 방식도 있음
- [ ] **토치 팁 오프셋:** `tool_xyz/rpy` 실측 (opti3 `tcp`는 플랜지에 오프셋 0)
- [ ] **베이스 정렬:** `leader_to_follower_rpy`, `position_scale`, `workspace_min/max`
- [ ] **그리퍼(aux) 축 용도:** `ee_mapping_node._on_aux` (예: 아크 ON/OFF 트리거)
- [ ] **클러치 입력 장치:** 풋페달·버튼 등 → `/teleop/clutch`

## 모방학습 데이터 기록 권장 토픽

리더의 축 수와 무관한 형식으로 남기려면 다음 토픽을 기록하세요.

```bash
ros2 bag record /joint_states /follower/ee_pose /follower/ee_target /follower/aux_command /teleop/clutch
```

`/leader/joint_states`는 디버깅용으로만 쓰고, 학습 입력으로는 쓰지 않는 것을 권장합니다.
