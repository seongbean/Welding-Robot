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
장면이 열리면 opti3가 시작 자세(토치가 로봇 앞에서 아래를 향함, `--initial-q`로 변경 가능)로 선다.

터미널 2: 리더 (`conda activate weld` 후, 저장소 루트에서)
```bat
python sim_bridge\leader_to_sim.py --mock --follower-urdf C:\workspace\indy-ros2-jazzy\indy_description\urdf_files\opti3.urdf
```
먼저 `--mock`(사인파 입력)으로 연결을 확인한 뒤, `--mock` 대신 `--port COM3`로 실제 리더를 쓴다.

## 3. 조작

- **Enter** = 클러치 ON/OFF (시작은 OFF). ON 순간의 리더·opti3 자세를 기준으로 리더의 상대 이동만 따라간다.
  OFF 상태에서 리더를 편한 자세로 옮긴 뒤 다시 ON 하면 튀지 않는다.
- 상태 줄: `ok` 정상, `clutch_off`, `waiting`(Isaac Sim 실측 대기), `joint_state_timeout`(Isaac Sim 끊김),
  `tracking_error`(명령과 실측 차이가 큼 → 재동기화), `ik_failed`(도달 불가·특이점 자세: 클러치 OFF 후 리더를 돌려놓고 다시 ON), `leader_lost`(리더 읽기 실패)
- 그리퍼 %는 표시만 한다(용도 미정).

## 확인 안 된 것

- `opti3.urdf` 경로: indy-ros2-jazzy 안의 실제 위치로 바꿔 쓴다(xacro만 있으면 알려 줄 것).
- Isaac Sim 장면의 관절 영점·방향이 `opti3.urdf`와 같은지. 다르면 토치가 리더와 다른 방향으로 움직인다.
- 위치 스케일·베이스 방향은 `mapping.yaml`(`position_scale`, `leader_to_follower_rpy`)에서 조정.
