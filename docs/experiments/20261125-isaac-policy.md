# 공용 MLP가 Isaac opti3를 직접 모는 닫힌 루프 (진행 중)

실험 ID `20261125-isaac-policy` · [실험 기록 목록](README.md) · 실행기 사용법은 [sim_bridge/README.md](../../sim_bridge/README.md) "학습 정책으로 몰기"

## 왜

지금까지 정책 평가([20261123-kin-4fam](20261123-kin-4fam.md))는 운동학 닫힌 루프였다. 로봇이 명령을 바로 따라간다고 가정해서
Isaac 관절 추종 오차가 정책 관측과 결과에 들어가지 않았다. 이번에는 네 군 공용 MLP(`mlp_all`)가 규칙 전문가 자리에서 매 주기
보정값을 내고, 그 값으로 Isaac의 opti3를 직접 몬다. 함께 미뤄 둔 화면 확인(필릿 모재 20 cm 옮긴 배치, 파이프 모양, 팔·토치가
모재를 뚫는지)도 이 실행기 화면으로 본다. 이 실행기가 다음 단계인 공유 제어(정책 구동 중 리더 암 보정)와 개입 데이터 수집의 바탕이다.

## 방법

- 실행기: `sim_bridge/policy_to_sim.py`. `expert_to_sim.run_episode`에서 토치 목표를 내는 부분만 `learning/policy_driver.py`로 바꿨다.
  모재 배치, 이음매 IK 도달 확인, 시작 자세 바로 놓기, lockstep, 기록은 전문가 Isaac 실행과 같다.
- 한 주기 흐름은 운동학 평가(`learning.evaluate.closed_loop`)와 같다: 단면 측정을 버퍼에 쌓고, 10 Hz로 정책 행동(보정 4 + 속도·위빙)을 받아
  추정 경로 + 보정 + 위빙으로 토치 목표를 만들고, 정책 속도로 진행한다. 정책 관측의 토치 자세는 Isaac 관절 실측의 FK다
  (기본 입력 `section,profile,progress,mode`에는 토치 자세가 없어 지금 정책에는 영향이 없다). 토치 롤은 전문가처럼 홈 자세 툴 x축에 맞춘다.
- 시나리오: 운동학 실험의 시험 나눔 240개(T 59, 코너 60, V홈 60, 파이프 61, 가혹 30씩 포함). 운동학 평가와 같은 시나리오라
  에피소드마다 바로 비교된다. 센서 노이즈 시드만 다르다(이름마다 고정).
  Isaac 검증용 1/5 묶음(`20261124-isaac-4fam/batch`)은 같은 시드로 뽑은 앞쪽 시나리오라 학습 나눔과 겹칠 수 있어 쓰지 않는다.
- 오차: **측정** = 실측 토치(FK) − 실제 이음매(좌우는 명령한 위빙을 뺌), **명령** = 정책 보정 − 실제 이음매(운동학 평가와 같은 정의).
  완벽 추종이면 명령 오차가 운동학 닫힌 루프와 똑같이 나오는 것을 테스트로 확인했다(`learning/test/test_policy_driver.py`).
  추종 95%(실측 팁과 목표 사이 거리), 뒤짐(이음매 방향으로 실측이 명령보다 늦은 평균)도 본다.

## PC 실행 명령 (Windows cmd, 저장소 폴더에서)

```bat
cd C:\workspace\Welding-Robot
git checkout main
git pull
conda activate weld
set URDF=C:\workspace\indy-ros2-jazzy\indy_description\urdf_files\opti3.urdf
set K=outputs\experiments\20261123-kin-4fam
set E=20261125-isaac-policy
```

처음 실행을 옛 ID(`20251125-isaac-policy`, 2026-11-25에 연도를 고치기 전)로 했으면 실행을 멈추고 폴더 이름을 바꾼 뒤 새 ID로 이어서 돌린다.
바꾸기 전에 새 ID로 돌리면 빈 폴더에서 처음부터 시작한다.

```bat
ren outputs\experiments\20251125-isaac-policy 20261125-isaac-policy
```

운동학 에피소드(`%K%\episodes`)를 지웠으면 시나리오만 다시 만들어 `--scenarios %K%\batch`로 쓴다 (1분 미만, 시드 고정이라
0bc9d6b 이후 시나리오 뽑는 코드가 안 바뀌어 같은 시나리오·같은 나눔이 나온다):

```bat
for %f in (fillet_t fillet_corner butt_v pipe_girth) do python -m sim_errors.make_scenarios --batch --family %f --out %K%\batch
```

**1. 화면 확인 (군마다 하나, 약 5분).** receiver를 화면 있게 띄운다:

```bat
C:\isaacsim\python.bat sim_bridge\isaac_opti3_receiver.py --lockstep
```

터미널 2:

```bat
python sim_bridge\policy_to_sim.py --policy %K%\runs\mlp_all\policy.pt --scenarios %K%\episodes --per-family 1 --exp %E% --run look --follower-urdf %URDF%
```

볼 것: 필릿(T·코너) 모재가 토치 아래 20 cm 옮긴 자리에 놓이는지, 파이프가 관 모양으로 보이고 토치가 호를 따라 ±50° 기우는지,
팔·토치가 모재를 뚫고 들어가는 순간이 있는지(모재는 충돌이 없어서 눈으로만 보인다), 토치 끝이 빨간 선(실제 이음매)을 따라가는지.

**2. 일괄 (시험 240개, 약 1시간 30분).** receiver를 닫고 `--headless --lockstep`으로 다시 띄운다:

```bat
C:\isaacsim\python.bat sim_bridge\isaac_opti3_receiver.py --lockstep --headless
python sim_bridge\policy_to_sim.py --policy %K%\runs\mlp_all\policy.pt --scenarios %K%\episodes --kin-eval %K%\runs\mlp_all\eval.json --exp %E% --follower-urdf %URDF%
```

멈추면 같은 명령으로 이어서 한다. 끝에 시편군별 표(측정·명령·운동학)가 나오고 `outputs\experiments\20261125-isaac-policy\runs\mlp_all\isaac\eval_isaac.json`에 저장된다.
먼저 짧게 보려면 `--per-family 5`를 붙인다(20개, 약 10분). 같은 폴더에 이어서 쌓이므로 그다음 전체 명령을 그대로 돌리면 된다.

**3. 팔·토치–모재 간격 점검 (Isaac 불필요, 에피소드당 몇 초).** 모재는 충돌이 없어 뚫어도 시뮬레이션이 막지 않으므로 저장된 관절 실측으로 잰다.
팔·토치 모양은 Isaac 장면의 표시 메시 그대로다 (`sim_bridge/clearance_check.py`).

```bat
python sim_bridge\clearance_check.py outputs\experiments\%E%\runs\look\isaac outputs\experiments\%E%\runs\mlp_all\isaac --follower-urdf %URDF%
```

**4. (선택) 규칙 기준선을 Isaac에서:** `--policy rule --kin-eval %K%\runs\rule\eval.json`으로 같은 명령. 정책과 규칙의 차이가 Isaac에서도 그대로인지 본다.

보내 줄 것: 2단계 끝의 표(또는 `eval_isaac.json`, 작은 파일), 3단계 끝의 간격 표.

## 판정

- 완료 100%, 추종 95%가 전문가 Isaac 실행(0.33–0.55 mm)과 비슷하면 정책 구동이 추종에 문제를 만들지 않는다.
- 측정 오차가 운동학 오차(T 0.16 / 0.15, 코너 0.16 / 0.16, V홈 0.23 / 0.14, 파이프 0.20 / 0.15 mm 좌우 / 높이)에서 추종 오차만큼만 늘면
  운동학 평가가 Isaac 성능을 잘 대신한다는 뜻이다. 많이 늘거나 명령 오차 자체가 달라지면(관측이 실측에 따라 바뀌는 경우) 원인을 본다.

## 결과

### 화면 확인 실행 (2026-11-25, 사용자 PC, 군마다 깨끗함 1개)

`--per-family 1 --run look`, 화면 있는 receiver. 네 개 모두 끝까지 갔고 IK 경고 0. 깨끗함이라 보정 0 오차가 0이어서 개선율은 없다.

| 에피소드 | 모드 | 측정 좌우 / 높이 mm | 명령 좌우 / 높이 mm | 추종 95% mm | 뒤짐 mm |
| --- | --- | --- | --- | --- | --- |
| T `clean_0010` | 직선 | 0.10 / 0.07 | 0.02 / 0.02 | 0.31 | 0.26 |
| 코너 `corner_clean_0017` | 위빙 | 0.21 / 0.05 | 0.03 / 0.04 | 0.43 | 0.24 |
| V홈 `butt_clean_0000` | 직선 | 0.04 / 0.15 | 0.02 / 0.04 | 0.32 | 0.24 |
| 파이프 `pipe_clean_0002` | 직선 | 0.04 / 0.14 | 0.02 / 0.04 | 0.42 | 0.27 |

- 추종 95%(0.31–0.43 mm)는 전문가 Isaac 실행(0.33–0.55 mm)과 같은 수준이다. 정책 구동이 추종을 나쁘게 하지 않았다.
- 측정 오차는 명령 오차에 추종 오차(0.1 mm 안팎)가 더해진 정도다. 코너 좌우 0.21 mm는 위빙 모드라서다: 실측이 위빙을 조금 늦게 따라가
  명령 위빙을 빼도 위빙 주파수 흔들림이 남는다. 그래서 위빙 한 주기 평균한 **위빙 중심 좌우**를 더했다
  (`--report-only`로 저장된 에피소드에서 다시 계산된다).

### 화면 항목을 숫자로 판정

화면으로는 판단이 어려워 숫자로 확인했다.

1. **토치 끝이 빨간 선(실제 이음매)을 따라가는지: 따라간다.** Isaac 장면(`posco_torch.usd`)의 링크·tcp·`weld_tip` 좌표계를 `opti3.urdf`와 비교하니
   관절 0에서 링크 6개 원점·방향이 같고 팁 위치도 같다(베이스 기준 (0.186, 0.000, 1.2705) m). 추종 오차는 Isaac 관절 실측의 URDF FK로 쟀으므로
   장면에 보이는 토치 끝도 같은 오차(95% 0.31–0.43 mm)로 이음매를 따라간다.
2. **팔·토치가 모재를 뚫는지: 뚫지 않는다.** 새 간격 점검(`sim_bridge/clearance_check.py`)으로 시험 240개를 실제 `opti3.urdf`로 돌린 관절 궤적에서
   장면 토치·링크 표시 메시와 모재 판 사이 최소 거리를 쟀다. 클라우드에서 가짜 receiver(1차 지연)와 규칙 기준선으로 돌렸다. Isaac 궤적과는
   추종 오차(1 mm 미만)와 정책 보정 차이(0.2 mm 안팎)만 달라 간격 결론은 같다. PC의 Isaac 기록으로도 같은 명령(3단계)으로 확인할 수 있다.

   | 시편군 | 개수 | 토치 앞 (팁 15–100 mm) | 토치 뒤 | 손목 (link5·6) | 팔 (link1–4) | 와이어 끝 (팁 15 mm 안) |
   | --- | --- | --- | --- | --- | --- | --- |
   | T 필릿 | 59 | 8.7 | 59.9 | 268 | 324 | −1.6 |
   | 코너 필릿 | 60 | 8.5 | 58.5 | 205 | 251 | −1.5 |
   | 맞대기 V홈 | 60 | 6.7 | 87.3 | 319 | 201 | −3.4 |
   | 파이프 원주 | 61 | 6.4 | 87.2 | 322 | 148 | −3.6 |

   (모든 에피소드 중 최소 간격 mm, 음수 = 판 안. 도달 실패·중단 0.)
   토치 금속 부분과 팔은 모두 6 mm 넘게 떨어져 있다. 와이어 끝만 판 면 안쪽으로 몇 mm 나오는데, 토치가 아니라 아크가 나는 와이어 끝이고 두 가지 때문이다.
   - 위빙 상한(`weave_amp_max_mm` 3 mm)이 standoff(2 mm)보다 커서, 큰 틈새의 위빙 끝에서 와이어 끝이 벽면 선을 넘는다 (음수의 대부분이 위빙 에피소드).
   - 점검의 모재는 이음매 벗어남만큼 옮긴 명목 단면이고 판 기울어짐(각변형)은 넣지 않아, 이음매 근처는 mm 단위 오차가 있다.
   장면의 토치는 와이어가 노즐 밖으로 약 55 mm 나와 있어(CAD 기준) 노즐이 모재에서 멀다. 실물 돌출(보통 15–20 mm)로 바꾸면 필릿에서 노즐 간격은
   몇 mm로 줄어든다.
3. **필릿 배치와 파이프 모양:** receiver는 간격 점검과 같은 모재 메시 코드(`nominal_meshes`)로 그린다. 같은 네 에피소드를 그린 참고 그림
   (프로젝트 파일 `experiments/isaac-policy/장면_참고_그림.png`)과 화면이 같으면 맞다. 베이스 기준으로 필릿 이음매는 x 0.655 m, 높이 0.126 m에서
   Y 방향(+0.15 → −0.15 m)으로 놓인다(V홈보다 X로 0.2 m 멀고 0.05 m 낮음). V홈은 x 0.455 m, 높이 0.176 m. 파이프는 반지름 0.15 m 관 두 토막이
   X축 방향(로봇에서 멀어지는 쪽)으로 이어져 있고, 빨간 이음매 호가 x 0.455 m에서 관 꼭대기(높이 0.176 m)를 넘어간다.

### 일괄 (시험 240개)

사용자 PC 실행 대기.
