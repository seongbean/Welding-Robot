# 실험 기록

실험마다 결과와 결론을 파일 하나에 적습니다. 파일 이름은 실험 ID(`YYYYMMDD-<이름>`)이고, 로컬 산출물 폴더
`outputs/experiments/<ID>/`와 같은 ID를 씁니다 ([산출물 규약](../outputs.md)). 새 실험은 이 표 맨 위에 한 줄을 더합니다.

| 실험 ID | 내용 | 로컬 산출물 | GitHub 실행 |
| --- | --- | --- | --- |
| [20261125-isaac-policy](20261125-isaac-policy.md) | 공용 MLP가 Isaac opti3를 직접 모는 닫힌 루프 (진행 중) | `outputs/experiments/20261125-isaac-policy/` | |
| [20261123-kin-4fam](20261123-kin-4fam.md) | 파이프 원주 추가, 네 군 공용 MLP (운동학, 사용자 PC) | `outputs/experiments/20261123-kin-4fam/` | |
| [20261119-butt-v](20261119-butt-v.md) | 맞대기 V홈 추가, 단차 대조와 보정 | | [44470621849](https://github.com/seongbean/Welding-Robot/actions/runs/44470621849), [44480051785](https://github.com/seongbean/Welding-Robot/actions/runs/44480051785), [44492062655](https://github.com/seongbean/Welding-Robot/actions/runs/44492062655) |
| [20261119-corner](20261119-corner.md) | ㄱ자 코너 필릿 추가, T/코너 교차 학습·수량 대조 | | [44453804249](https://github.com/seongbean/Welding-Robot/actions/runs/44453804249), [44460136780](https://github.com/seongbean/Welding-Robot/actions/runs/44460136780) |
| [20261118-t-start-zone](20261118-t-start-zone.md) | 남은 높이 오차 원인(시작 30 mm), 시작 구간 스캔 | | |
| [20261118-t-quantity-perturb](20261118-t-quantity-perturb.md) | 학습 수량·섭동 비교 (T 330개) | | |
| [20261118-t-isaac](20261118-t-isaac.md) | Isaac 기록으로 같은 비교 | `outputs/experiments/20261118-t-small/` | |
| [20261118-t-profile](20261118-t-profile.md) | 단면 프로파일 입력 추가, 토치 자세 입력 제외 | `outputs/experiments/20261118-t-small/` | |
| [20261117-t-first-loop](20261117-t-first-loop.md) | T 필릿 첫 한 바퀴 (MLP/GRU) | `outputs/experiments/20261118-t-small/` | |

- Isaac 물리 검증(코너·V홈·파이프 1/5 묶음)은 [sim_errors/README.md](../../sim_errors/README.md) "Isaac 물리 검증"에 있고, 로컬 산출물은 `outputs/experiments/20261124-isaac-4fam/`입니다.
- 내려받은 CI 평가 결과는 로컬 `outputs/ci/<실행 ID>/`에 있습니다.
- 2026-11-25 전 기록은 `learning/README.md`에서 옮겨 왔습니다. 그래서 일부 ID는 기록 날짜로 붙였고, 11-17~18 T 실험들은 산출물 폴더 하나(`20261118-t-small`)를 같이 씁니다.
