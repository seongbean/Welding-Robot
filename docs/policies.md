# 고정한 기본 정책

공유 제어와 잔차 모델은 고정한 기본 정책 위에서만 쌓입니다. 개입 데이터와 잔차 모델이 어느 기본 정책 위에서 만들어졌는지 묶기 위해
기본 정책을 이름으로 고정하고, 파일 사본은 GitHub Release, sha256은 [`learning/base_policies.json`](../learning/base_policies.json)에 둡니다.
결정 이유는 [설계 결정 "기본 정책 고정"](design-decisions.md#기본-정책-고정-base-v1).

| 이름 | 정책 | sha256 | 사본 |
| --- | --- | --- | --- |
| base-v1 | `20261123-kin-4fam` 의 `mlp_all` (공용 MLP, 관측 section·profile·progress·mode, [실험 기록](experiments/20261123-kin-4fam.md)) | `39a8e9c6e3add5ffdf9cc7d11bb931feb2c38988130e5eeed5a378f51e814ae7` | Release `base-policy-v1`의 `policy.pt` |

## 쓰는 법

- `sim_bridge/shared_to_sim.py --policy <policy.pt> --base-tag base-v1`: 해시가 다르면 시작하지 않습니다. `--base-tag` 없이도 해시로 이름을 찾아
  에피소드 meta(`policy`, `policy_sha256`, `policy_tag`)와 실험 `manifest.json`에 남깁니다.
- 새 PC에서는 Release에서 `policy.pt`를 받아 `outputs/experiments/20261123-kin-4fam/runs/mlp_all/`(또는 아무 곳)에 두고 `--policy`로 줍니다.

## Release 올리기 (한 번, 저장소 관리자)

1. https://github.com/seongbean/Welding-Robot/releases/new 에서 태그 `base-policy-v1`(main에서 새로 만듦), 제목 `base-v1 기본 정책 (mlp_all)`.
2. 설명에 위 표의 sha256과 출처 실험을 적고, PC의 `C:\workspace\Welding-Robot\outputs\experiments\20261123-kin-4fam\runs\mlp_all\policy.pt`를 첨부합니다
   (같은 폴더의 `train_log.json`, `eval.json`도 함께 두면 재현에 좋습니다).
3. Publish release.

## 새 기본 정책

기본 정책을 바꾸면 `base-v2`처럼 새 이름을 `base_policies.json`과 위 표에 더하고 새 Release를 만듭니다. 잔차 모델은 새 기본 정책 위에서
모은 개입 데이터로 다시 학습합니다.
