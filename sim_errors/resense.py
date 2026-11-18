#!/usr/bin/env python3
"""기록된 에피소드의 단면 센서 열(laser)만 지금 센서 모델로 다시 계산한다. Isaac Sim을 다시 돌리지 않는다.

단면 측정은 시나리오, 토치 진행도 s, 아크 진행도 s_arc만으로 정해지고 로봇 추종과는 무관하다. 그래서 센서 모델이
바뀌면(예: 평면 맞춤으로 작업각·진행각 추가) 이미 만든 기록에서 laser, laser_valid만 바꿔 새 폴더에 쓴다.
나머지 필드(정답, 목표, 실측 관절, 토치 포즈)는 그대로 복사한다.

사용법 (저장소 루트에서)
  python -m sim_errors.resense episodes/t_small --out episodes/t_small_prof
  python -m sim_errors.resense episodes/t_small --out episodes/t_small_prof --config sim_errors/config/part_errors.yaml

- 센서 설정은 기본으로 지금 config의 sensors 섹션을 쓴다(시나리오에 저장된 sensors를 덮어씀). 시나리오 것을 그대로
  쓰려면 --keep-sensors (그 시나리오에 평면 맞춤 설정이 없으면 기본값).
- 노이즈 난수는 파일마다 고정 시드(--seed + 파일 순서)라 다시 돌려도 같은 결과.
- batch_report.json은 복사하지 않는다. 새 폴더에서 run_batch.py --report-only로 다시 만든다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

from .make_scenarios import DEFAULT_CFG
from .part_errors import PartErrorModel, sensors_from_cfg


def resense_file(src: str, dst: str, sensors: dict | None, seed: int) -> dict:
    d = np.load(src, allow_pickle=False)
    data = {k: d[k] for k in d.files}
    scn = json.loads(str(data['scenario_json']))
    if sensors is not None:
        scn['sensors'] = dict(sensors)
    else:
        scn['sensors'] = {**sensors_from_cfg({'laser_lookahead_mm': scn['sensors']['laser_lookahead'] * 1e3,
                                              'laser_noise_mm': scn['sensors']['laser_noise'] * 1e3,
                                              'laser_dropout': scn['sensors']['laser_dropout']}),
                          **scn['sensors']}
    m = PartErrorModel(scn)
    rng = np.random.default_rng(seed)
    weld = data['arc_on'].astype(bool) & (data['phase'].astype(str) == 'weld')
    las = np.full((len(weld), 5), np.nan)
    for i in np.flatnonzero(weld):
        las[i] = m.laser(float(data['s'][i]), float(data['s_arc'][i]), rng)
    data['laser'] = las
    data['laser_valid'] = np.isfinite(las[:, 0])
    data['scenario_json'] = np.array(json.dumps(scn))
    if 'meta_json' in data:
        meta = json.loads(str(data['meta_json']))
        meta['resensed_from'] = os.path.basename(src)
        data['meta_json'] = np.array(json.dumps(meta, ensure_ascii=False))
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    np.savez_compressed(dst, **data)
    return {'file': os.path.basename(src), 'rows': int(weld.sum()), 'valid': float(np.mean(np.isfinite(las[weld, 0])))}


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episodes', help='에피소드 .npz 폴더')
    ap.add_argument('--out', required=True, help='새 폴더 (원본은 그대로 둠)')
    ap.add_argument('--config', default=DEFAULT_CFG)
    ap.add_argument('--keep-sensors', action='store_true', help='시나리오에 저장된 센서 설정 사용')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(args.episodes):
        sys.exit('--out은 원본과 다른 폴더여야 합니다')
    sensors = None
    if not args.keep_sensors:
        with open(args.config, encoding='utf-8') as f:
            sensors = sensors_from_cfg(yaml.safe_load(f)['sensors'])
    files = sorted(f for f in glob.glob(os.path.join(args.episodes, '*.npz')) if not f.endswith('.failed.npz'))
    for k, src in enumerate(files):
        r = resense_file(src, os.path.join(args.out, os.path.basename(src)), sensors, args.seed * 100000 + k)
        print(f'[{k + 1}/{len(files)}] {r["file"]}  용접 {r["rows"]} 주기, 유효 측정 {r["valid"] * 100:.1f}%', flush=True)
    print(f'완료 → {os.path.abspath(args.out)}  (점검: run_batch.py <시나리오 폴더> --out {args.out} --report-only)')


if __name__ == '__main__':
    main()
