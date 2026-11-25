"""산출물 경로 규약 (docs/outputs.md).

모든 실행 산출물은 한 루트(기본 <저장소>/outputs, 환경변수 WELD_OUTPUTS로 바꿈) 아래에 둔다.

  outputs/
    experiments/<YYYYMMDD>-<이름>/   시나리오 묶음 하나 = 실험 하나
      manifest.json                  단계마다 커밋·명령·시각을 덧붙임 (write_manifest)
      batch/ reach/ episodes/ episodes_isaac/<시편군>/ data/ runs/<정책>/ report.md
    ci/<GitHub 실행 ID>/<정책>/      내려받은 CI 평가
    scratch/                         단독 시나리오·디버그용

표준 라이브러리만 쓰므로 Isaac Sim 내장 파이썬에서도 import할 수 있다.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_VAR = 'WELD_OUTPUTS'

# 실험 폴더 안의 하위 폴더 이름 (learning/run_experiment.py가 쓰는 exp/ 배치와 같음)
LAYOUT = {
    'batch': 'batch',                    # 시나리오 json (sim_errors.make_scenarios)
    'reach': 'reach',                    # reach_report.json (run_batch --reach-only)
    'episodes': 'episodes',              # 운동학 에피소드 + batch_report.json (run_batch --kinematic)
    'episodes_isaac': 'episodes_isaac',  # Isaac 에피소드, 시편군별 하위 폴더
    'data': 'data',                      # frames.npz, splits.json, lerobot/ (learning.to_lerobot)
    'runs': 'runs',                      # <정책>/policy.pt, train_log.json, eval.json
}

_EXP_ID = re.compile(r'^\d{8}-[a-z0-9][a-z0-9_.-]*$')


def outputs_root() -> Path:
    """산출물 루트. WELD_OUTPUTS가 있으면 그 경로(상대경로면 현재 폴더 기준), 없으면 <저장소>/outputs."""
    env = os.environ.get(ENV_VAR)
    return Path(env).expanduser().resolve() if env else REPO_ROOT / 'outputs'


def check_exp_id(exp_id: str) -> str:
    """실험 ID 형식 확인: YYYYMMDD-<이름> (이름은 소문자·숫자·_·-·.). 맞으면 그대로 돌려준다."""
    if not _EXP_ID.match(exp_id):
        raise ValueError(f'실험 ID는 YYYYMMDD-<이름> 형식이어야 합니다 (예: 20261201-weave): {exp_id!r}')
    _dt.datetime.strptime(exp_id[:8], '%Y%m%d')
    return exp_id


def new_exp_id(name: str, date: _dt.date | None = None) -> str:
    """오늘 날짜(또는 date)로 실험 ID를 만든다."""
    d = date or _dt.date.today()
    return check_exp_id(f'{d:%Y%m%d}-{name}')


def exp_dir(exp_id: str, part: str | None = None, *, create: bool = False) -> Path:
    """outputs/experiments/<exp_id>[/<part>]. part는 LAYOUT의 키."""
    p = outputs_root() / 'experiments' / check_exp_id(exp_id)
    if part is not None:
        if part not in LAYOUT:
            raise KeyError(f'알 수 없는 하위 폴더 {part!r}, 가능한 값: {", ".join(LAYOUT)}')
        p = p / LAYOUT[part]
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def ci_dir(run_id: str | int, *, create: bool = False) -> Path:
    """outputs/ci/<GitHub 실행 ID>."""
    s = str(run_id)
    if not s.isdigit():
        raise ValueError(f'GitHub 실행 ID는 숫자여야 합니다: {run_id!r}')
    p = outputs_root() / 'ci' / s
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def scratch_dir(*parts: str, create: bool = False) -> Path:
    """outputs/scratch[/parts...]. 지워도 되는 단독 실행·디버그 산출물."""
    p = outputs_root().joinpath('scratch', *parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_out(out: str | os.PathLike | None, exp_id: str | None, part: str) -> Path:
    """CLI 출력 경로 결정: --out을 주면 그대로(지금 동작 유지), 아니면 --exp의 part 폴더."""
    if out:
        return Path(out)
    if exp_id:
        return exp_dir(exp_id, part)
    raise ValueError('--out 또는 --exp 중 하나는 있어야 합니다')


def add_exp_arg(ap: argparse.ArgumentParser) -> None:
    """CLI에 --exp 옵션을 더한다. 주면 입력·출력 경로를 outputs/experiments/<ID>/ 배치에서 채운다."""
    ap.add_argument('--exp', type=check_exp_id, metavar='YYYYMMDD-이름',
                    help='실험 ID. 입력·출력 기본값을 outputs/experiments/<ID>/에서 채우고 manifest.json에 기록 '
                         '(--out 등을 따로 주면 그 경로가 우선)')


def file_sha256(path: str | os.PathLike) -> str | None:
    """파일 내용 해시 (설정 파일 기록용). 없으면 None."""
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def record_stage(exp_id: str | None, stage: str, **extra) -> Path | None:
    """--exp를 줬을 때만 실험 폴더의 manifest.json에 이번 단계를 기록한다 (단계 시작 시점).

    extra의 경로 값은 문자열로 바꿔 적는다.
    """
    if not exp_id:
        return None
    clean = {k: (str(v) if isinstance(v, os.PathLike) else
                 [str(x) for x in v] if isinstance(v, (list, tuple)) else v)
             for k, v in extra.items() if v is not None}
    return write_manifest(exp_dir(exp_id, create=True), stage, clean)


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_info() -> dict:
    """현재 커밋과 작업 트리 변경 여부 (git이 없으면 None)."""
    status = _git('status', '--porcelain', '--untracked-files=no')
    return {'commit': _git('rev-parse', 'HEAD'), 'dirty': None if status is None else bool(status)}


def write_manifest(directory: str | os.PathLike, stage: str, extra: dict | None = None,
                   argv: list[str] | None = None) -> Path:
    """<directory>/manifest.json의 stages 목록에 이번 단계 기록을 덧붙인다 (없으면 새로 만듦).

    기록: 단계 이름, UTC 시각, git 커밋·dirty, 명령줄, 파이썬·OS, extra(설정 해시·시드·입력 경로 등).
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'manifest.json'
    data = {'stages': []}
    if path.is_file():
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        data.setdefault('stages', [])
    data['stages'].append({
        'stage': stage,
        'time_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
        'git': git_info(),
        'argv': list(sys.argv if argv is None else argv),
        'python': platform.python_version(),
        'platform': platform.platform(),
        **({'extra': extra} if extra else {}),
    })
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path
