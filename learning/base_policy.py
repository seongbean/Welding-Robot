"""고정한 기본 정책 확인 (표준 라이브러리만). 목록은 learning/base_policies.json, 설명은 docs/policies.md."""
from __future__ import annotations

import hashlib
import json
import os

REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'base_policies.json')


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: str = REGISTRY) -> dict:
    with open(path, encoding='utf-8') as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith('_')}


def identify(policy_path: str, tag: str | None = None, registry: dict | None = None) -> tuple[str | None, str]:
    """정책 파일 → (고정 이름 또는 None, sha256). tag를 주면 그 이름의 해시와 같아야 한다 (다르면 ValueError).
    규칙 기준선('rule' 등 파일이 아닌 이름)은 (None, '')."""
    if not os.path.isfile(policy_path):
        if tag:
            raise ValueError(f'{policy_path}: 파일이 아니라 기본 정책 {tag}로 확인할 수 없음')
        return None, ''
    reg = load_registry() if registry is None else registry
    digest = sha256(policy_path)
    if tag:
        if tag not in reg:
            raise ValueError(f'기본 정책 목록에 없는 이름: {tag} (있는 것: {", ".join(reg)})')
        if reg[tag]['sha256'] != digest:
            raise ValueError(f'{policy_path}의 sha256 {digest[:12]}…가 {tag}({reg[tag]["sha256"][:12]}…)와 다름')
        return tag, digest
    return next((k for k, v in reg.items() if v['sha256'] == digest), None), digest
