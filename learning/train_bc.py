#!/usr/bin/env python3
"""기준 보정 정책 학습: 작은 MLP 또는 GRU 행동 복제 (정답 = 시뮬 규칙 전문가).

사용법 (저장소 루트에서, pip install torch)
  python -m learning.train_bc datasets/t_small --model mlp --out runs/mlp            # 기본 입력 section,profile,progress,mode
  python -m learning.train_bc datasets/t_small --model mlp --obs section,progress,mode --out runs/mlp_noprofile
  python -m learning.train_bc datasets/t_small --model gru --out runs/gru

- 입력 묶음(--obs): section(단면 거리 버퍼 좌우·높이·틈새 + 유효 36), profile(같은 버퍼의 작업각·진행각 18),
  torch(토치 자세 어긋남 4), progress(s), mode(직선/위빙 2).
  torch는 전문가가 직전에 보낸 보정을 거의 그대로 담고 있어서, 정책이 센서 대신 자기 직전 출력을 따라 하는
  지름길을 배운다(첫 한 바퀴에서 닫힌 루프 높이 오차가 나빠짐). 그래서 기본 입력에서 뺐다.
- 손실: 행동 7개를 학습 세트 평균·표준편차로 정규화한 MSE.
- MLP: 프레임 하나 → 행동. GRU: 길이 --seq (기본 20 = 2초) 구간을 펼쳐 학습하고, 실행 때는 은닉 상태를 이어 간다.
- 검증 손실이 가장 낮은 가중치를 <out>/policy.pt로 저장 (정규화 값, 입력 묶음, 모델 설정 포함).
- 수량·섭동 비교: --max-train N (학습 에피소드 N개만, 강도별 비율 유지, 같은 시드면 작은 N이 큰 N의 부분집합),
  --no-perturb (섭동 에피소드를 학습에서 뺌). 고른 에피소드는 train_log.json의 train_files에 남는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

from .features import ACTION_NAMES, GROUPS

DEFAULT_GROUPS = ('section', 'profile', 'progress', 'mode')


def group_names(groups) -> list[str]:
    return [n for g in groups for n in GROUPS[g]]


def columns(all_names, names) -> np.ndarray:
    """관측 열 이름 → all_names 안의 번호. 없는 이름이 있으면 오류 (예: 프로파일 없는 이전 데이터)."""
    all_names = [str(n) for n in all_names]
    missing = [n for n in names if n not in all_names]
    if missing:
        raise SystemExit(f'데이터에 없는 관측 {len(missing)}개 (예: {missing[0]}). 변환을 다시 하거나 --obs를 줄이세요.')
    return np.array([all_names.index(n) for n in names])


class MLPPolicy(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        mods, d = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*mods)

    def forward(self, x, h=None):          # x (B, T, n_in) 또는 (B, n_in)
        return self.net(x), None


class GRUPolicy(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 128, layers: int = 1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, h=None):          # x (B, T, n_in)
        y, h = self.gru(self.inp(x), h)
        return self.head(y), h


def build(cfg: dict) -> nn.Module:
    cls = MLPPolicy if cfg['model'] == 'mlp' else GRUPolicy
    return cls(cfg['n_in'], cfg['n_out'], cfg['hidden'], cfg['layers'])


class Policy:
    """저장한 정책을 불러와 한 프레임씩 실행 (평가 롤아웃, 나중에 Isaac 실행에서 사용)."""

    def __init__(self, path: str):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.cfg = ck['cfg']
        self.net = build(self.cfg)
        self.net.load_state_dict(ck['state'])
        self.net.eval()
        if 'obs_names' not in self.cfg:
            raise SystemExit(f'{path}: 관측 이름이 없는 이전 형식 정책입니다. 다시 학습하세요.')
        from .features import OBS_NAMES
        self.idx = columns(OBS_NAMES, self.cfg['obs_names'])
        self.mu_o, self.sd_o = np.array(ck['mu_o']), np.array(ck['sd_o'])
        self.mu_a, self.sd_a = np.array(ck['mu_a']), np.array(ck['sd_a'])
        self.h = None

    def reset(self):
        self.h = None

    @torch.no_grad()
    def __call__(self, obs_full: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs_full)[self.idx] - self.mu_o) / self.sd_o
        y, self.h = self.net(torch.as_tensor(x, dtype=torch.float32)[None, None], self.h)
        return y[0, -1].numpy() * self.sd_a + self.mu_a

    @torch.no_grad()
    def predict_episode(self, obs: np.ndarray) -> np.ndarray:
        """한 에피소드 관측 (T, 61, OBS_NAMES 순서) → 행동 (T, 7). 열린 루프 평가용."""
        x = (obs[:, self.idx] - self.mu_o) / self.sd_o
        y, _ = self.net(torch.as_tensor(x, dtype=torch.float32)[None])
        return y[0].numpy() * self.sd_a + self.mu_a


def load_frames(path: str) -> dict:
    d = np.load(os.path.join(path, 'frames.npz'), allow_pickle=False)
    return {k: d[k] for k in d.files}


def episodes_of(fr: dict, split: str) -> list[int]:
    return [i for i, s in enumerate(fr['ep_split']) if s == split]


def pick_train(fr: dict, eps: list[int], max_train: int | None = None, no_perturb: bool = False,
               seed: int = 0) -> list[int]:
    """학습 에피소드 고르기. no_perturb면 섭동 에피소드를 빼고, max_train이면 강도별 비율을 유지해 N개만.

    강도마다 섞은 뒤 (강도 안 순번 + 0.5) / 강도 개수 순으로 늘어놓고 앞에서 자르므로, 같은 시드에서는
    작은 N의 선택이 큰 N의 선택에 포함된다 (수량 곡선용)."""
    if no_perturb:
        if 'ep_perturb' not in fr:
            raise SystemExit('frames.npz에 섭동 표시가 없음: to_lerobot으로 다시 변환하세요.')
        eps = [e for e in eps if not bool(fr['ep_perturb'][e])]
    if max_train is None or max_train >= len(eps):
        return list(eps)
    rng = np.random.default_rng(seed)
    kinds: dict[str, list[int]] = {}
    for e in eps:
        kinds.setdefault(str(fr['ep_kind'][e]), []).append(e)
    order = []
    for g in kinds.values():
        g = list(rng.permutation(g))
        order += [((k + 0.5) / len(g), rng.random(), int(e)) for k, e in enumerate(g)]
    return sorted(e for _, _, e in sorted(order)[:max_train])


def windows(fr: dict, eps: list[int], seq: int) -> tuple[np.ndarray, np.ndarray]:
    """에피소드를 seq 길이 구간으로 자름 (겹침 절반). 반환 (N, seq, n_obs), (N, seq, n_act)."""
    X, Y = [], []
    for e in eps:
        m = fr['episode_index'] == e
        o, a = fr['obs'][m], fr['action'][m]
        for s0 in range(0, max(1, len(o) - seq + 1), max(1, seq // 2)):
            if s0 + seq <= len(o):
                X.append(o[s0:s0 + seq])
                Y.append(a[s0:s0 + seq])
    return np.array(X), np.array(Y)


def train(data: str, out: str, model: str = 'mlp', groups=DEFAULT_GROUPS,
          epochs: int = 40, seq: int = 20, hidden: int | None = None, layers: int | None = None, lr: float = 1e-3,
          batch: int = 256, seed: int = 0, max_train: int | None = None, no_perturb: bool = False,
          verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    fr = load_frames(data)
    names = group_names(groups)
    idx = columns(fr['obs_names'], names)
    tr, va = pick_train(fr, episodes_of(fr, 'train'), max_train, no_perturb, seed), episodes_of(fr, 'val')
    if not tr:
        raise SystemExit('학습 에피소드가 없음')
    va = va or tr
    seq_len = 1 if model == 'mlp' else seq
    Xtr, Ytr = windows(fr, tr, seq_len)
    Xva, Yva = windows(fr, va, seq_len)
    Xtr, Xva = Xtr[..., idx], Xva[..., idx]
    flat_o, flat_a = Xtr.reshape(-1, Xtr.shape[-1]), Ytr.reshape(-1, Ytr.shape[-1])
    mu_o, sd_o = flat_o.mean(0), flat_o.std(0) + 1e-3
    mu_a, sd_a = flat_a.mean(0), flat_a.std(0) + 1e-3
    norm = lambda X, mu, sd: torch.as_tensor((X - mu) / sd, dtype=torch.float32)  # noqa: E731
    xt, yt = norm(Xtr, mu_o, sd_o), norm(Ytr, mu_a, sd_a)
    xv, yv = norm(Xva, mu_o, sd_o), norm(Yva, mu_a, sd_a)
    cfg = {'model': model, 'groups': list(groups), 'obs_names': names, 'n_in': len(idx),
           'n_out': len(ACTION_NAMES), 'hidden': hidden or (256 if model == 'mlp' else 128),
           'layers': layers or (2 if model == 'mlp' else 1), 'seq': seq_len, 'fps': int(fr['fps']),
           'prescan': bool(fr['prescan']) if 'prescan' in fr else False}
    net = build(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, hist = float('inf'), []
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        tl = 0.0
        for k in range(0, len(xt), batch):
            b = perm[k:k + batch]
            y, _ = net(xt[b])
            loss = ((y - yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            y, _ = net(xv)
            per = ((y - yv) ** 2).mean(dim=(0, 1)).numpy()
        vl = float(per.mean())
        hist.append({'epoch': ep + 1, 'train': tl / len(xt), 'val': vl, 'val_per_action': per.tolist()})
        if vl < best:
            best = vl
            torch.save({'cfg': cfg, 'state': net.state_dict(), 'mu_o': mu_o, 'sd_o': sd_o, 'mu_a': mu_a, 'sd_a': sd_a,
                        'epoch': ep + 1, 'val': vl}, os.path.join(out, 'policy.pt'))
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f'epoch {ep + 1:3d}  학습 {tl / len(xt):.4f}  검증 {vl:.4f}  '
                  + ' '.join(f'{n} {v:.3f}' for n, v in zip(ACTION_NAMES, per)))
    res = {'cfg': cfg, 'best_val': best, 'history': hist, 'train_episodes': len(tr), 'val_episodes': len(va),
           'train_samples': int(len(xt)), 'seconds': round(time.time() - t0, 1),
           'train_files': [str(fr['ep_file'][e]) for e in tr], 'max_train': max_train, 'no_perturb': no_perturb}
    with open(os.path.join(out, 'train_log.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='to_lerobot.py 출력 폴더 (frames.npz)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', choices=('mlp', 'gru'), default='mlp')
    ap.add_argument('--obs', default=','.join(DEFAULT_GROUPS), help=f'입력 묶음 ({", ".join(GROUPS)})')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seq', type=int, default=20, help='GRU 학습 구간 길이 (프레임)')
    ap.add_argument('--hidden', type=int)
    ap.add_argument('--layers', type=int)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-train', type=int, help='학습 에피소드 N개만 (강도별 비율 유지)')
    ap.add_argument('--no-perturb', action='store_true', help='섭동 에피소드를 학습에서 뺌')
    args = ap.parse_args(argv)
    r = train(args.data, args.out, args.model, args.obs.split(','), args.epochs, args.seq, args.hidden, args.layers,
              args.lr, seed=args.seed, max_train=args.max_train, no_perturb=args.no_perturb)
    print(f'최저 검증 손실 {r["best_val"]:.4f} (정규화 MSE), 학습 에피소드 {r["train_episodes"]}개, '
          f'{r["seconds"]} s → {os.path.join(args.out, "policy.pt")}')


if __name__ == '__main__':
    main()
