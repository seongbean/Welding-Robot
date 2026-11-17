"""run_batch/expert_to_sim 통신 시험: 가짜 receiver(UDP)와 시험용 6축 팔로 에피소드를 연달아 만든다.

가짜 receiver는 isaac_opti3_receiver.py와 같은 규약(reset_q·reset_id 바로 놓기, seq·t 답장, ack 잠금 단계,
scene.scenario 교체)을 흉내 내고, 관절은 목표를 1차 지연으로 따라간다.
"""
import copy
import json
import os
import socket
import sys
import threading
import time

import numpy as np
import pytest
import yaml

pytest.importorskip('pinocchio')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from expert_to_sim import SimLink, build_stepper, run_episode  # noqa: E402
from sim_errors.check_episode import OK, WARN, check  # noqa: E402
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch  # noqa: E402
from sim_errors.part_errors import save_scenario  # noqa: E402

HOME = np.array([-0.02, -1.557, 2.039, -2.053, -1.571, -1.591])
DT = 1 / 60


class FakeReceiver(threading.Thread):
    def __init__(self, listen, reply, lockstep=True):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(listen)
        self.sock.settimeout(0.001)
        self.reply = reply
        self.q = HOME.copy()
        self.target = HOME.copy()
        self.seq, self.last_ack, self.locked, self.reset_id = 0, None, False, None
        self.lockstep = lockstep
        self.scenarios, self.resets, self.stop = [], 0, False
        self.pause_at = None          # 이 seq에서 0.3초 멈춤 (Isaac 끊김 흉내)

    def handle(self, data):
        msg = json.loads(data.decode())
        sc = (msg.get('scene') or {}).get('scenario')
        if sc and (not self.scenarios or self.scenarios[-1] != sc):
            self.scenarios.append(sc)
        if msg.get('reset_q') is not None:
            if msg.get('reset_id') != self.reset_id:
                self.q = self.target = np.array(msg['reset_q'])
                self.reset_id, self.last_ack, self.locked = msg['reset_id'], None, False
                self.resets += 1
        elif msg.get('reset_id', self.reset_id) == self.reset_id:
            self.target = np.array(msg['q'])
            if msg.get('ack') is not None:
                self.last_ack, self.locked = int(msg['ack']), self.lockstep

    def drain(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (socket.timeout, BlockingIOError, OSError):
                return
            self.handle(data)

    def run(self):
        while not self.stop:
            self.drain()
            if self.stop:
                return
            if self.locked:
                t_end = time.monotonic() + 0.5
                while (self.last_ack is None or self.last_ack < self.seq) and time.monotonic() < t_end:
                    self.drain()
                if self.last_ack is None or self.last_ack < self.seq:
                    self.locked = False
            if self.pause_at is not None and self.seq == self.pause_at:
                time.sleep(0.3)
                self.pause_at = None
            self.q = self.q + 0.6 * (self.target - self.q)
            self.seq += 1
            self.sock.sendto(json.dumps({'q': self.q.tolist(), 'seq': self.seq, 't': self.seq * DT,
                                         'reset_id': self.reset_id}).encode(), self.reply)
            if not self.locked:
                time.sleep(0.0005)


@pytest.fixture
def setup(tmp_path):
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    paths = []
    for name, scn in make_batch(cfg, {'clean': 1, 'single': 0, 'mixed': 2, 'hard': 0}, seed=2):
        scn = copy.deepcopy(scn)
        scn['geometry']['length'] = 0.05
        p = tmp_path / name
        save_scenario(scn, str(p))
        paths.append(str(p))
    rx = FakeReceiver(('127.0.0.1', 19870), ('127.0.0.1', 19871))
    rx.start()
    link = SimLink('127.0.0.1:19871', '127.0.0.1:19870', 6)
    stepper = build_stepper(os.path.join(HERE, 'follower_test_arm.yaml'), os.path.join(HERE, 'test_arm.urdf'))
    yield paths, rx, link, stepper
    rx.stop = True
    rx.join(2)
    link.close()
    rx.sock.close()


def test_episodes_back_to_back(setup):
    paths, rx, link, stepper = setup
    from sim_errors.part_errors import load_scenario
    for k, p in enumerate(paths):
        if k == 1:
            rx.pause_at = rx.seq + 100            # 두 번째 에피소드 중간에 receiver 0.3초 멈춤
        rec, scn, result = run_episode(link, stepper, load_scenario(p), p, home_q=HOME, sensor_seed=k,
                                       reset_id=100 + k, max_time=60, log=lambda *a, **kw: None)
        assert result == 'done', (p, result)
        rows = {kk: np.array(v) for kk, v in rec.rows.items()}
        # 시뮬 시각으로 진행: 전문가 주기 = 물리 주기, 멈춰도 시간이 건너뛰지 않음
        assert np.allclose(np.diff(rows['t']), DT, atol=1e-9)
        assert rows['phase'][0] == 'weld'
        res = {n: lv for n, lv, _ in check(rows, scn, track_tol_mm=2.0)}
        assert all(lv in (OK, WARN) for lv in res.values()), res
        assert res['단계 순서'] == OK and res['정답 재계산'] == OK and res['목표 기하'] == OK
    assert rx.resets == len(paths)
    assert [os.path.basename(s) for s in rx.scenarios] == [os.path.basename(p) for p in paths]
