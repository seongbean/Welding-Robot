"""리더 암(SO-101 개조형) 서보 읽기 드라이버.

- MockDriver   : 하드웨어 없이 파이프라인 테스트용(사인파)
- FeetechDriver: STS3215 계열 서보를 직접 읽음(토크 OFF, 읽기 전용)

TODO(확인 필요)
- 개조형 7축의 서보 ID/모델이 기본 SO-101과 같은지 (기업 측 확인)
- Present_Position 주소(56)와 길이(2)는 STS3215 기준 값. 추가된 서보가 다른 모델이면 바꿔야 함
- LeRobot의 so101_leader 드라이버를 그대로 쓰고 싶다면, read()만 같은 형태로 감싸면 됨
"""
from __future__ import annotations

import math
import time

import numpy as np


class LeaderDriver:
    """read() → (arm_joint_rad[n], aux_value | None)"""

    def read(self) -> tuple[np.ndarray, float | None]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockDriver(LeaderDriver):
    def __init__(self, n_joints: int, amplitude: float = 0.2, period: float = 8.0,
                 has_aux: bool = True):
        self.n = n_joints
        self.amp = amplitude
        self.period = period
        self.has_aux = has_aux
        self.t0 = time.monotonic()

    def read(self):
        t = time.monotonic() - self.t0
        w = 2 * math.pi / self.period
        q = np.array([self.amp * math.sin(w * t + i * 0.7) for i in range(self.n)])
        aux = 0.5 + 0.5 * math.sin(w * t) if self.has_aux else None
        return q, aux


class FeetechDriver(LeaderDriver):
    ADDR_PRESENT_POSITION = 56  # STS3215
    LEN_PRESENT_POSITION = 2
    TICKS_PER_REV = 4096

    def __init__(self, port: str, baudrate: int, arm_ids: list[int],
                 offsets: list[int], signs: list[int],
                 aux_id: int | None = None, aux_range: tuple[int, int] = (0, 4095)):
        from scservo_sdk import (COMM_SUCCESS, GroupSyncRead, PacketHandler,  # noqa
                                 PortHandler)
        self._COMM_SUCCESS = COMM_SUCCESS
        if not (len(arm_ids) == len(offsets) == len(signs)):
            raise ValueError("arm_ids / offsets / signs 길이가 다름")
        self.arm_ids = arm_ids
        self.offsets = np.asarray(offsets, dtype=float)
        self.signs = np.asarray(signs, dtype=float)
        self.aux_id = aux_id
        self.aux_range = aux_range

        self.port = PortHandler(port)
        if not self.port.openPort():
            raise RuntimeError(f"포트 열기 실패: {port}")
        if not self.port.setBaudRate(baudrate):
            raise RuntimeError(f"보레이트 설정 실패: {baudrate}")
        self.pkt = PacketHandler(0)
        self.sync = GroupSyncRead(self.port, self.pkt,
                                  self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
        for sid in self._all_ids():
            self.sync.addParam(sid)

    def _all_ids(self):
        return self.arm_ids + ([self.aux_id] if self.aux_id is not None else [])

    def _raw(self, sid: int) -> int:
        return self.sync.getData(sid, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)

    def read(self):
        if self.sync.txRxPacket() != self._COMM_SUCCESS:
            raise IOError("서보 sync read 실패")
        ticks = np.array([self._raw(s) for s in self.arm_ids], dtype=float)
        q = (ticks - self.offsets) * self.signs * (2 * math.pi / self.TICKS_PER_REV)
        aux = None
        if self.aux_id is not None:
            lo, hi = self.aux_range
            aux = float(np.clip((self._raw(self.aux_id) - lo) / max(hi - lo, 1), 0.0, 1.0))
        return q, aux

    def close(self):
        self.port.closePort()
