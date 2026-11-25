"""tools/leader_probe.py: 하드웨어 없이 스텁 패킷 핸들러로 스캔/변환 로직 검증."""
import leader_probe as lp


class StubPkt:
    def __init__(self, servos):
        self.servos = servos   # id -> tick

    def ping(self, port, sid):
        return (777, 0, 0) if sid in self.servos else (0, -1001, 0)

    def read2ByteTxRx(self, port, sid, addr):
        assert addr == lp.ADDR_PRESENT_POSITION
        return (self.servos[sid], 0, 0) if sid in self.servos else (0, -1001, 0)


def test_scan_reports_only_responding_ids():
    pkt = StubPkt({1: 2048, 3: 1024, 7: 3072})
    found = lp.scan(None, pkt, range(0, 10))
    assert found == {1: (777, 2048), 3: (777, 1024), 7: (777, 3072)}


def test_ticks_to_deg():
    assert lp.ticks_to_deg(2048) == 0.0
    assert lp.ticks_to_deg(3072) == 90.0
    assert lp.ticks_to_deg(1024) == -90.0


def test_most_moved():
    prev = {1: 2000, 2: 2000, 3: 2000}
    assert lp.most_moved(prev, {1: 2005, 2: 2100, 3: 1950}) == 2
    assert lp.most_moved(prev, {1: 2005, 2: 2010, 3: 1995}) is None
    assert lp.most_moved(prev, {1: -1, 2: 2000, 3: 2000}) is None
