import numpy as np

from weld_teleop.leader_drivers import MockDriver


def test_mock_driver_shapes_and_ranges():
    d = MockDriver(6, amplitude=0.2)
    q, aux = d.read()
    assert q.shape == (6,)
    assert np.all(np.abs(q) <= 0.2)
    assert 0.0 <= aux <= 1.0
    q, aux = MockDriver(5, has_aux=False).read()
    assert q.shape == (5,) and aux is None
