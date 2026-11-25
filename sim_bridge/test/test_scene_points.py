"""팔·토치 표면 점(sim_bridge/geometry/opti3_scene_points.npz)이 follower 설정과 맞는지 (pinocchio 불필요)."""
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def test_scene_points_match_follower_tool():
    d = np.load(os.path.join(REPO, 'sim_bridge', 'geometry', 'opti3_scene_points.npz'))
    with open(os.path.join(REPO, 'weld_teleop', 'config', 'follower_opti3.yaml'), encoding='utf-8') as f:
        fo = yaml.safe_load(f)['follower_ik_node']['ros__parameters']
    assert np.allclose(d['tip'], fo['tool_xyz'], atol=1e-4)            # 장면 weld_tip = IK 툴 팁
    assert {f'link{i}' for i in range(1, 7)} <= set(d.files)
    torch = d['torch']
    assert len(torch) > 1000
    assert np.linalg.norm(torch - d['tip'], axis=1).min() < 0.002      # 토치 메시 끝이 팁에 닿음
