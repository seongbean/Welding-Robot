import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PKG_ROOT)
SO102_URDF = os.path.join(PKG_ROOT, 'urdf', 'so102.urdf')

# colcon 빌드 없이 `pytest weld_teleop/test`로도 실행되도록
for p in (PKG_ROOT, REPO_ROOT, os.path.join(REPO_ROOT, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)
