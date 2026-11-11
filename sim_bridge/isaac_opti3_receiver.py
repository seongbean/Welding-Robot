"""Isaac Sim 쪽: UDP로 받은 관절 목표를 opti3(/World/indy)에 넣고 실측 관절값을 돌려보낸다.

Isaac Sim 5.1 standalone 스크립트. Isaac Sim에 들어 있는 파이썬으로 실행한다(추가 설치 없음).
  C:\\isaacsim\\python.bat sim_bridge\\isaac_opti3_receiver.py --usd C:\\...\\opti3_welding_scene\\posco_torch.usd

- 장면 README대로 관절 drive 게인을 kp=1e4, kd=1e3 으로 올린다(그냥 Play하면 중력에 처짐).
- 시작 자세(--initial-q)는 토치가 로봇 앞에서 아래를 향하는 자세(관절 0, -30, 100, 0, 50, 0도).
  follower_opti3.yaml의 initial_q(토치 팁이 로봇 뒤 바닥 근처)보다 사방으로 움직일 여유가 크다.
  목표가 안 오면 마지막 목표를 유지한다.
- 리더가 가리키는 목표 포즈(토치 팁)를 축 마커로 표시한다. 토치 팁(weld_tip)과 마커가 같이 움직이면
  매핑이 맞는 것이다. 마커 z축(파랑)이 토치 방향.
- 리더 쪽은 leader_to_sim.py. 종료는 창 닫기 또는 Ctrl+C.
- --scenario(sim_errors/make_scenarios.py가 만든 JSON)를 주면 모재(바닥 판·웹 판·가접)를 만들고,
  패킷의 scene.s_arc(아크 진행도)에 맞춰 변형·밀림을 그린다. 빨간 선 = 실제 이음매, 파란 선 = 추정 경로.
  보내는 쪽은 expert_to_sim.py (모재 배치도 패킷으로 받는다). 모재는 표시 전용(충돌 없음).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--usd', required=True, help='posco_torch.usd 경로 (폴더 구조 그대로)')
ap.add_argument('--prim', default='/World/indy')
ap.add_argument('--joints', default='joint0,joint1,joint2,joint3,joint4,joint5')
ap.add_argument('--initial-q', default='0,-0.5236,1.7453,0,0.8727,0', help='rad, joint0~5 시작 자세')
ap.add_argument('--listen', default='127.0.0.1:9870', help='관절 목표 수신 주소')
ap.add_argument('--reply', default='127.0.0.1:9871', help='실측 관절값 보낼 주소 (leader_to_sim.py)')
ap.add_argument('--kp', type=float, default=1e4)
ap.add_argument('--kd', type=float, default=1e3)
ap.add_argument('--headless', action='store_true')
ap.add_argument('--base-link', default='/World/indy/link0', help='URDF base_frame(link0)에 해당하는 prim')
ap.add_argument('--no-marker', action='store_true', help='목표 포즈 마커 끄기')
ap.add_argument('--scenario', help='모재 오차 시나리오 JSON (sim_errors/make_scenarios.py)')
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': args.headless})

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, Vt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # sim_errors (numpy만 사용)


def addr(s: str) -> tuple[str, int]:
    host, port = s.rsplit(':', 1)
    return host, int(port)


JOINTS = args.joints.split(',')
Q0 = np.array([float(x) for x in args.initial_q.split(',')])


def check_articulation(stage, usd_path: str, prim_path: str) -> None:
    """world.reset() 전에 articulation이 실제로 있는지 본다. 없으면 이유를 알려 주고 끝낸다.

    posco_torch.usd는 로봇을 ./robot/opti3/opti3.usd, 토치를 ./assets/... 상대 경로로 참조한다.
    usd 파일만 다른 폴더로 복사하면 참조가 끊겨 /World/indy가 빈 껍데기(over)가 되고,
    Isaac은 "Pattern '/World/indy' did not match any articulations"로 죽는다.
    """
    from pxr import UsdPhysics
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if any(r == prim_path or r.startswith(prim_path + '/') for r in roots):
        return
    missing = []
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if layer.anonymous:
            continue
        here = os.path.dirname(layer.realPath or os.path.abspath(usd_path))
        for dep in layer.GetCompositionAssetDependencies():
            if not dep or '://' in dep or dep.startswith('omniverse:'):
                continue
            full = os.path.normpath(dep if os.path.isabs(dep) else os.path.join(here, dep))
            if not os.path.exists(full):
                missing.append(full)
    # Kit의 sys.exit는 정수만 받고, Windows 콘솔은 한글이 깨지므로 영어로 출력 후 exit(1)
    lines = [f'[receiver] ERROR: no articulation (ArticulationRootAPI) under {prim_path}']
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsDefined():
        lines.append(f'  {prim_path} is not defined (probably an unresolved reference)')
    if missing:
        lines.append('  referenced files not found:')
        lines += [f'    {m}' for m in missing]
        lines.append('  -> put the robot/ and assets/ folders next to the usd, or open the usd in the original scene folder')
    if roots:
        lines.append(f'  articulations elsewhere: {roots} (use --prim)')
    print('\n'.join(lines), flush=True)
    app.close()
    sys.exit(1)


if not open_stage(args.usd):
    sys.exit(f'장면 열기 실패: {args.usd}')
check_articulation(omni.usd.get_context().get_stage(), args.usd, args.prim)
world = World(stage_units_in_meters=1.0)
robot = world.scene.add(SingleArticulation(prim_path=args.prim, name='indy'))


def setup() -> tuple[list[int], np.ndarray]:
    """reset 직후 호출: 게인 설정, 시작 자세. (관절 인덱스, 전체 목표 벡터) 반환."""
    world.reset()
    missing = [j for j in JOINTS if j not in robot.dof_names]
    if missing:
        sys.exit(f'{args.prim}에 없는 관절: {missing} (있는 관절: {robot.dof_names})')
    idx = [robot.dof_names.index(j) for j in JOINTS]
    n = robot.num_dof
    robot.get_articulation_controller().set_gains(kps=np.full(n, args.kp), kds=np.full(n, args.kd))
    target = robot.get_joint_positions().copy()
    target[idx] = Q0
    robot.set_joint_positions(target)
    return idx, target


MARKER_ROOT = '/World/teleop_marker'


def make_marker(stage):
    """리더가 가리키는 토치 팁 목표 포즈를 보여 줄 축 마커 (x 빨강, y 초록, z 파랑 = 토치 방향).

    목표는 opti3.urdf의 link0 기준이므로 마커 부모를 link0의 월드 자세에 맞춘다. 물리 없음(표시 전용).
    """
    base = stage.GetPrimAtPath(args.base_link)
    if base.IsValid():
        base_m = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    else:
        print(f'[receiver] 경고: {args.base_link} 없음, 마커를 월드 원점 기준으로 표시')
        base_m = Gf.Matrix4d(1.0)
    UsdGeom.Xform.Define(stage, MARKER_ROOT).AddTransformOp().Set(base_m)
    frame = UsdGeom.Xform.Define(stage, MARKER_ROOT + '/target')
    op = frame.AddTransformOp()
    for i, (axis, color) in enumerate((('X', (1, 0, 0)), ('Y', (0, 1, 0)), ('Z', (0, 0, 1)))):
        c = UsdGeom.Cylinder.Define(stage, f'{MARKER_ROOT}/target/{axis.lower()}')
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(0.003)
        c.CreateHeightAttr(0.06)
        c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        offset = [0.0, 0.0, 0.0]
        offset[i] = 0.03
        c.AddTranslateOp().Set(Gf.Vec3d(*offset))
    tip = UsdGeom.Sphere.Define(stage, MARKER_ROOT + '/target/tip')
    tip.CreateRadiusAttr(0.006)
    tip.CreateDisplayColorAttr([Gf.Vec3f(1, 0.8, 0)])
    UsdGeom.Imageable(frame.GetPrim()).MakeInvisible()   # 첫 목표가 올 때까지 숨김
    return frame, op


marker_shown = False


def set_marker(xyzquat) -> None:
    global marker_shown
    x, y, z, qx, qy, qz, qw = (float(v) for v in xyzquat)
    m = Gf.Matrix4d(1.0)
    m.SetRotateOnly(Gf.Quatd(qw, qx, qy, qz))
    m.SetTranslateOnly(Gf.Vec3d(x, y, z))
    marker_op.Set(m)
    if not marker_shown:
        UsdGeom.Imageable(marker.GetPrim()).MakeVisible()
        marker_shown = True


WP_ROOT = '/World/workpiece'


def base_matrix(stage) -> Gf.Matrix4d:
    base = stage.GetPrimAtPath(args.base_link)
    if base.IsValid():
        return UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return Gf.Matrix4d(1.0)


def np_to_gf(T) -> Gf.Matrix4d:
    return Gf.Matrix4d(np.asarray(T, dtype=float).T.tolist())   # USD는 행벡터 규약


class Workpiece:
    """시나리오의 모재를 link0 기준으로 그리고, 아크 진행도에 따라 변형한다 (표시 전용)."""

    def __init__(self, stage, path: str):
        from sim_errors.part_errors import PartErrorModel, load_scenario
        self.model = PartErrorModel(load_scenario(path))
        self.pts0, self.plate, counts, idx = self.model.nominal_meshes()
        self.s_grid = np.linspace(0.0, 1.0, 121)
        UsdGeom.Xform.Define(stage, WP_ROOT).AddTransformOp().Set(base_matrix(stage))
        self.meshes = []
        for pid, name, color in ((0, 'base_plate', (0.55, 0.57, 0.6)), (1, 'web_plate', (0.62, 0.6, 0.58))):
            sel = self.plate == pid
            remap = -np.ones(len(self.plate), dtype=int)
            remap[sel] = np.arange(sel.sum())
            k, f_counts, f_idx = 0, [], []
            for c in counts:
                face = idx[k:k + c]
                k += c
                if sel[face].all():
                    f_counts.append(int(c))
                    f_idx += remap[face].tolist()
            m = UsdGeom.Mesh.Define(stage, f'{WP_ROOT}/{name}')
            m.CreateFaceVertexCountsAttr(f_counts)
            m.CreateFaceVertexIndicesAttr(f_idx)
            m.CreateSubdivisionSchemeAttr('none')
            m.CreateDoubleSidedAttr(True)
            m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            self.meshes.append((m, sel))
        self.tacks = []
        for i, t in enumerate(self.model.scn['shape']['tacks']):
            x = UsdGeom.Xform.Define(stage, f'{WP_ROOT}/tack{i}')
            op = x.AddTransformOp()
            c = UsdGeom.Capsule.Define(stage, f'{WP_ROOT}/tack{i}/bead')
            c.CreateAxisAttr('X')
            c.CreateRadiusAttr(t['height'])
            c.CreateHeightAttr(max(0.0, t['length'] - 2 * t['height']))
            c.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.3, 0.25)])
            self.tacks.append((op, t['s']))
        self.true_curve = self._curve(stage, f'{WP_ROOT}/seam_true', (1, 0.1, 0.1))
        self.est_curve = self._curve(stage, f'{WP_ROOT}/seam_est', (0.1, 0.3, 1))
        self.s_arc = None
        self.update(0.0, force=True)

    @staticmethod
    def _curve(stage, path, color):
        c = UsdGeom.BasisCurves.Define(stage, path)
        c.CreateTypeAttr('linear')
        c.CreateWidthsAttr([0.0015])
        c.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return c

    @staticmethod
    def _set_curve(c, P):
        c.CreateCurveVertexCountsAttr([len(P)])
        c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(P, dtype=np.float32)))

    def set_nominal(self, T_nominal) -> None:
        T = np.asarray(T_nominal, dtype=float).reshape(4, 4)
        if not np.allclose(T, np.asarray(self.model.scn['T_nominal'])):
            self.model.set_nominal(T)
            self.update(self.s_arc or 0.0, force=True)

    def update(self, s_arc: float, force: bool = False) -> None:
        if not force and self.s_arc is not None and abs(s_arc - self.s_arc) < 2e-3:
            return
        self.s_arc = s_arc
        P = self.model.deform_points(self.pts0, self.plate, s_arc).astype(np.float32)
        for m, sel in self.meshes:
            m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P[sel]))
        Pt, _ = self.model.seam_true(self.s_grid, s_arc)
        Pe, _ = self.model.seam_est(self.s_grid)
        self._set_curve(self.true_curve, Pt)
        self._set_curve(self.est_curve, Pe)
        for op, s in self.tacks:
            p, R = self.model.seam_true(s, s_arc)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R[0], p[0]
            op.Set(np_to_gf(T))


idx, target = setup()
marker, marker_op = (None, None) if args.no_marker else make_marker(omni.usd.get_context().get_stage())
workpiece = Workpiece(omni.usd.get_context().get_stage(), args.scenario) if args.scenario else None
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(addr(args.listen))
sock.setblocking(False)
reply = addr(args.reply)
print(f'[receiver] {args.prim} {JOINTS} 대기 중: {args.listen} → 실측 {args.reply}')

n_rx = 0
try:
    while app.is_running():
        if world.is_stopped():           # GUI에서 Stop 후 Play 하면 다시 초기화
            world.step(render=True)
            if world.is_playing():
                idx, target = setup()
            continue
        while True:                      # 쌓인 목표 중 마지막 것만 사용
            try:
                data, _ = sock.recvfrom(65536)
            except (BlockingIOError, ConnectionResetError):
                break
            try:
                msg = json.loads(data.decode())
                q = np.asarray(msg['q'], dtype=float)
                tgt = np.asarray(msg.get('target', []), dtype=float)
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            if q.shape == (len(idx),) and np.all(np.isfinite(q)):
                target[idx] = q
                n_rx += 1
            if marker is not None and tgt.shape == (7,) and np.all(np.isfinite(tgt)):
                set_marker(tgt)
            scene = msg.get('scene') if isinstance(msg, dict) else None
            if workpiece is not None and isinstance(scene, dict):
                try:
                    if 'T_nominal' in scene:
                        workpiece.set_nominal(scene['T_nominal'])
                    workpiece.update(float(scene.get('s_arc', 0.0)))
                except (ValueError, TypeError) as e:
                    print(f'[receiver] scene 무시: {e}')
        robot.apply_action(ArticulationAction(joint_positions=target))
        world.step(render=not args.headless)
        if world.is_playing():
            q_meas = robot.get_joint_positions()[idx]
            sock.sendto(json.dumps({'q': q_meas.tolist(), 'rx': n_rx}).encode(), reply)
except KeyboardInterrupt:
    pass
finally:
    sock.close()
    app.close()
