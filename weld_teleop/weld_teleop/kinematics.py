"""URDF 기반 기구학 유틸 (pinocchio).

리더/팔로워 모두 같은 클래스를 사용한다.
- 리더: fk()만 사용 → 축 수(6/7)는 URDF와 joint 목록(config)으로만 결정됨
- 팔로워: fk() + ik()

툴 오프셋(토치 팁)은 pinocchio 모델에 프레임을 추가하지 않고
flange 포즈에 곱하는 방식으로 처리한다(pinocchio 버전 차이 회피).
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin


def se3_from_xyz_rpy(xyz, rpy) -> pin.SE3:
    return pin.SE3(pin.rpy.rpyToMatrix(*rpy), np.asarray(xyz, dtype=float))


def se3_to_xyzquat(T: pin.SE3) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw]"""
    return pin.SE3ToXYZQUAT(T)


def xyzquat_to_se3(v) -> pin.SE3:
    return pin.XYZQUATToSE3(np.asarray(v, dtype=float))


class ChainKinematics:
    """URDF에서 지정한 관절들만 남긴 체인(나머지 관절은 고정)."""

    def __init__(self, urdf_path: str, joint_names: list[str], ee_frame: str,
                 tool_offset: pin.SE3 | None = None):
        full = pin.buildModelFromUrdf(urdf_path)

        missing = [j for j in joint_names if not full.existJointName(j)]
        if missing:
            raise ValueError(f"URDF에 없는 관절: {missing}")
        if not full.existFrame(ee_frame):
            raise ValueError(f"URDF에 없는 프레임: {ee_frame}")

        # joint_names에 없는 가동 관절(예: 그리퍼)은 0 위치로 잠금
        lock_ids = [full.getJointId(n) for n in full.names[1:] if n not in joint_names]
        self.model = (pin.buildReducedModel(full, lock_ids, pin.neutral(full))
                      if lock_ids else full)
        self.data = self.model.createData()
        self.ee_fid = self.model.getFrameId(ee_frame)
        self.tool = tool_offset if tool_offset is not None else pin.SE3.Identity()

        # config 순서 ↔ pinocchio q 인덱스 매핑 (모든 관절이 1-DoF revolute라고 가정)
        self.joint_names = list(joint_names)
        self._q_idx = [self.model.joints[self.model.getJointId(n)].idx_q
                       for n in joint_names]
        self.lower = self.model.lowerPositionLimit[self._q_idx].copy()
        self.upper = self.model.upperPositionLimit[self._q_idx].copy()

    @property
    def nq(self) -> int:
        return len(self.joint_names)

    # --- 내부: config 순서 벡터 <-> pinocchio q ---
    def _to_pin(self, q: np.ndarray) -> np.ndarray:
        qp = pin.neutral(self.model)
        qp[self._q_idx] = q
        return qp

    def _from_pin(self, qp: np.ndarray) -> np.ndarray:
        return qp[self._q_idx].copy()

    # --- FK ---
    def fk_flange(self, q: np.ndarray) -> pin.SE3:
        qp = self._to_pin(np.asarray(q, dtype=float))
        pin.framesForwardKinematics(self.model, self.data, qp)
        return pin.SE3(self.data.oMf[self.ee_fid])

    def fk(self, q: np.ndarray) -> pin.SE3:
        """베이스 기준 툴(TCP) 포즈."""
        return self.fk_flange(q) * self.tool

    # --- IK (Damped Least Squares, pinocchio 공식 예제 방식) ---
    def ik(self, T_target_tool: pin.SE3, q_seed: np.ndarray,
           iters: int = 20, eps: float = 1e-4, damp: float = 1e-3,
           dt: float = 0.5) -> tuple[np.ndarray, bool, float]:
        """툴 목표 포즈 → 관절각. 반환: (q, 수렴 여부, 최종 오차 norm)."""
        oMdes = T_target_tool * self.tool.inverse()  # flange 목표
        qp = self._to_pin(np.asarray(q_seed, dtype=float))
        err_norm = np.inf
        for _ in range(iters):
            pin.framesForwardKinematics(self.model, self.data, qp)
            iMd = self.data.oMf[self.ee_fid].actInv(oMdes)
            err = pin.log(iMd).vector
            err_norm = float(np.linalg.norm(err))
            if err_norm < eps:
                break
            J = pin.computeFrameJacobian(self.model, self.data, qp, self.ee_fid)
            J = -pin.Jlog6(iMd.inverse()) @ J
            v = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            qp = pin.integrate(self.model, qp, v * dt)
            qp[self._q_idx] = np.clip(qp[self._q_idx], self.lower, self.upper)
        return self._from_pin(qp), err_norm < eps, err_norm
