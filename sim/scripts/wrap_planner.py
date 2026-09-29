"""캔 옆면 감싸잡기(hand robot) 경로 계획기.

SO-101은 5축이라 손가락 축(= wrist_roll 축, 손 로컬 +Z)이 항상 "베이스에서 목표를 향하는
수직 평면" 안에 있어야 한다. 그래서 손 자세를 월드 고정값으로 주면(예전
GRASP_QUAT0_HAND_WRAP) 캔이 정면(y=0)이 아닌 곳에서는 도달 불가능했고, 손목을 수직으로
세우는 roll ±90°는 wrist_roll 관절 한계(±90°)와 정확히 겹쳐 여유가 0이었다.

여기서는 자세를 (방위각 phi, 손가락 하향 각 theta, roll psi)로 표현해 팔이 실제로 낼 수 있는
자세 가족 안에서만 고른다. 현재는 사람이 캔 잡듯 옆면 감싸잡기: theta=0(손가락 수평), psi=+90
(손바닥이 옆을 보고 엄지가 위 — wrist_roll 한계를 ±150deg로 넓혀서 가능), 캔 중심은 손 로컬
HAND_GRASP_POINT. 파라미터는 순간이동 평가(펼친 손이 캔을 파고들지 않음 + 관절 여유 + 오므린 뒤
12cm 들어올림 + 손바닥·손가락 4개 접촉)로 골랐다.

경로는 "안전 고도 -> 대기지점 위 -> 수직 하강 -> shoulder_pan 회전으로 옆에서 진입"이고, 대기지점은
잡는 자세를 pan 축 둘레로 돌려 손바닥이 RETREAT만큼 물러난 곳이다. 모든 경로를 5mm 간격 카테시안
경유점으로 쪼개 IK를 이어 풀고, 실행 전에 경유점마다 충돌 검사를 한다.
"""
from __future__ import annotations

import numpy as np
import mujoco

from sim.tasks.grasp_env import HAND_GRASP_POINT as CAN_IN_HAND, PAN_AXIS_XY, _quat_err

TCP_LOCAL = np.array([0.026, 0.001, 0.119])   # 손 로컬 좌표에서의 tcp site 위치(모델 실측)
GRASP_THETA = np.radians(0.0)
GRASP_PSI = np.radians(90.0)
RETREAT = 0.06        # 대기지점: 잡는 자세에서 손바닥 반대 방향으로 물러나는 거리
PULLBACK = 0.045      # 대기지점: 손가락 축(-Z)으로도 물러남 — 캔이 엄지(tip_col_4)와 새끼 끝(tip_col_3) 사이 틈으로 들어오게(4:3 비율이 최대 여유)
HOVER_DZ = 0.10       # 대기지점 위 안전 높이
LIFT_DZ = 0.12
STEP_POS = 0.005      # 카테시안 경유점 간격
STEP_ROT = np.radians(3.0)
PEN_TOL = -0.001      # 이보다 깊게 파고들면 충돌로 판정



def _frame(phi, th, ps):
    f = np.array([np.cos(th) * np.cos(phi), np.cos(th) * np.sin(phi), -np.sin(th)])
    t = np.array([-np.sin(phi), np.cos(phi), 0.0])
    x0 = np.cross(t, f)
    X = np.cos(ps) * x0 + np.sin(ps) * t
    Y = -np.sin(ps) * x0 + np.cos(ps) * t
    return np.column_stack([X, Y, f])


def grasp_frame(cube_pos: np.ndarray, th: float = GRASP_THETA, ps: float = GRASP_PSI,
                can_in_hand: np.ndarray = CAN_IN_HAND) -> np.ndarray:
    """손 로컬 X/Y/Z가 월드에서 향하는 방향을 열로 갖는 회전행렬.

    5축 팔은 손가락 축(= wrist_roll 축, 손 로컬 Z)을 shoulder_pan 축을 지나는 수직 평면 안에만
    둘 수 있다. 캔은 손바닥 앞(CAN_IN_HAND)에 있어서 손목 축은 캔 중심선에서 비켜나 있으므로,
    방위각 phi를 "베이스→캔"이 아니라 "pan축→손 원점" 방향이 되도록 고정점 반복으로 맞춘다
    (안 맞추면 캔이 가까울수록 10~20도 도달 불가 방향을 요구하게 됨)."""
    phi = np.arctan2(cube_pos[1] - PAN_AXIS_XY[1], cube_pos[0] - PAN_AXIS_XY[0])
    for _ in range(50):
        hand = cube_pos - _frame(phi, th, ps) @ can_in_hand
        nxt = np.arctan2(hand[1] - PAN_AXIS_XY[1], hand[0] - PAN_AXIS_XY[0])
        if abs(nxt - phi) < 1e-9:
            break
        phi = nxt
    return _frame(phi, th, ps)


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def _slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    if np.dot(q0, q1) < 0:
        q1 = -q1
    out = np.zeros(4)
    dq = np.zeros(4)
    q0inv = np.zeros(4)
    mujoco.mju_negQuat(q0inv, q0)
    mujoco.mju_mulQuat(dq, q1, q0inv)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, dq, 1.0)
    step = np.zeros(4)
    mujoco.mju_axisAngle2Quat(step, v / (np.linalg.norm(v) + 1e-12), np.linalg.norm(v) * s)
    mujoco.mju_mulQuat(out, step, q0)
    return out / np.linalg.norm(out)


def _rot_about_pan(p, q, a):
    """자세 (p, q)를 shoulder_pan 축(월드 z, PAN_AXIS_XY 통과) 둘레로 a rad 회전 — pan 관절만 돌린 것과 같다."""
    c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    o = np.array([*PAN_AXIS_XY, 0.0])
    qz, out = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]), np.zeros(4)
    mujoco.mju_mulQuat(out, qz, q)
    return o + Rz @ (p - o), out


def _segment(p0, q0, p1, q1):
    dq = np.zeros(4)
    q0inv = np.zeros(4)
    mujoco.mju_negQuat(q0inv, q0)
    mujoco.mju_mulQuat(dq, q1, q0inv)
    ang = 2 * np.arccos(np.clip(abs(dq[0]), -1, 1))
    n = max(1, int(np.ceil(max(np.linalg.norm(p1 - p0) / STEP_POS, ang / STEP_ROT))))
    return [(p0 + (p1 - p0) * (k / n), _slerp(q0, q1, k / n)) for k in range(1, n + 1)]


class PlanError(RuntimeError):
    pass


def solve_pose(env, p, qt, q, iters=100):
    """위치+방향을 동시에 맞추는 6D DLS IK (방향 1rad를 위치 0.1m로 환산해 가중).

    env._solve_ik(위치 우선, 방향은 널스페이스 + 예전 rest 자세 바이어스)는 옆잡기 자세처럼
    rest에서 먼 자세에선 방향 12~38deg 오차에서 멈춘다 — 도달 가능한 자세면 이건 0으로 수렴."""
    m, d = env.model, env.data
    qa, da, sid = env._qadr_arm, env._dadr_arm, env._sid_tcp
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    qc = np.zeros(4)
    q_save = d.qpos[qa].copy()
    q = np.asarray(q, float).copy()
    for _ in range(iters):
        d.qpos[qa] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mju_mat2Quat(qc, d.site_xmat[sid])
        e = np.concatenate([p - d.site_xpos[sid], 0.1 * _quat_err(qc, qt)])
        if np.linalg.norm(e) < 1e-5:
            break
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, da], 0.1 * jr[:, da]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.1, 0.1), env._arm_lo, env._arm_hi)
    d.qpos[qa] = q_save
    mujoco.mj_forward(m, d)
    return q


def _robot_collision(env, obstacle_gids: tuple, base_bid: int) -> float:
    """로봇(베이스 제외)이 캔 또는 장애물(테이블·카메라 스탠드)을 가장 깊게 파고든 깊이(m, 음수일수록 깊음)."""
    m, d = env.model, env.data
    env_gids = (env._cube_gid, *obstacle_gids)
    worst = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        g1, g2 = con.geom1, con.geom2
        if (g1 in env_gids) == (g2 in env_gids):
            continue
        b = m.geom_bodyid[g2 if g1 in env_gids else g1]
        if b in (0, env._bid_cube, base_bid):
            continue
        worst = min(worst, con.dist)
    return worst


def plan(env, cube_pos: np.ndarray):
    """현재 팔 상태에서 잡는 자세까지의 관절 경유점 목록과, 들어올리기 경유점 목록을 반환.

    반환: (approach_qs, lift_qs, info). 충돌/수렴 실패 시 PlanError.
    """
    m, d = env.model, env.data
    obstacles = tuple(g for g in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                                  for n in ("table_top", "cam_stand_bot", "cam_stand_top")) if g >= 0)
    pan_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
    base_bid = m.body_parentid[m.jnt_bodyid[pan_jid]]
    qadr = env._qadr_arm
    q_save = d.qpos.copy()
    v_save = d.qvel.copy()

    R = grasp_frame(cube_pos)
    qg = _quat(R)
    hand_g = cube_pos - R @ CAN_IN_HAND
    tcp_g = hand_g + R @ TCP_LOCAL
    # 옆으로 진입: 5축 팔은 손 방향을 고정한 채 옆으로 평행이동을 못 한다(손가락 축이 pan 축을
    # 벗어남 -> IK가 방향을 14deg까지 틀면서 엄지가 캔을 긁음). 대신 shoulder_pan만 돌리면 손바닥이
    # 캔 쪽으로 호를 그리며 들어온다. 손바닥이 RETREAT만큼 물러나는 pan 각도를 대기지점으로 쓴다.
    r = tcp_g - np.array([*PAN_AXIS_XY, 0.0])
    alpha = -RETREAT / float(np.dot(np.cross([0.0, 0.0, 1.0], r), R[:, 0]))
    # 손바닥 방향으로만 들어오면 엄지 끝이 캔 경로에 걸려서, 손가락 축 방향으로도 PULLBACK만큼
    # 물러났다가 같이 줄이며 들어온다(캔이 손 기준 대각선 위에서 들어옴).
    def pre(s):
        return _rot_about_pan(tcp_g - PULLBACK * s * R[:, 2], qg, alpha * s)
    tcp_pre, q_pre = pre(1.0)
    tcp_hover = tcp_pre + np.array([0.0, 0.0, HOVER_DZ])
    n_slide = max(1, int(np.ceil(np.hypot(RETREAT, PULLBACK) / STEP_POS)))
    slide = [pre(1 - k / n_slide) for k in range(1, n_slide + 1)]

    # 대기지점 위까지는 관절공간 직선 보간: 카테시안 slerp 중간 자세는 5축 팔이 못 내는 방향이라
    # IK가 5~9deg 틀어진다. 관절 보간은 매 점이 정확히 실행 가능하고, 충돌 검사도 똑같이 한다.
    q_start = d.qpos[qadr].copy()
    q_hover = solve_pose(env, tcp_hover, q_pre, q_start, iters=300)
    n_hover = max(1, int(np.ceil(np.max(np.abs(q_hover - q_start)) / STEP_ROT)))
    legs = [
        ("to_hover", [q_start + (q_hover - q_start) * (k / n_hover) for k in range(1, n_hover + 1)]),
        ("descend", _segment(tcp_hover, q_pre, tcp_pre, q_pre)),
        ("slide_in", slide),
    ]
    lift = _segment(tcp_g, qg, tcp_g + np.array([0.0, 0.0, LIFT_DZ]), qg)

    lo, hi = env._arm_lo, env._arm_hi
    q = q_start
    approach, worst_pen, min_margin, max_err, max_ori = [], 0.0, np.inf, 0.0, 0.0
    leg_ends = {}
    try:
        for name, pts in legs:
            for pt in pts:
                if name == "to_hover":
                    q = pt
                else:
                    p, qt = pt
                    q = solve_pose(env, p, qt, q)
                d.qpos[qadr] = q
                mujoco.mj_forward(m, d)
                tcp, qc = env._tcp_pose()
                min_margin = min(min_margin, float(np.min(np.minimum(q - lo, hi - q))))
                if name != "to_hover":
                    max_err = max(max_err, float(np.linalg.norm(tcp - p)))
                    max_ori = max(max_ori, float(np.linalg.norm(_quat_err(qc, qt))))
                pen = _robot_collision(env, obstacles, base_bid)
                worst_pen = min(worst_pen, pen)
                if pen < PEN_TOL:
                    raise PlanError(f"{name}: 충돌 {pen*1000:.1f}mm at tcp={np.round(tcp, 3)}")
                approach.append(q.copy())
            leg_ends[name] = len(approach)
        if max_err > 0.01 or max_ori > np.radians(5):
            raise PlanError(f"잡는 경로 도달 불가 (위치오차 {max_err*100:.1f}cm, 방향오차 {np.degrees(max_ori):.0f}deg)")
        lift_qs = []
        for p, qt in lift:
            q = solve_pose(env, p, qt, q)
            lift_qs.append(q.copy())
    finally:
        d.qpos[:] = q_save
        d.qvel[:] = v_save
        mujoco.mj_forward(m, d)
    info = dict(min_margin_deg=float(np.degrees(min_margin)), worst_pen_mm=worst_pen * 1000,
                max_err_cm=max_err * 100, max_ori_deg=float(np.degrees(max_ori)),
                n_approach=len(approach), n_lift=len(lift_qs), leg_ends=leg_ends)
    return approach, lift_qs, info
