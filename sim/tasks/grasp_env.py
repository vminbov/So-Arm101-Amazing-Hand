"""SO-101 tabletop grasp environment (MuJoCo + Gymnasium).

A안(시뮬레이션 데이터 비율 비교 실험)용 최소 파지 환경.

- 로봇: SO-101 (TheRobotStudio 공식 MJCF, 미수정) + 네이티브 평행 그리퍼
- 태스크: 테이블 위 고정 좌표(또는 지정 박스)에 놓인 정육면체를 집어 들어올리기
- 액션: 엔드이펙터 카테시안 델타 [dx, dy, dz, dyaw, grip]
        (내부 damped least-squares IK로 5축 관절 목표각 생성)
- 관측: 상태 벡터 28D (기본) / image_obs=True 시 카메라 RGB 딕셔너리 추가
- AmazingHand는 시각 검증용 별도 모델(sim/robots/amazinghand/scene_ah_only.xml).
  이 환경은 파지 물리를 SO-101 그리퍼로 처리한다(sim-to-real 격차 최소, 대량 롤아웃 안정).

사용 예:
    from sim.tasks.grasp_env import SO101GraspEnv
    env = SO101GraspEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install gymnasium (프로젝트 .venv 안에서)") from e

import mujoco

_HERE = Path(__file__).resolve().parent
MODEL_PATHS = {
    # jaw: SO-101 네이티브 평행 그리퍼 (단순·안정, 대량 롤아웃용 폴백)
    "jaw": _HERE.parent / "robots" / "so101" / "so101_grasp.xml",
    # hand: SO-101 + AmazingHand(오른손). 실물 하드웨어와 동일. 그립은 8모터를
    #       grip 스칼라 1개로 개폐 (A안 '잡기/펴기 이진'). build_combined.py 로 생성.
    "hand": _HERE.parent / "robots" / "so101_amazinghand" / "so101_amazinghand_scene.xml",
}

# --- 관절 / 액추에이터 인덱스 -----------------------------------------------------
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ARM_DOF = 5
GRIPPER_JOINT = "gripper"

# jaw: gripper 관절 각도. 값이 높을수록 열림, 낮을수록 닫힘 (측정 확인).
GRIPPER_OPEN = 1.50
GRIPPER_CLOSE = -0.13

# hand: AmazingHand 8 finger-motor (ah_fingerN_motorM) 개폐 프리셋.
#       grip 0→1 이 OPEN↔CLOSE 를 선형 보간. build_combined.GRIP_CLOSE 와 동일.
HAND_GRIP_OPEN = np.zeros(8)
HAND_GRIP_CLOSE = np.array([1.15, -1.15, 1.15, -1.15, 1.15, -1.15, 1.00, -1.00])
HAND_MOTORS = tuple(f"ah_finger{f}_motor{m}" for f in (1, 2, 3, 4) for m in (1, 2))

# 파지 접근 자세(월드 기준 쿼터니언 wxyz), 로봇별 기본값.
#   jaw : identity — TCP가 그리퍼 손끝, 이 자세로 x≈0.20 근방 테이블까지 도달 확인.
#   hand: [2026-09-15, "우리 모델" 확정 마운트로 재도출] 인터페이스/마운트/home자세를
#         전부 사용자가 직접(축 매핑 지정 + 수동 미세조정) 확정한 뒤, 그 새 기준으로
#         다시 스윕. 이전 값(90° Z, 구 마운트 기준)은 더 이상 안 맞음 — 새 마운트에서
#         Z축 회전은 전부 부분적으로 자기충돌 남아있고(중립 자세에서 워밍스타트해 재확인),
#         월드 Y축 120~180° 범위가 충돌 0/20, 그중 180°(손바닥이 정확히 반대로 뒤집히는
#         자세)가 IK 오차도 제일 작아서(평균 1.5mm, 최대 1.8mm, 20개 테스트점) 확정.
GRASP_QUAT0 = np.array([1.0, 0.0, 0.0, 0.0])
GRASP_QUAT0_HAND = np.array([0.0, 0.0, 1.0, 0.0])   # world Y축 180°
# hand robot IK 널스페이스 rest 자세: 새 마운트에서 자기충돌 없는 해를 오프라인(damp=0.05,
# iters=60)으로 찾아 그 분기(branch)를 그대로 굳힌 값 — z0.06 부근 대표 해.
# [2026-09-15] "우리 모델"(새 마운트) + GRASP_QUAT0_HAND(Y180)로 다시 찾음 — 이전 값은
# 구 마운트 기준이라 완전히 다른 분기였음(그대로 두면 워밍스타트 IK가 잘못된 분기로
# 끌려갈 위험).
Q_REST_HAND = np.array([0.01334, -0.5565, 0.46555, 1.34798, -0.06067])

# 물체 범위의 z = 스폰 높이이자 lift 판정 기준 높이. jaw 씬은 18mm 큐브, hand 씬은
# 나랑드사이다 245ml 슬림캔(원통, 반높이 0.0575m, 이름은 하위호환 위해 "cube" 유지).
DEFAULT_CUBE_RANGE = np.array([[0.16, -0.08, 0.011],
                               [0.24, 0.08, 0.011]])
# hand robot 캔 배치: 로봇 베이스(shoulder_pan 축) 기준 부채꼴에서 뽑는다. 옆잡기 가능 여부는
# pan 축에서의 반경만으로 정해짐(방향 무관, ±60deg 확인): r 0.37 여유 20deg, 0.43 여유 40deg.
# 방향은 실물 고정 카메라(정면 450mm 스탠드, 아래 64.95deg)에 캔 전체가 화면 가장자리 60px 안쪽으로
# 보이는 좌우 13~41deg(scratch_export/cam_region2.py). 실물리 롤아웃: 왼쪽 10~15deg는 손이 대기지점으로
# 물러날 때(정면 쪽) 카메라 스탠드에 부딪혀서, 좌우 대칭 20~40deg로 확정 (arc_grid2.py).
PAN_AXIS_XY = np.array([0.039, 0.0])   # shoulder_pan 회전축의 월드 수평 위치(모델 실측)
HAND_CAN_ARC_R = (0.37, 0.43)
HAND_CAN_ARC_DEG = ((-40.0, -20.0), (20.0, 40.0))   # 같은 길이 구간 — 균등하게 고름
# hand는 z(스폰 높이 = 캔 반높이)만 씀. xy는 위 부채꼴(cube_range를 직접 주면 그 박스).
DEFAULT_CUBE_RANGE_HAND = np.array([[0.30, -0.30, 0.0575],
                                    [0.46, 0.30, 0.0575]])
TABLE_TOP_Z = 0.0
LIFT_HEIGHT = 0.06        # 성공 판정: 큐브가 테이블에서 이만큼 올라오면
GRASP_DIST = 0.05         # 성공 판정: 파지 중심점-큐브 거리 임계
# hand robot의 파지 중심점(손 로컬 좌표, ah_r_wrist_interface 기준). 옆면 감싸잡기에서 캔 중심이
# 들어오는 자리(손바닥 앞 5cm, 새끼 쪽 1.5cm, 손가락 방향 10.5cm -> 손바닥+손가락 4개가 모두 캔에 닿음).
# tcp site(손가락 끝 중심, 로컬 (0.026,0.001,0.119))와 달라서 tcp 기준으로 판정하면 실제로 들어올려도
# 실패로 나옴. wrap_planner가 이 점을 캔에 맞춰 계획한다.
HAND_GRASP_POINT = np.array([0.05, -0.015, 0.105])


def _quat_err(q_cur: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """월드 쿼터니언 두 개의 방향 오차를 3D 회전 벡터로."""
    q_cur_inv = np.zeros(4)
    mujoco.mju_negQuat(q_cur_inv, q_cur)
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, q_des, q_cur_inv)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, dq, 1.0)
    return vel


class SO101GraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 25}

    def __init__(
        self,
        robot: str = "jaw",
        render_mode: str | None = None,
        image_obs: bool = False,
        image_size: tuple[int, int] = (240, 320),
        cameras: tuple[str, ...] = ("front", "top"),
        cube_pos: tuple[float, float] | None = None,
        cube_range: np.ndarray | None = None,
        cube_grid: list[tuple[float, float]] | None = None,
        reward_type: str = "dense",
        max_steps: int = 150,
        control_decimation: int = 20,
        max_pos_delta: float = 0.02,
        max_yaw_delta: float = 0.15,
        seed: int | None = None,
    ):
        super().__init__()
        if robot not in MODEL_PATHS:
            raise ValueError(f"robot must be one of {list(MODEL_PATHS)}")
        self.robot = robot
        model_path = MODEL_PATHS[robot]
        if not model_path.exists():
            hint = " (먼저 `python -m sim.robots.build_combined` 실행)" if robot == "hand" else ""
            raise FileNotFoundError(f"{model_path}{hint}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.image_obs = image_obs
        self.image_size = image_size
        self.cameras = cameras
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.decim = control_decimation
        self.max_pos_delta = max_pos_delta
        self.max_yaw_delta = max_yaw_delta

        # 큐브 위치 샘플링 방식 (우선순위: 고정 > 그리드 > 박스)
        self.cube_pos_fixed = np.array(cube_pos, dtype=float) if cube_pos is not None else None
        self.cube_grid = [np.array(p, dtype=float) for p in cube_grid] if cube_grid else None
        if cube_range is not None:
            self.cube_range = np.array(cube_range)
        else:
            self.cube_range = DEFAULT_CUBE_RANGE_HAND if robot == "hand" else DEFAULT_CUBE_RANGE
        self.can_arc = robot == "hand" and cube_range is None
        self._grid_i = 0

        # id 캐시
        self._sid_tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._bid_cube = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._jid_arm = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self._qadr_arm = np.array([self.model.jnt_qposadr[j] for j in self._jid_arm])
        self._dadr_arm = np.array([self.model.jnt_dofadr[j] for j in self._jid_arm])
        # 그립 액추에이터: jaw = 1개(index 5), hand = 8개(ah_finger*)
        if robot == "jaw":
            jid_grip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
            self._qadr_grip = self.model.jnt_qposadr[jid_grip]
            self._grip_act = np.array([mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_JOINT)])
        else:
            self._qadr_grip = None
            self._grip_act = np.array([mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in HAND_MOTORS])
            self._cube_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
            self._hand_col_gids = {
                i for i in range(self.model.ngeom)
                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
                    ("ah_tip_col", "ah_palm_col"))}
            self._bid_hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ah_r_wrist_interface")
        self._qadr_cube = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self._key_home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._arm_lo = self.model.jnt_range[self._jid_arm, 0].copy()
        self._arm_hi = self.model.jnt_range[self._jid_arm, 1].copy()

        # 액션/관측 공간
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
        )
        state_dim = 28
        if image_obs:
            img_space = {
                c: spaces.Box(0, 255, (image_size[0], image_size[1], 3), np.uint8)
                for c in cameras
            }
            self.observation_space = spaces.Dict(
                {"state": spaces.Box(-np.inf, np.inf, (state_dim,), np.float32), **img_space}
            )
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, (state_dim,), np.float32)

        self._grasp_quat0 = GRASP_QUAT0_HAND if robot == "hand" else GRASP_QUAT0
        # IK 반복횟수 로봇별 최적값(스윕으로 확인): hand는 반복이 많으면 근특이점에서
        # shoulder_pan이 폭주하는 경향이 있어 jaw보다 적게 준다.
        self._ik_iters = 10 if robot == "hand" else 18

        self._renderer: mujoco.Renderer | None = None
        self._viewer = None
        self._step_count = 0
        self._tcp_target_pos = np.zeros(3)
        self._grasp_quat = self._grasp_quat0.copy()
        self._grip_closed = False
        self._grip_cmd = 0.0            # 명령된 그립 정도 0(open)~1(close)
        self._grasp_err = None         # 그립이 처음 닫힌 순간의 |cube_xy - tcp_xy|
        self._t_first_grasp = None     # 처음 파지 성립한 스텝
        if seed is not None:
            self.reset(seed=seed)

    def _apply_grip(self, g: float):
        """g in [0,1] → 액추에이터 ctrl (jaw 1개 / hand 8개)."""
        self._grip_cmd = float(np.clip(g, 0.0, 1.0))
        if self.robot == "jaw":
            self.data.ctrl[self._grip_act[0]] = (
                GRIPPER_CLOSE if self._grip_cmd > 0.5 else GRIPPER_OPEN)
        else:
            targets = HAND_GRIP_OPEN + self._grip_cmd * (HAND_GRIP_CLOSE - HAND_GRIP_OPEN)
            self.data.ctrl[self._grip_act] = targets

    # ------------------------------------------------------------------ IK
    def _solve_ik(self, target_pos, target_quat, iters: int = 12,
                  damp: float = 0.12, ori_gain: float = 0.4):
        """5축 arm IK. 위치를 1순위(DLS), 방향은 널스페이스 2순위로 푼다.
        SO-101은 5자유도라 임의 6D 포즈가 불가능 → 위치 우선이 안정적.
        sim 상태는 건드리지 않고 복원한다.
        iters: 매 control step마다 새로 호출되는 실시간 속도제어형 IK라 완전 수렴이
        필요 없다 — 오히려 iters를 너무 크게(20+) 두면 근특이점 근방에서
        damped-least-squares가 한 스텝 안에서 나쁜 방향으로 폭주해 shoulder_pan이 확
        틀어지는 현상이 있었음(hand 로봇으로 재현·확인). 반대로 너무 작으면(8) 수렴 부족.
        8/12/16/20 스윕 결과 12가 jaw·hand 양쪽에서 최적."""
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        q = self.data.qpos[self._qadr_arm].copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        # rest 자세 바이어스: jaw용 [0,-1.0,1.2,0.9,0]은 hand에 그대로 쓰면 안 됨 —
        # AmazingHand 장착 후의 충돌-없는 해가 있는 elbow/wrist_flex 분기가 jaw와 달라서
        # (shoulder_lift 양수/elbow_flex 큼/wrist_flex 큰 음수), jaw 기준값으로 널스페이스를
        # 당기면 워밍스타트된 실시간 IK가 반대(자기충돌) 분기로 수렴해버림 — 실제로
        # DESCEND 단계에서 ah_palm_col-lower_arm 충돌에 걸려 하강이 멈추는 것으로 확인.
        q_rest = (Q_REST_HAND if self.robot == "hand" else np.array([0.0, -1.0, 1.2, 0.9, 0.0]))
        I3 = np.eye(3)
        for _ in range(iters):
            self.data.qpos[self._qadr_arm] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            p_cur = self.data.site_xpos[self._sid_tcp].copy()
            q_cur = np.zeros(4)
            mujoco.mju_mat2Quat(q_cur, self.data.site_xmat[self._sid_tcp])
            err_p = target_pos - p_cur
            if np.linalg.norm(err_p) < 5e-4:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._sid_tcp)
            Jp = jacp[:, self._dadr_arm]
            Jr = jacr[:, self._dadr_arm]
            # 1순위: 위치
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + damp ** 2 * I3, err_p)
            # 2순위: 방향 (위치를 깨지 않는 널스페이스에서)
            Jp_pinv = Jp.T @ np.linalg.solve(Jp @ Jp.T + damp ** 2 * I3, I3)
            N = np.eye(ARM_DOF) - Jp_pinv @ Jp
            err_r = _quat_err(q_cur, target_quat) * ori_gain
            dq_r = Jr.T @ np.linalg.solve(Jr @ Jr.T + damp ** 2 * I3, err_r)
            dq += N @ dq_r
            # 3순위: rest 자세 바이어스
            dq += N @ (0.02 * (q_rest - q))
            dq = np.clip(dq, -0.15, 0.15)
            q = np.clip(q + dq, self._arm_lo, self._arm_hi)
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)
        return q

    # ------------------------------------------------------------------ obs
    def _tcp_pose(self):
        p = self.data.site_xpos[self._sid_tcp].copy()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self._sid_tcp])
        return p, q

    def _get_state(self) -> np.ndarray:
        arm_q = self.data.qpos[self._qadr_arm].astype(np.float32)
        arm_v = self.data.qvel[self._dadr_arm].astype(np.float32)
        if self.robot == "jaw":
            grip = np.array([(self.data.qpos[self._qadr_grip] - GRIPPER_OPEN)
                             / (GRIPPER_CLOSE - GRIPPER_OPEN)], np.float32)
        else:
            grip = np.array([self._grip_cmd], np.float32)
        tcp_p, tcp_q = self._tcp_pose()
        cube_p = self.data.xpos[self._bid_cube].copy()
        cube_q = self.data.xquat[self._bid_cube].copy()
        return np.concatenate([arm_q, arm_v, grip, tcp_p, tcp_q,
                               cube_p, cube_q, cube_p - tcp_p]).astype(np.float32)

    def _get_obs(self):
        state = self._get_state()
        if not self.image_obs:
            return state
        return {"state": state, **{c: self._render_cam(c) for c in self.cameras}}

    def _render_cam(self, cam: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.image_size)
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    # ------------------------------------------------------------------ core
    def _sample_cube_xy(self) -> np.ndarray:
        if self.cube_pos_fixed is not None:
            return np.array([*self.cube_pos_fixed, self.cube_range[0, 2]])
        if self.cube_grid is not None:
            p = self.cube_grid[self._grid_i % len(self.cube_grid)]
            self._grid_i += 1
            return np.array([p[0], p[1], self.cube_range[0, 2]])
        if self.can_arc:
            lo, hi = HAND_CAN_ARC_DEG[self.np_random.integers(len(HAND_CAN_ARC_DEG))]
            a = np.radians(self.np_random.uniform(lo, hi))
            xy = PAN_AXIS_XY + self.np_random.uniform(*HAND_CAN_ARC_R) * np.array([np.cos(a), np.sin(a)])
            return np.array([xy[0], xy[1], self.cube_range[0, 2]])
        lo, hi = self.cube_range
        xy = self.np_random.uniform(lo[:2], hi[:2])
        return np.array([xy[0], xy[1], lo[2]])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_home)

        cube = self._sample_cube_xy()
        if options and "cube_pos" in options:
            cp = options["cube_pos"]
            cube = np.array([cp[0], cp[1], self.cube_range[0, 2]])
        self.data.qpos[self._qadr_cube:self._qadr_cube + 3] = cube
        self.data.qpos[self._qadr_cube + 3:self._qadr_cube + 7] = [1, 0, 0, 0]
        yaw0 = self.np_random.uniform(-0.4, 0.4) if not (options and options.get("no_cube_yaw")) else 0.0
        self.data.qpos[self._qadr_cube + 3:self._qadr_cube + 7] = \
            [np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]

        mujoco.mj_forward(self.model, self.data)
        # 내부 카테시안 타깃: 위치는 현재 TCP, 방향은 로봇별 기본 접근자세
        self._tcp_target_pos, _ = self._tcp_pose()
        self._grasp_quat = self._grasp_quat0.copy()
        self._grip_closed = False
        self._grasp_err = None
        self._t_first_grasp = None
        self.data.ctrl[:5] = self.data.qpos[self._qadr_arm]   # 팔 액추에이터는 항상 0..4
        self._apply_grip(0.0)
        self._step_count = 0
        return self._get_obs(), self._info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).clip(self.action_space.low,
                                                           self.action_space.high)
        d_pos = action[:3] * self.max_pos_delta
        d_yaw = float(action[3]) * self.max_yaw_delta
        was_closed = self._grip_closed
        self._grip_closed = action[4] > 0.5
        # 그립이 처음 닫히는 순간의 파지 위치 오차 기록 (A안 지표)
        if self._grip_closed and not was_closed and self._grasp_err is None:
            tcp_p, _ = self._tcp_pose()
            cube_p = self.data.xpos[self._bid_cube]
            self._grasp_err = float(np.linalg.norm((cube_p - tcp_p)[:2]))

        # 카테시안 타깃 적분 + 워크스페이스 클램프
        self._tcp_target_pos = self._tcp_target_pos + d_pos
        self._tcp_target_pos[0] = np.clip(self._tcp_target_pos[0], 0.05, 0.34)
        self._tcp_target_pos[1] = np.clip(self._tcp_target_pos[1], -0.22, 0.22)
        self._tcp_target_pos[2] = np.clip(self._tcp_target_pos[2], 0.005, 0.35)
        # 목표가 실제 TCP에서 너무 멀어지지 않도록 leash (IK 추종 실패 시 폭주 방지)
        tcp_now, _ = self._tcp_pose()
        leash = 1.5 * self.max_pos_delta
        self._tcp_target_pos = tcp_now + np.clip(self._tcp_target_pos - tcp_now, -leash, leash)
        if abs(d_yaw) > 1e-9:
            dq = np.array([np.cos(d_yaw / 2), 0, 0, np.sin(d_yaw / 2)])
            new_q = np.zeros(4)
            mujoco.mju_mulQuat(new_q, dq, self._grasp_quat)
            self._grasp_quat = new_q / np.linalg.norm(new_q)

        q_arm = self._solve_ik(self._tcp_target_pos, self._grasp_quat, iters=self._ik_iters)
        self.data.ctrl[:5] = q_arm
        self._apply_grip(1.0 if self._grip_closed else 0.0)

        for _ in range(self.decim):
            mujoco.mj_step(self.model, self.data)

        return self._finish_step()

    def _finish_step(self):
        """step()/step_direct_joint()가 공유하는 마무리 로직(물리 스텝 이후)."""
        self._step_count += 1
        obs = self._get_obs()
        rew, success = self._reward()
        if self._is_grasped() and self._t_first_grasp is None:
            self._t_first_grasp = self._step_count
        terminated = success
        truncated = self._step_count >= self.max_steps
        info = self._info()
        info["is_success"] = success
        if self.render_mode == "human":
            self.render()
        return obs, rew, terminated, truncated, info

    def step_direct_joint(self, q_arm_target: np.ndarray, grip: float = 0.0):
        """IK를 거치지 않고 팔 관절 목표각(5)과 grip(0~1)을 직접 명령한다 (hand robot 스크립트
        정책의 기본 경로 — 관절 경유점은 wrap_planner가 미리 계산). step()과 같은
        (obs, rew, terminated, truncated, info)를 반환."""
        self.data.ctrl[:5] = q_arm_target
        self._apply_grip(grip)
        for _ in range(self.decim):
            mujoco.mj_step(self.model, self.data)
        self._tcp_target_pos, _ = self._tcp_pose()  # 이후 step()을 섞어 써도 여기서부터 이어가도록
        return self._finish_step()

    # ------------------------------------------------------------------ reward
    def _cube_touches_hand(self) -> bool:
        for c in range(self.data.ncon):
            g1, g2 = self.data.contact[c].geom1, self.data.contact[c].geom2
            if self._cube_gid in (g1, g2):
                other = g2 if g1 == self._cube_gid else g1
                if other in self._hand_col_gids:
                    return True
        return False

    def _grasp_point(self) -> np.ndarray:
        if self.robot == "jaw":
            return self._tcp_pose()[0]
        R = self.data.xmat[self._bid_hand].reshape(3, 3)
        return self.data.xpos[self._bid_hand] + R @ HAND_GRASP_POINT

    def _is_grasped(self) -> bool:
        cube_p = self.data.xpos[self._bid_cube]
        near = np.linalg.norm(cube_p - self._grasp_point()) < GRASP_DIST
        if self.robot == "jaw":
            # 턱이 활짝(>0.8) 벌어져 있지 않고 큐브가 TCP 근처면 파지로 간주
            return bool(near and self.data.qpos[self._qadr_grip] < 0.8)
        # hand: 그립 명령이 닫힘이고, 큐브가 손 충돌 지오메트리에 접촉 중
        return bool(near and self._grip_cmd > 0.5 and self._cube_touches_hand())

    def _reward(self):
        cube_p = self.data.xpos[self._bid_cube].copy()
        dist = float(np.linalg.norm(cube_p - self._grasp_point()))
        lifted = cube_p[2] - (TABLE_TOP_Z + self.cube_range[0, 2])
        grasped = self._is_grasped()
        success = bool(grasped and lifted >= LIFT_HEIGHT and dist < GRASP_DIST)
        if self.reward_type == "sparse":
            return (1.0 if success else 0.0), success
        r = -0.02
        r += 0.4 * (1.0 - np.tanh(6.0 * dist))
        r += 0.25 * float(grasped)
        r += 2.0 * float(np.clip(lifted, 0.0, LIFT_HEIGHT)) / LIFT_HEIGHT
        if success:
            r += 5.0
        return float(r), success

    def _info(self) -> dict[str, Any]:
        tcp_p, tcp_q = self._tcp_pose()
        return {
            "tcp_pos": tcp_p, "tcp_quat": tcp_q,
            "qarm": self.data.qpos[self._qadr_arm].copy(),  # UNFOLD 등 관절공간 직접제어용
            "cube_pos": self.data.xpos[self._bid_cube].copy(),
            "cube_quat": self.data.xquat[self._bid_cube].copy(),
            "grip_closed": self._grip_closed,
            "step": self._step_count,
            # --- A안 비율 비교 실험용 에피소드 지표 ---
            "grasp_pos_err": self._grasp_err,          # 그립 닫는 순간 |cube_xy - tcp_xy| (m)
            "time_to_grasp": self._t_first_grasp,      # 파지 성립 스텝 (제어 25Hz)
        }

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_cam(self.cameras[0])
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


try:
    gym.register(id="SO101Grasp-v0", entry_point="sim.tasks.grasp_env:SO101GraspEnv")
except Exception:
    pass
