"""실물 SO-101(팔 5모터) + 외부 카메라 + 시뮬 도우미. real/step*.py 가 공통으로 쓴다 (.venv-train 에서 실행).

안전 원칙
- 모터에 캘리브레이션/설정을 절대 쓰지 않는다(팀 캘리브 보존). 읽기 + Goal_Position + Torque 만.
- 토크를 켜기 전에 Goal_Position = 현재 위치로 맞춘다(켜는 순간 튀지 않게).
- 모든 명령은 실측 관절 한계(hardware_spec.json)로 자르고, 한 스텝 이동량을 제한한다.
- 손가락(AmazingHand)은 건드리지 않는다. wrist_roll 은 현재 실물 값 그대로 고정.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SPEC = json.loads((ROOT / "handeye_calib_Ateam" / "hardware_spec.json").read_text(encoding="utf-8"))
REAL_LIMITS_DEG = np.array([SPEC["joint_limits_deg"][j] for j in ARM])   # 5도 안전여유 포함 실측
OUT = ROOT / "real" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# 실물 각도(LeRobot DEGREES) -> 시뮬 라디안: sim = SIGN * (deg + OFFSET_DEG) (단위 rad 변환).
# 팀 README: 그쪽 각도 = LeRobot 캘리브 기준이라 시뮬에 바로 넣으면 된다고 함 -> 기본은 변환 없음.
# step2 결과(실물 사진 vs 같은 각도의 시뮬 그림)가 어긋나면 여기를 고친다.
SIGN = np.array([1, 1, 1, 1, 1], float)
OFFSET_DEG = np.array([0, 0, 0, 0, 0], float)


def deg_to_sim(deg):
    return np.radians(SIGN * (np.asarray(deg, float) + OFFSET_DEG))


def sim_to_deg(rad):
    return np.degrees(np.asarray(rad, float)) * SIGN - OFFSET_DEG


# ------------------------------------------------------------------ 팔
class Arm:
    """팔 모터 1~5 (그리퍼 6번 자리는 AmazingHand 라 제외)."""

    def __init__(self, port: str, calib_json: str):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        cal = json.loads(Path(calib_json).read_text(encoding="utf-8"))
        motors = {n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(ARM)}
        calib = {n: MotorCalibration(**{k: cal[n][k] for k in ("id", "drive_mode", "homing_offset",
                                                              "range_min", "range_max")}) for n in ARM}
        self.bus = FeetechMotorsBus(port=port, motors=motors, calibration=calib)

    def __enter__(self):
        self.bus.connect()          # 모터 1~5 핑 확인. 캘리브를 모터에 쓰지 않음.
        return self

    def __exit__(self, *exc):
        self.bus.disconnect(disable_torque=False)   # 토크는 그대로 둔다(끄면 팔이 떨어짐)

    def read_deg(self) -> np.ndarray:
        v = self.bus.sync_read("Present_Position")
        return np.array([v[n] for n in ARM], float)

    def torque_on_here(self) -> np.ndarray:
        """현재 위치를 목표로 먼저 써 두고 토크를 켠다 (켜는 순간 이전 목표로 튀는 것 방지)."""
        q = self.read_deg()
        self.bus.sync_write("Goal_Position", dict(zip(ARM, q)))
        self.bus.enable_torque()
        return q

    def torque_off(self):
        self.bus.disable_torque()

    def goto_deg(self, q_deg):
        self.bus.sync_write("Goal_Position", dict(zip(ARM, np.asarray(q_deg, float))))


def clamp_limits(q_deg):
    return np.clip(q_deg, REAL_LIMITS_DEG[:, 0], REAL_LIMITS_DEG[:, 1])


def outside_limits(q_deg):
    q = np.asarray(q_deg)
    return [ARM[i] for i in range(5) if not REAL_LIMITS_DEG[i, 0] <= q[i] <= REAL_LIMITS_DEG[i, 1]]


# ------------------------------------------------------------------ 카메라
class Camera:
    """외부 고정 카메라. 학습 영상과 같게: 16:9 -> 가운데 4:3 자르기 -> 320x240 RGB."""

    def __init__(self, index=0, width=1280, height=720):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {index} 를 못 열었습니다 (--camera 번호 확인)")
        for _ in range(10):         # 자동 노출 안정화
            self.cap.read()

    def read_raw(self) -> np.ndarray:
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("카메라 프레임 읽기 실패")
        return bgr[:, :, ::-1].copy()

    def to_policy(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        cw = min(w, h * 4 // 3)
        x0 = (w - cw) // 2
        return self.cv2.resize(rgb[:, x0:x0 + cw], (320, 240), interpolation=self.cv2.INTER_AREA)

    def close(self):
        self.cap.release()


# ------------------------------------------------------------------ 시뮬 도우미
class Sim:
    """같은 관절각을 시뮬에 넣어 그림/손 높이를 보는 용도 (물리 진행 안 함)."""

    def __init__(self, fitted_camera=False, image_size=(240, 320)):
        """fitted_camera=True: ext_cam 을 팀 캘리브 점으로 맞춘 실물 카메라(real/camera_fit.json)로 바꿔 렌더."""
        import mujoco
        from sim.tasks.grasp_env import SO101GraspEnv
        self.mj = mujoco
        self.env = SO101GraspEnv(robot="hand", image_obs=True, cameras=("ext_cam",), image_size=image_size)
        self.env.reset(seed=0, options={"cube_pos": (0.2, 0.9)})    # 캔은 화면 밖으로 치움
        self.m, self.d = self.env.model, self.env.data
        if fitted_camera:
            fit = json.loads((ROOT / "real" / "camera_fit.json").read_text(encoding="utf-8"))
            cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "ext_cam")
            self.m.cam_pos[cid], self.m.cam_quat[cid], self.m.cam_fovy[cid] = fit["pos"], fit["quat_mujoco"], fit["fovy_deg"]

    def set_can(self, xy):
        e = self.env
        self.d.qpos[e._qadr_cube:e._qadr_cube + 3] = [xy[0], xy[1], e.cube_range[0, 2]]

    def pose(self, q_rad):
        self.d.qpos[self.env._qadr_arm] = q_rad
        self.mj.mj_forward(self.m, self.d)

    def render(self, q_rad) -> np.ndarray:
        self.pose(q_rad)
        return self.env._render_cam("ext_cam")

    def hand_min_z(self, q_rad) -> float:
        """손 충돌형상의 최저점 z(m, 시뮬 책상면=0)."""
        self.pose(q_rad)
        zs = []
        for g in self.env._hand_col_gids:
            p, R, s = self.d.geom_xpos[g], self.d.geom_xmat[g].reshape(3, 3), self.m.geom_size[g]
            if self.m.geom_type[g] == self.mj.mjtGeom.mjGEOM_CAPSULE:
                zs.append(p[2] - s[0] - s[1] * abs(R[2, 2]))
            else:
                zs.append(p[2] - np.abs(R[2, :]) @ s)
        return float(min(zs))


def save_side_by_side(path, *imgs, labels=()):
    from PIL import Image, ImageDraw, ImageFont
    h = max(i.shape[0] for i in imgs)
    tiles = [np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in imgs]
    im = Image.fromarray(np.concatenate(tiles, 1))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 14)
    x = 0
    for t, lab in zip(tiles, labels):
        dr.rectangle([x, 0, x + t.shape[1], 20], fill=(0, 0, 0))
        dr.text((x + 4, 2), lab, font=font, fill=(255, 255, 0))
        x += t.shape[1]
    im.save(path)
    return path


def common_args(ap):
    ap.add_argument("--port", default="COM3", help="팔 USB 포트 (장치관리자 > 포트 에서 확인)")
    ap.add_argument("--calib", default=str(ROOT / "real" / "calibration" / "so101_follower.json"),
                    help="팀이 쓰는 LeRobot 캘리브레이션 json (모터에 쓰지 않고 읽기만)")
    ap.add_argument("--camera", type=int, default=0, help="카메라 번호")
    return ap
