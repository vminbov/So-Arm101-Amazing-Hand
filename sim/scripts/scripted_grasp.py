"""스크립트 파지 전문가 (open-loop state machine).

SO101GraspEnv 위에서 파지 시퀀스를 실행한다. A안의 '시뮬레이션 데이터 수집 자동화
스크립트'이며, record_dataset.py가 policy_step()으로 이 policy를 굴려 데모 데이터를 만든다.

- jaw robot: 카테시안 델타 액션 [dx,dy,dz,dyaw,grip]으로 큐브 위 → 하강 → 잡기 → 들기.
- hand robot(AmazingHand + 캔): 관절 목표 액션 [q_arm(5), grip]. UNFOLD(관절공간으로 home
  자세 탈출) → TRANSIT(wrap_planner가 계획한 충돌 없는 경로로 캔 옆면까지) → GRASP → LIFT → HOLD.

    python -m sim.scripts.scripted_grasp --robot hand --episodes 20
    python -m sim.scripts.scripted_grasp --robot hand --episodes 3 --video media/scripted.mp4
"""
from __future__ import annotations

import argparse

import numpy as np

from sim.scripts import wrap_planner
from sim.tasks.grasp_env import SO101GraspEnv


class ScriptedGraspPolicy:
    # jaw: phase -> (도달 허용오차 m, 최소 체류 스텝, 최대 체류 스텝).
    #   tol > 0 : min_steps 이후, 도달하거나 max_steps 초과 시 다음 단계
    #   tol = 0 : 도달 개념 없음 → 정확히 max_steps 유지 (그리퍼 개폐 안정화용)
    _BUDGET = {"APPROACH": (0.012, 20, 70), "DESCEND": (0.010, 20, 70),
               "GRASP": (0.0, 30, 30), "LIFT": (0.03, 15, 80), "HOLD": (0.0, 20, 20)}
    _JAW_DEFAULTS = dict(approach_h=0.08, grasp_h=0.000, lift_h=0.13, align_yaw=True)

    # hand: "우리 모델" home 자세는 shoulder_lift/wrist_flex가 관절 한계에 붙어 있어서
    # 카테시안 IK가 자기충돌 근처 국소해에 갇힌다 — 관절공간으로 먼저 이 자세까지 편다.
    # 손목은 -15deg(손가락이 앞·위, 끝 높이 0.2m 이상)까지만: 예전 목표 +77deg(손이 테이블 쪽을
    # 향함)는 위에서 감싸던 시절 것이라, 옆잡기 캔 위치(x>=0.38)에선 뒤집는 도중 손끝이 캔 윗면을 쳤다.
    Q_UNFOLD_TARGET = np.array([0.01334, -0.5565, 0.46555, np.radians(-15.0), -0.06067])
    UNFOLD_STEPS = 60
    # 5관절 동시 선형보간은 wrist_flex의 ~172도 뒤집기가 테이블 높이를 휩쓸어 캔을 친다.
    # 팔을 먼저 들고(0~45%) 손목은 그 뒤(50~100%)에 돌린다. 손목 구간을 85~100%(9스텝)로
    # 몰면 스텝당 19deg 명령을 서보가 못 따라가(실제 13deg) 72deg 뒤처지고, 그 뒤처진 자세에서
    # TRANSIT을 계획하니 첫 명령이 88deg 되돌아가 손목이 "팍" 튀었다.
    _UNFOLD_JOINT_LO = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
    _UNFOLD_JOINT_HI = np.array([1.0, 0.45, 0.45, 1.0, 1.0])
    UNFOLD_SETTLE_TOL = np.radians(1.0)   # 펴기 끝 — 실제 관절이 목표에 이만큼 들어와야 계획 시작
    UNFOLD_SETTLE_MAX = 20                # 그래도 안 들어오면 이 스텝 뒤엔 그냥 계획
    SETTLE_STEPS = 10     # 잡는 자세 도착 후 팔이 가라앉을 시간
    GRASP_STEPS = 30      # grip 0->1을 서서히 — 한 번에 닫으면 손가락 하나가 먼저 캔을 친다
    HOLD_STEPS = 20

    def __init__(self, robot="jaw", approach_h=None, grasp_h=None, lift_h=None,
                 align_yaw=None, max_pos_delta=0.02):
        self.robot = robot
        d = self._JAW_DEFAULTS
        self.approach_h = d["approach_h"] if approach_h is None else approach_h
        self.grasp_h = d["grasp_h"] if grasp_h is None else grasp_h
        self.lift_h = d["lift_h"] if lift_h is None else lift_h
        self.align_yaw = d["align_yaw"] if align_yaw is None else align_yaw
        self.max_pos_delta = max_pos_delta
        self.reset()

    def reset(self):
        self.phase = "UNFOLD" if self.robot == "hand" else "APPROACH"
        self._t_in_phase = 0
        self._grasp_xy = None       # jaw: DESCEND 종료 시점 큐브 XY로 고정 (드래그 방지)
        self._unfold_start = None
        self._approach = None       # hand: 계획된 관절 경유점
        self._lift = None
        self.plan_info = None
        self.plan_error = None

    def _advance(self, nxt):
        self.phase = nxt
        self._t_in_phase = 0

    # ------------------------------------------------------------------ jaw
    def _goal_for_phase(self, cube):
        z0 = 0.011   # jaw 씬 큐브 스폰 높이 (DEFAULT_CUBE_RANGE z)
        if self.phase == "APPROACH":
            return np.array([cube[0], cube[1], z0 + self.approach_h]), 0.0
        if self.phase == "DESCEND":
            # 큐브는 정지 상태 → 실제 큐브 XY를 계속 추종해 정확히 가운데로
            return np.array([cube[0], cube[1], z0 + self.grasp_h]), 0.0
        # GRASP 진입 시 고정한 XY 사용 (닫는 동안 큐브가 밀려도 목표는 유지)
        xy = self._grasp_xy if self._grasp_xy is not None else cube[:2]
        if self.phase == "GRASP":
            return np.array([xy[0], xy[1], z0 + self.grasp_h]), 1.0
        return np.array([xy[0], xy[1], z0 + self.lift_h]), 1.0

    @staticmethod
    def _cube_yaw(info):
        """큐브 쿼터니언(wxyz, z축 회전만)에서 yaw를 뽑아 정사각형 대칭 [-45°,45°]로 접기."""
        q = np.asarray(info["cube_quat"], float)
        yaw = 2.0 * np.arctan2(q[3], q[0])
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return float(yaw)

    def act(self, info):
        """jaw 전용: [dx,dy,dz,dyaw,grip] 카테시안 델타 액션."""
        tcp = np.asarray(info["tcp_pos"], float)
        cube = np.asarray(info["cube_pos"], float)
        goal, grip = self._goal_for_phase(cube)
        err = goal - tcp
        self._t_in_phase += 1

        # 손목 yaw를 큐브 yaw에 정렬 (LIFT 전까지). 환경이 dyaw를 월드 z축 회전으로 적분.
        dyaw = 0.0
        if self.align_yaw and self.phase in ("APPROACH", "DESCEND"):
            tcp_yaw = 2.0 * np.arctan2(info["tcp_quat"][3], info["tcp_quat"][0])
            dyaw = np.clip((self._cube_yaw(info) - tcp_yaw) / 0.15, -1.0, 1.0)

        tol, min_steps, max_steps = self._BUDGET[self.phase]
        reached = (tol > 0.0 and self._t_in_phase >= min_steps
                   and np.linalg.norm(err) < tol)
        # DESCEND는 실제로 낮게 내려왔을 때만 통과 (도중 접촉으로 err가 작아지는 것 방지)
        if self.phase == "DESCEND":
            reached = reached and tcp[2] < goal[2] + 0.010
        timeout = self._t_in_phase >= max_steps
        if reached or timeout:
            nxt = {"APPROACH": "DESCEND", "DESCEND": "GRASP", "GRASP": "LIFT",
                   "LIFT": "HOLD", "HOLD": "HOLD"}[self.phase]
            if self.phase == "DESCEND":
                self._grasp_xy = cube[:2].copy()   # 닫기 직전 큐브 XY 고정
            if nxt != self.phase:
                self._advance(nxt)

        a = np.clip(err / self.max_pos_delta, -1.0, 1.0)
        return np.array([a[0], a[1], a[2], dyaw, grip], dtype=np.float32)

    # ------------------------------------------------------------------ hand
    def act_hand(self, info, env):
        """hand 전용: (관절 목표 q[5], grip) — env.step_direct_joint()로 넘긴다."""
        self._t_in_phase += 1
        if self.phase == "UNFOLD":
            if self._unfold_start is None:
                self._unfold_start = np.asarray(info["qarm"], float).copy()
            frac = min(1.0, self._t_in_phase / self.UNFOLD_STEPS)
            local = np.clip((frac - self._UNFOLD_JOINT_LO)
                            / (self._UNFOLD_JOINT_HI - self._UNFOLD_JOINT_LO), 0.0, 1.0)
            q = self._unfold_start + local * (self.Q_UNFOLD_TARGET - self._unfold_start)
            settled = np.max(np.abs(np.asarray(info["qarm"]) - self.Q_UNFOLD_TARGET)) < self.UNFOLD_SETTLE_TOL
            if self._t_in_phase >= self.UNFOLD_STEPS and (
                    settled or self._t_in_phase >= self.UNFOLD_STEPS + self.UNFOLD_SETTLE_MAX):
                self._advance("TRANSIT")
                try:
                    self._approach, self._lift, self.plan_info = wrap_planner.plan(
                        env, np.asarray(info["cube_pos"], float))
                except wrap_planner.PlanError as e:
                    self.plan_error = str(e)
                    self._advance("FAILED")
            return q, 0.0

        if self.phase == "FAILED":
            return np.asarray(info["qarm"], float), 0.0

        if self.phase == "TRANSIT":
            k = self._t_in_phase - 1
            if k >= len(self._approach) + self.SETTLE_STEPS - 1:
                self._advance("GRASP")
            return self._approach[min(k, len(self._approach) - 1)], 0.0

        if self.phase == "GRASP":
            grip = min(1.0, self._t_in_phase / self.GRASP_STEPS)
            if self._t_in_phase >= self.GRASP_STEPS:
                self._advance("LIFT")
            return self._approach[-1], grip

        if self.phase == "LIFT":
            k = self._t_in_phase - 1
            if k >= len(self._lift) - 1:
                self._advance("HOLD")
            return self._lift[min(k, len(self._lift) - 1)], 1.0

        return self._lift[-1], 1.0   # HOLD


def policy_step(env: SO101GraspEnv, policy: ScriptedGraspPolicy, info):
    """policy 한 스텝 실행. rollout()과 record_dataset.py가 공유한다.

    반환: (action, next_obs, reward, terminated, truncated, info)
      jaw  action = [dx, dy, dz, dyaw, grip]
      hand action = [q_arm(5), grip]
    """
    if policy.robot == "hand":
        q, grip = policy.act_hand(info, env)
        nobs, rew, term, trunc, info = env.step_direct_joint(q, grip=grip)
        action = np.append(q, grip).astype(np.float32)
    else:
        action = policy.act(info)
        nobs, rew, term, trunc, info = env.step(action)
    return action, nobs, rew, term, trunc, info


def rollout(env: SO101GraspEnv, policy: ScriptedGraspPolicy, seed: int,
            cube_pos=None, record: bool = False):
    opts = {"cube_pos": cube_pos} if cube_pos is not None else None
    obs, info = env.reset(seed=seed, options=opts)
    policy.reset()
    frames, traj = [], []
    done = False
    ep_ret = 0.0
    while not done:
        action, nobs, rew, term, trunc, info = policy_step(env, policy, info)
        ep_ret += rew
        traj.append({"obs": obs, "action": action, "reward": rew,
                     "next_obs": nobs, "terminated": term, "truncated": trunc})
        obs = nobs
        if record:
            frames.append(env._render_cam(env.cameras[0]))
        done = term or trunc
    return {"success": bool(info.get("is_success")), "return": ep_ret,
            "steps": info["step"], "frames": frames, "traj": traj,
            "final_cube_z": float(info["cube_pos"][2]), "plan_error": policy.plan_error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=("jaw", "hand"), default="jaw")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-cube", type=float, nargs=2, default=None,
                    metavar=("X", "Y"), help="모든 에피소드에서 큐브를 이 좌표로 고정")
    ap.add_argument("--video", type=str, default=None, help="처음 몇 에피소드를 mp4로 저장")
    ap.add_argument("--video-episodes", type=int, default=3)
    args = ap.parse_args()

    need_img = args.video is not None
    max_steps = 400 if args.robot == "hand" else 180
    env = SO101GraspEnv(robot=args.robot, image_obs=need_img, cameras=("front",),
                        image_size=(480, 640), max_steps=max_steps)
    policy = ScriptedGraspPolicy(robot=args.robot)

    succ = 0
    all_frames = []
    for ep in range(args.episodes):
        rec = need_img and ep < args.video_episodes
        r = rollout(env, policy, seed=args.seed + ep,
                    cube_pos=tuple(args.fixed_cube) if args.fixed_cube else None,
                    record=rec)
        succ += r["success"]
        note = f" | plan: {r['plan_error']}" if r["plan_error"] else ""
        print(f"ep {ep:3d} | {'OK ' if r['success'] else 'x  '} "
              f"| steps {r['steps']:3d} | return {r['return']:7.2f} | cube_z {r['final_cube_z']:.3f}{note}")
        if rec:
            all_frames.extend(r["frames"])

    n = args.episodes
    print(f"\n성공률: {succ}/{n} = {succ / n:.1%}")

    if args.video and all_frames:
        import imageio.v2 as imageio
        from pathlib import Path
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, all_frames, fps=25)
        print(f"영상 저장: {args.video}  ({len(all_frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
