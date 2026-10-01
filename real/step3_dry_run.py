"""3단계: 정책 드라이런 — 실물 카메라 + 실물 관절값으로 정책을 돌리되 **모터에는 아무것도 안 보냄**.

정책이 보내려던 팔 목표를 출력하고, [실물 카메라 | 목표 자세를 넣은 시뮬 그림] 을 영상으로 저장.
팔이 안 움직이니 정책은 같은 화면을 계속 보고, 처음 예측한 100스텝(4초) 계획을 그대로 보여 준다.
검사: 실측 한계 밖, 한 스텝 이동량, 시뮬 기준 손 최저점(책상 밑으로 가는지).

    .venv-train/Scripts/python.exe real/step3_dry_run.py --port COM3
    (로봇 없이 코드만 점검: --no-robot --sim-camera)
"""
import argparse

import imageio.v2 as imageio
import numpy as np

from common import (OUT, Arm, Camera, Sim, common_args, deg_to_sim, outside_limits, save_side_by_side)
from policy_io import RealPolicy
from sim.scripts.scripted_grasp import ScriptedGraspPolicy

ap = common_args(argparse.ArgumentParser())
ap.add_argument("--policy", default=None)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--no-robot", action="store_true", help="로봇 없이: 관절값을 시뮬 팔 펴기 자세로 가정")
ap.add_argument("--sim-camera", action="store_true", help="카메라 대신 시뮬 화면(캔 왼쪽 30도)")
args = ap.parse_args()

sim = Sim()
sim.set_can((0.039 + 0.40 * 0.866, 0.40 * 0.5))
pol = RealPolicy(args.policy)
if args.no_robot:
    from common import sim_to_deg
    q = sim_to_deg(ScriptedGraspPolicy.Q_UNFOLD_TARGET)
else:
    with Arm(args.port, args.calib) as arm:
        q = arm.read_deg()
print("현재 관절(도):", np.round(q, 1))
cam = None if args.sim_camera else Camera(args.camera)
img = sim.render(deg_to_sim(q)) if args.sim_camera else cam.to_policy(cam.read_raw())

frames, prev, worst = [], q[:4].copy(), {"limit": 0, "step": 0.0, "z": 1.0}
for t in range(args.steps):
    tgt4, raw = pol.act(q, img)
    tgt = np.append(tgt4, q[4])                      # wrist_roll 은 실물 현재값 고정
    lim = outside_limits(tgt)
    step = float(np.max(np.abs(tgt4 - prev)))
    z = sim.hand_min_z(deg_to_sim(tgt))
    worst["limit"] += bool(lim); worst["step"] = max(worst["step"], step); worst["z"] = min(worst["z"], z)
    if t % 10 == 0:
        print(f"t{t:3d} 목표(도) {np.round(tgt4, 1)}  grip지시 {raw[5]:.2f}  시뮬손최저 {z*1000:5.0f}mm"
              f"{'  한계밖:' + str(lim) if lim else ''}", flush=True)
    frames.append(np.concatenate([img, sim.render(deg_to_sim(tgt))], 1))
    prev = tgt4
if cam:
    cam.close()
imageio.mimsave(OUT / "step3_dry_run.mp4", frames, fps=25)
save_side_by_side(OUT / "step3_last.png", img, frames[-1][:, 320:], labels=("정책 입력", "마지막 목표 자세(시뮬)"))
print(f"\n요약: 한계 밖 목표 {worst['limit']}/{args.steps} 스텝, 스텝당 최대 {worst['step']:.1f}도, "
      f"시뮬 손 최저점 최소 {worst['z']*1000:.0f}mm")
print("저장:", OUT / "step3_dry_run.mp4")
