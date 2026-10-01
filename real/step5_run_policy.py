"""5단계: 정책으로 팔 4관절 실제 제어. **--go 를 붙여야 움직임.** (4단계로 시작 자세에 온 다음 실행)

- 움직이는 관절: shoulder_pan / shoulder_lift / elbow_flex / wrist_flex.
  wrist_roll 은 현재값 고정, 손가락은 안 건드림(잡기 없음).
- 매 스텝 안전 검사: 실측 한계로 자르기, 스텝당 이동량 제한(--max-step),
  시뮬 기준 손 최저점이 --min-z 아래면 그 자리에서 멈춤.
- Ctrl+C: 그 자리에서 멈춤(토크 유지). 기록은 real/out/step5_*.npz / .mp4
- --max-steps 기본 250(10초). 시뮬 시연은 팔 펴기 이후 약 6초에 손가락을 감쌌다.

    .venv-train/Scripts/python.exe real/step5_run_policy.py --port COM3 --go
"""
import argparse
import time

import imageio.v2 as imageio
import numpy as np

from common import (OUT, Arm, Camera, Sim, clamp_limits, common_args, deg_to_sim, sim_to_deg)
from policy_io import RealPolicy
from sim.scripts.scripted_grasp import ScriptedGraspPolicy

ap = common_args(argparse.ArgumentParser())
ap.add_argument("--policy", default=None)
ap.add_argument("--go", action="store_true")
ap.add_argument("--max-steps", type=int, default=250)
ap.add_argument("--max-step", type=float, default=2.0, help="스텝(1/25초)당 최대 이동 각도")
ap.add_argument("--min-z", type=float, default=0.03, help="시뮬 기준 손 최저점 하한(m)")
args = ap.parse_args()
if not args.go:
    raise SystemExit("실제로 움직이려면 --go (먼저 step3 드라이런 결과를 확인하세요)")

sim, pol, cam = Sim(), RealPolicy(args.policy), Camera(args.camera)
start = sim_to_deg(ScriptedGraspPolicy.Q_UNFOLD_TARGET)
log = {"q": [], "target": [], "raw": [], "img": []}
with Arm(args.port, args.calib) as arm:
    q = arm.read_deg()
    far = np.max(np.abs(q[:4] - start[:4]))
    if far > 10:
        raise SystemExit(f"시작 자세에서 {far:.0f}도 떨어져 있음 -> 먼저 step4_go_start.py --go")
    input("주변을 비우고 Enter -> 정책 실행 (Ctrl+C 로 정지) ")
    q = arm.torque_on_here()
    cmd = q.copy()
    reason = "max-steps 도달"
    try:
        for t in range(args.max_steps):
            t0 = time.perf_counter()
            img = cam.to_policy(cam.read_raw())
            q = arm.read_deg()
            tgt4, raw = pol.act(q, img)
            nxt = cmd.copy()
            nxt[:4] = cmd[:4] + np.clip(tgt4 - cmd[:4], -args.max_step, args.max_step)
            nxt = clamp_limits(nxt)
            z = sim.hand_min_z(deg_to_sim(nxt))
            if z < args.min_z:
                reason = f"t{t}: 시뮬 손 최저점 {z*1000:.0f}mm < {args.min_z*1000:.0f}mm -> 정지"
                break
            arm.goto_deg(nxt)
            cmd = nxt
            for k, v in zip(log, (q, nxt, raw, img)):
                log[k].append(v)
            if t % 25 == 0:
                print(f"t{t:3d} 현재 {np.round(q[:4], 1)} -> 명령 {np.round(nxt[:4], 1)}  grip지시 {raw[5]:.2f}", flush=True)
            time.sleep(max(0.0, 1 / 25 - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    arm.goto_deg(arm.read_deg())
    print(f"\n정지 ({reason}). 현재 자세 유지(토크 ON). 되돌리려면 step4_go_start.py --go")
cam.close()
np.savez(OUT / "step5_log.npz", **{k: np.array(v) for k, v in log.items()})
if log["img"]:
    imageio.mimsave(OUT / "step5_camera.mp4", log["img"], fps=25)
print("기록:", OUT / "step5_log.npz", OUT / "step5_camera.mp4")
