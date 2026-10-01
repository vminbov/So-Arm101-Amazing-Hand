"""4단계: 시작 자세(시뮬 시연의 '팔 펴기' 끝 자세)로 천천히 이동. **--go 를 붙여야 실제로 움직임.**

- 토크를 켜기 전에 목표 = 현재 위치로 맞춤 (켜는 순간 튀지 않게).
- 시뮬과 같은 순서: 팔(어깨·팔꿈치) 먼저 -> 손목 굽힘 나중. wrist_roll 은 현재값 그대로.
- 스텝당 최대 --max-step 도 (기본 1도 @25Hz = 초당 25도).
- Ctrl+C: 그 자리에서 멈춤(토크 유지). 끝나도 토크는 켜 둔다 — 끄면 팔이 떨어진다.
  토크를 끄려면 팔을 손으로 받친 채 --release 로 실행.

    .venv-train/Scripts/python.exe real/step4_go_start.py --port COM3          (계획만 출력)
    .venv-train/Scripts/python.exe real/step4_go_start.py --port COM3 --go
    .venv-train/Scripts/python.exe real/step4_go_start.py --port COM3 --release
"""
import argparse
import time

import numpy as np

from common import ARM, Arm, clamp_limits, common_args, outside_limits, sim_to_deg
from sim.scripts.scripted_grasp import ScriptedGraspPolicy

ap = common_args(argparse.ArgumentParser())
ap.add_argument("--go", action="store_true")
ap.add_argument("--release", action="store_true", help="토크 끄기 (팔을 손으로 받치고!)")
ap.add_argument("--max-step", type=float, default=1.0, help="스텝(1/25초)당 최대 이동 각도")
args = ap.parse_args()

with Arm(args.port, args.calib) as arm:
    if args.release:
        input("팔을 손으로 받치고 Enter -> 토크 OFF ")
        arm.torque_off()
        print("토크 OFF")
        raise SystemExit
    q0 = arm.read_deg()
    goal = sim_to_deg(ScriptedGraspPolicy.Q_UNFOLD_TARGET)
    goal[4] = q0[4]                                    # wrist_roll 고정
    if outside_limits(goal):
        raise SystemExit(f"시작 자세가 실측 한계 밖: {outside_limits(goal)} -> 중단")
    print("현재(도):", np.round(q0, 1))
    print("목표(도):", np.round(goal, 1))
    # 팔 먼저(0~50%), 손목 굽힘 나중(50~100%) — 관절 직선보간 도중 손이 책상을 훑지 않게
    n = int(np.ceil(np.max(np.abs(goal - q0)) / args.max_step * 2))
    lo, hi = np.array([0, 0, 0, 0.5, 0.5]), np.array([0.5, 0.5, 0.5, 1, 1])
    path = [q0 + np.clip((k / n - lo) / (hi - lo), 0, 1) * (goal - q0) for k in range(1, n + 1)]
    print(f"{n} 스텝 ({n / 25:.1f}초)")
    if not args.go:
        raise SystemExit("계획만 출력했습니다. 실제로 움직이려면 --go")
    input("주변을 비우고 Enter -> 이동 시작 (중간에 Ctrl+C 로 정지) ")
    arm.torque_on_here()
    try:
        for q in path:
            arm.goto_deg(clamp_limits(q))
            time.sleep(1 / 25)
        time.sleep(1.0)
        q1 = arm.read_deg()
        print("도착(도):", np.round(q1, 1), " 목표와 차이:", np.round(q1 - goal, 1))
    except KeyboardInterrupt:
        arm.goto_deg(arm.read_deg())
        print("\n정지: 현재 자세 유지 (토크 ON)")
