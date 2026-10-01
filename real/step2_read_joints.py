"""2단계: 모터 연결 + 관절값 읽기 (읽기만. 토크·목표 안 건드림 -> 팔 안 움직임).

- 모터 1~5 가 응답하는지 확인하고, 현재 관절값(도)을 몇 초간 출력.
- 마지막에 [실물 카메라 사진 | 같은 관절각을 넣은 시뮬 그림] 을 real/out/step2_compare.png 로 저장.
  -> 두 그림의 팔 모양이 같으면 각도 기준이 맞는 것. 다르면 common.py 의 SIGN/OFFSET_DEG 를 고친다.
- 팔을 손으로 여러 자세로 옮겨 가며 여러 번 돌려 보면 더 확실하다 (토크가 꺼져 있을 때만).

    .venv-train/Scripts/python.exe real/step2_read_joints.py --port COM3
"""
import argparse
import time

import numpy as np

from common import ARM, OUT, Arm, Camera, Sim, common_args, deg_to_sim, outside_limits, save_side_by_side

ap = common_args(argparse.ArgumentParser())
ap.add_argument("--seconds", type=float, default=5)
args = ap.parse_args()

with Arm(args.port, args.calib) as arm:
    print("모터 1~5 연결 OK")
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        q = arm.read_deg()
        print("  " + "  ".join(f"{n}={v:7.1f}" for n, v in zip(ARM, q)), flush=True)
        time.sleep(0.5)

print("실측 한계 밖:", outside_limits(q) or "없음")
sim = Sim()
z = sim.hand_min_z(deg_to_sim(q))
print(f"같은 각도의 시뮬 손 최저점: {z * 1000:.0f} mm (시뮬 책상면 = 0)")
cam = Camera(args.camera)
real = cam.to_policy(cam.read_raw())
cam.close()
p = save_side_by_side(OUT / "step2_compare.png", real, sim.render(deg_to_sim(q)),
                      labels=("실물", "시뮬 (같은 관절각)"))
np.save(OUT / "step2_joints_deg.npy", q)
print("저장:", p)
