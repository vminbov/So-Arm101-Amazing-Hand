"""1단계: 카메라만 확인 (로봇 연결 불필요, 팔 움직임 없음).

실물 카메라 한 장을 찍어 real/out/ 에 저장:
  step1_raw.jpg          원본
  step1_compare.png      [실물 원본 | 팀 캘리브 점으로 맞춘 카메라의 시뮬 | 예전 시뮬 카메라(v0~v3 학습에 쓴 것)]
실물과 가운데 그림이 비슷하면(책상·스탠드·캔 위치/크기, 위아래 방향) 카메라 맞추기가 맞은 것.
캔을 로봇 정면 반경 약 0.35m(로봇 베이스 중심에서 35cm 앞)에 하나 세워 두고 찍으면 비교가 쉽다.

    .venv-train/Scripts/python.exe real/step1_camera.py --camera 0
"""
import argparse

from PIL import Image

from common import OUT, Camera, Sim, common_args, save_side_by_side
from sim.scripts.scripted_grasp import ScriptedGraspPolicy

args = common_args(argparse.ArgumentParser()).parse_args()
cam = Camera(args.camera)
raw = cam.read_raw()
cam.close()
Image.fromarray(raw).save(OUT / "step1_raw.jpg")
print(f"원본 해상도: {raw.shape[1]}x{raw.shape[0]}")

import numpy as np
from PIL import Image as _I
q_home = ScriptedGraspPolicy.Q_UNFOLD_TARGET
fit = Sim(fitted_camera=True, image_size=(360, 640)); fit.set_can((0.039 + 0.35, 0.0))
old = Sim(image_size=(360, 640)); old.set_can((0.039 + 0.35, 0.0))
small = np.asarray(_I.fromarray(raw).resize((640, 360)))
p = save_side_by_side(OUT / "step1_compare.png", small, fit.render(q_home), old.render(q_home),
                      labels=("실물 원본", "맞춘 카메라 시뮬 (캔 정면 35cm)", "예전 시뮬 카메라"))
print("저장:", OUT / "step1_raw.jpg", "/", p)
