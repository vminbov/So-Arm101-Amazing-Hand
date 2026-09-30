"""record_dataset.py 결과(data/<name>/*.npz) -> LeRobotDataset(v3) 변환.

학습 환경(.venv-train, lerobot 0.6.1)에서 실행한다 — mujoco 불필요.

    .venv-train/Scripts/python.exe -m sim.scripts.to_lerobot --name sim_hand_v0

결과: data/lerobot/<name>/ (lerobot-train --dataset.repo_id=local/<name> --dataset.root=<그 경로>)
학습 입력은 실물에서도 얻을 수 있는 값만: observation.state = obs_state(관절5 + grip),
observation.images.<cam>, action(관절 목표5 + grip). 28D 디버깅 state는 넣지 않는다.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_ROOT = Path(__file__).resolve().parents[2]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "grip"]
TASK = "grasp the can"


def convert(name: str, only_success: bool = True, overwrite: bool = False) -> Path:
    src = _ROOT / "data" / name
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    if meta["robot"] != "hand" or not meta["images"]:
        raise ValueError("hand 로봇 + --images 로 녹화한 데이터만 변환 (학습 입력에 카메라가 필요)")
    dst = _ROOT / "data" / "lerobot" / name
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} 이미 있음 (--overwrite)")
        shutil.rmtree(dst)

    h, w = meta["image_size"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        **{f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3),
                                        "names": ["height", "width", "channels"]}
           for c in meta["cameras"]},
    }
    ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=meta["fps"], features=features,
                               root=dst, robot_type="so101_amazinghand")
    n = 0
    for f in sorted(src.glob("episode_*.npz")):
        ep = np.load(f)
        if only_success and not bool(ep["success"]):
            continue
        for t in range(len(ep["action"])):
            frame = {"observation.state": ep["obs_state"][t].astype(np.float32),
                     "action": ep["action"][t].astype(np.float32), "task": TASK}
            for c in meta["cameras"]:
                frame[f"observation.images.{c}"] = ep[c][t]
            ds.add_frame(frame)
        ds.save_episode()
        n += 1
    ds.finalize()
    print(f"{n} episodes -> {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--all", action="store_true", help="실패 에피소드도 포함")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    convert(args.name, only_success=not args.all, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
