"""학습된 LeRobot 정책(ACT 등)을 시뮬에서 폐루프로 돌려 성공률 측정 (.venv-train에서 실행).

    .venv-train/Scripts/python.exe -m sim.scripts.eval_policy \
        --policy outputs/<run>/checkpoints/last/pretrained_model --episodes 20 --video media/eval.mp4

정책 입력은 녹화 때와 같게: observation.state = 관절5 + grip, observation.images.ext_cam.
출력 액션(관절 목표5 + grip)은 step_direct_joint로 그대로 실행한다. 평가 seed는 녹화(0~)와
겹치지 않게 기본 10000부터.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from lerobot.common.control_utils import predict_action
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

from sim.scripts.record_dataset import OBS_STATE_IDX
from sim.tasks.grasp_env import PAN_AXIS_XY, SO101GraspEnv

# 고정 평가 좌표 세트(A안: 모든 조건을 같은 좌표로 평가). 캔 배치 부채꼴 안, 좌우 거울 대칭.
# 좌우 각 9각도 × 5반경 = 90곳. 30곳(5도·3cm 간격)은 95% 신뢰구간이 ±15%p 넘게 넓어서
# 조건 간 차이를 못 가렸다(v2 24/30 vs v1 18/30, p=0.16). 이전 30곳은 이 세트의 부분집합.
GRID_DEG = tuple(np.arange(20.0, 40.01, 2.5))
GRID_R = (0.37, 0.385, 0.40, 0.415, 0.43)


def eval_grid():
    """[(side, deg, r, (x, y))] 오른쪽(-) 45곳 + 왼쪽(+) 45곳."""
    out = []
    for side, sgn in (("right", -1), ("left", 1)):
        for deg in GRID_DEG:
            for r in GRID_R:
                a = np.radians(sgn * deg)
                xy = PAN_AXIS_XY + r * np.array([np.cos(a), np.sin(a)])
                out.append((side, sgn * deg, r, (float(xy[0]), float(xy[1]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="pretrained_model 폴더")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--video", default=None, help="처음 3 에피소드를 mp4로")
    ap.add_argument("--grid", action="store_true", help="무작위 대신 고정 좌표 세트(좌우 대칭 90곳)로 평가")
    args = ap.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    policy = get_policy_class(cfg.type).from_pretrained(args.policy).to("cuda").eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.policy)
    cams = [k.removeprefix("observation.images.") for k in cfg.image_features]
    h, w = cfg.image_features[f"observation.images.{cams[0]}"].shape[1:]
    env = SO101GraspEnv(robot="hand", image_obs=True, cameras=tuple(cams), image_size=(h, w),
                        max_steps=args.max_steps)
    device = torch.device("cuda")

    grid = eval_grid() if args.grid else None
    n_ep = len(grid) if grid else args.episodes
    succ, frames, per_side = 0, [], {}
    for ep in range(n_ep):
        opts = {"cube_pos": grid[ep][3]} if grid else None
        obs, info = env.reset(seed=args.seed + ep, options=opts)
        policy.reset()
        pre.reset(); post.reset()
        done = False
        while not done:
            o = {"observation.state": obs["state"][OBS_STATE_IDX].astype(np.float32),
                 **{f"observation.images.{c}": obs[c] for c in cams}}
            a = predict_action(o, policy, device, pre, post, use_amp=False,
                               task="grasp the can", robot_type="so101_amazinghand")
            a = a.squeeze(0).cpu().numpy() if a.dim() > 1 else a.cpu().numpy()
            obs, _, term, trunc, info = env.step_direct_joint(a[:5], grip=float(np.clip(a[5], 0, 1)))
            done = term or trunc
            if args.video and ep < 3:
                frames.append(obs[cams[0]])
        ok = bool(info.get("is_success"))
        succ += ok
        where = ""
        if grid:
            side, deg, r, _ = grid[ep]
            per_side.setdefault(side, []).append(ok)
            where = f" | {side:5s} {deg:+.1f}deg r={r:.3f}"
        print(f"ep {ep:3d} | {'OK ' if ok else 'x  '} | steps {info['step']:3d} | cube_z {info['cube_pos'][2]:.3f}{where}",
              flush=True)
    print(f"\n성공률: {succ}/{n_ep} = {succ / n_ep:.1%}")
    for side, oks in per_side.items():
        print(f"  {side}: {sum(oks)}/{len(oks)}")
    if args.video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=25)


if __name__ == "__main__":
    main()
