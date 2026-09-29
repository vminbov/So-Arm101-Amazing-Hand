"""시뮬레이션 데모 데이터 수집 (A안 '시뮬레이션 데이터 수집 자동화 스크립트').

스크립트 파지 policy를 N 에피소드 굴려 각 스텝의 (관측, 액션, 보상, done)을
에피소드별 .npz + 전체 meta.json 으로 저장한다.

    python -m sim.scripts.record_dataset --name sim_grasp_v0 --episodes 100
    python -m sim.scripts.record_dataset --name sim_grasp_img --episodes 50 --images

레이아웃 (LeRobotDataset 변환을 염두):
    data/<name>/
        meta.json                # fps, features, 에피소드 수, 성공률 등
        episode_000000.npz       # obs_state[T,6] (학습용: 관절5+grip), state[T,28] (디버깅용),
                                 # action[T,5 jaw | 6 hand], reward[T], done[T],
                                 # (옵션) front[T,H,W,3], top[T,H,W,3]
        episode_000001.npz
        ...

LeRobotDataset(v2)로 올리려면 (Ubuntu/py3.11 환경에서):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset.create(repo_id="local/sim_grasp_v0", fps=meta["fps"], features={...})
    for ep in episodes:
        for t in range(T):
            ds.add_frame({"observation.state": state[t], "action": action[t], ...})
        ds.save_episode(task="grasp the cube")
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.tasks.grasp_env import SO101GraspEnv
from sim.scripts.scripted_grasp import ScriptedGraspPolicy, policy_step

_ROOT = Path(__file__).resolve().parents[2]
# 학습용 관측 = 실물 로봇에서도 똑같이 얻을 수 있는 값만: 팔 관절각 5 + grip 1 (= 액션과 같은 배치).
# state(28D)에는 캔 위치·tcp 등 실물엔 없는 값이 섞여 있어서 디버깅용으로만 남긴다.
OBS_STATE_IDX = [0, 1, 2, 3, 4, 10]


def record(name: str, episodes: int, images: bool, seed: int, robot: str = "jaw",
           cameras=None, image_size=(240, 320), fps=25,
           only_success: bool = False):
    if cameras is None:   # hand: 실물과 같은 외부 고정 카메라 1대뿐
        cameras = ("ext_cam",) if robot == "hand" else ("front", "top")
    out = _ROOT / "data" / name
    out.mkdir(parents=True, exist_ok=True)

    max_steps = 400 if robot == "hand" else 220  # hand: 성공 에피소드가 보통 ~200스텝
    env = SO101GraspEnv(robot=robot, image_obs=images, cameras=cameras, image_size=image_size,
                        max_steps=max_steps, reward_type="dense")
    pol = ScriptedGraspPolicy(robot=robot)

    saved, succ, t0 = 0, 0, time.time()
    ep_index = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        pol.reset()
        S, A, R, D = [], [], [], []
        IMG = {c: [] for c in cameras} if images else None
        done = False
        while not done:
            if images:
                st = obs["state"]
                for c in cameras:
                    IMG[c].append(obs[c])
            else:
                st = obs
            action, nobs, rew, term, trunc, info = policy_step(env, pol, info)
            S.append(np.asarray(st, np.float32))
            A.append(np.asarray(action, np.float32))
            R.append(np.float32(rew))
            D.append(bool(term or trunc))
            obs = nobs
            done = term or trunc

        is_succ = bool(info.get("is_success"))
        succ += is_succ
        if only_success and not is_succ:
            print(f"ep {ep:4d}  skip (실패, only_success)")
            continue

        S = np.stack(S)
        payload = dict(state=S, obs_state=S[:, OBS_STATE_IDX], action=np.stack(A),
                       reward=np.stack(R), done=np.array(D),
                       success=np.array(is_succ))
        if images:
            for c in cameras:
                payload[c] = np.stack(IMG[c]).astype(np.uint8)
        np.savez_compressed(out / f"episode_{ep_index:06d}.npz", **payload)
        ep_index += 1
        saved += 1
        print(f"ep {ep:4d} -> episode_{ep_index-1:06d}.npz  T={len(S):3d}  {'OK' if is_succ else 'x '}")

    meta = {
        "name": name,
        "robot": robot,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": saved,
        "attempted": episodes,
        "success_rate": round(succ / max(episodes, 1), 4),
        "fps": fps,
        "control_hz": round(1.0 / (env.decim * env.model.opt.timestep), 2),
        "action_space": ("[q_arm(5) joint targets rad, grip 0..1]" if robot == "hand" else
                         "[dx, dy, dz, dyaw, grip]  (dx..dz,dyaw in [-1,1] scaled; grip>0.5=close)"),
        "state_layout": ("[디버깅 전용, 실물에 없는 값 포함] arm_qpos[5], arm_qvel[5], grip_norm[1], tcp_pos[3], tcp_quat[4], "
                         "cube_pos[3], cube_quat[4], cube_minus_tcp[3]  = 28"),
        "obs_state_layout": "arm_qpos[5] rad, grip[1] 0..1  = 6  (학습엔 이것 + 카메라만 사용)",
        "images": bool(images),
        "cameras": list(cameras) if images else [],
        "image_size": list(image_size) if images else None,
        "source": "scripted_grasp.ScriptedGraspPolicy on SO101GraspEnv",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장 완료: {out}  ({saved} 에피소드, 성공률 {meta['success_rate']:.0%}, "
          f"{time.time() - t0:.1f}s)")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robot", choices=["jaw", "hand"], default="jaw")
    ap.add_argument("--images", action="store_true", help="카메라 RGB도 저장 (용량 큼)")
    ap.add_argument("--only-success", action="store_true", help="성공 에피소드만 저장")
    args = ap.parse_args()
    record(args.name, args.episodes, args.images, args.seed, robot=args.robot,
           only_success=args.only_success)


if __name__ == "__main__":
    main()
