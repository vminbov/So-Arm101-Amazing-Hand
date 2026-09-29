"""MuJoCo 인터랙티브 뷰어 + 키보드 텔레오퍼레이션.

    python -m sim.scripts.view                # 그냥 씬 구경 (물리 정지)
    python -m sim.scripts.view --sim          # 물리 돌리면서 구경
    python -m sim.scripts.view --teleop       # 키보드로 EE 조작 (사람 데이터 수집 환경)
    python -m sim.scripts.view --scripted     # 스크립트 파지를 뷰어에서 재생
    python -m sim.scripts.view --model amazinghand   # AmazingHand 시각 검증 모델

텔레오퍼레이션 키 (뷰어 창에 포커스):
    W/S : +x / -x        A/D : +y / -y        R/F : +z / -z (위/아래)
    Q/E : yaw + / -       스페이스 : 그리퍼 열기/닫기 토글
    N   : 에피소드 리셋    ESC : 종료
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# [FIX] 콘솔이 cp949(한글 Windows 기본)로 캡처/리다이렉트될 때 print()의 "—" 같은
# 유니코드 문자에서 UnicodeEncodeError로 즉시 죽는 문제 — stdout/stderr를 UTF-8로 강제.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "grasp": _ROOT / "robots" / "so101" / "so101_grasp.xml",
    "amazinghand": _ROOT / "robots" / "amazinghand" / "scene_ah_only.xml",
    "handarm": _ROOT / "robots" / "so101_amazinghand" / "so101_amazinghand_scene.xml",
}


def _jog(model_key: str, arm_only: bool = True):
    """키보드로 관절 하나씩 골라 각도 조절 (home 자세 잡기용). 카메라는 마우스로 자유롭게
    회전/줌(스크롤) — 시작할 때 측면 각도로 한 번만 맞춰두고 이후 건드리지 않음.
    [ ] : 조절할 관절 선택 이전/다음     - = : 선택된 관절 각도 감소(2도)/증가(2도)
    P : 지금 qpos 콘솔에 출력(관절별 deg + rad 배열)   N : 전부 0으로 리셋   ESC: 종료
    콘솔에 [선택: 관절이름] 형태로 지금 뭘 조절 중인지, 값이 얼마인지 매번 출력됨.
    """
    m = mujoco.MjModel.from_xml_path(str(MODELS[model_key]))
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(m, d, kid)  # 0이 아니라 지금 확정된 home에서 시작
    mujoco.mj_forward(m, d)

    # 조절 대상: 1~4번 관절만 (shoulder_pan/shoulder_lift/elbow_flex/wrist_flex) —
    # wrist_roll(5번)이랑 손 모터(finger*)는 사용자 요청으로 제외.
    ONLY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
    jids = []
    for j in range(m.njnt):
        if m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        if arm_only and name not in ONLY:
            continue
        jids.append(j)
    print("조절 가능한 관절:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids])

    state = {"sel": 0}
    STEP = np.radians(2.0)

    def cur_name():
        j = jids[state["sel"]]
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)

    def print_status():
        j = jids[state["sel"]]
        qadr = m.jnt_qposadr[j]
        deg = np.degrees(d.qpos[qadr])
        print(f"[선택: {cur_name()}] = {deg:+.1f}deg")

    def _nudge(sign):
        j = jids[state["sel"]]
        qadr = m.jnt_qposadr[j]
        step = STEP * sign
        if m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            d.qpos[qadr] = np.clip(d.qpos[qadr] + step, lo, hi)
        else:
            d.qpos[qadr] = d.qpos[qadr] + step
        mujoco.mj_forward(m, d)
        print_status()

    viewer_box = [None]  # launch_passive가 반환하는 handle을 나중에 채워넣는 상자
                         # (key_cb는 handle이 생기기 전에 정의되므로 클로저로 참조)

    def key_cb(keycode):
        try:
            _key_cb_inner(keycode)
        except Exception as e:
            import traceback
            print("!! key_cb 에러:", e)
            traceback.print_exc()

    def _key_cb_inner(keycode):
        key = chr(keycode) if 0 <= keycode < 256 else ""
        kl = key.lower()
        if kl == "[":
            state["sel"] = (state["sel"] - 1) % len(jids)
            print_status()
        elif kl == "]":
            state["sel"] = (state["sel"] + 1) % len(jids)
            print_status()
        elif key == "-":
            _nudge(-1)
        elif key == "=":
            _nudge(+1)
        elif kl in ("a", "d", "w", "s") and viewer_box[0] is not None:
            # [FIX] 마우스 드래그로 카메라 회전이 이 환경에서 안 먹혀서(스크롤 줌만 됨)
            # 키보드로 대체 — 단, 매 프레임이 아니라 키 누를 때만 1회 반영해서
            # 스크롤 줌(마우스는 이건 잘 됨)이랑 안 부딪히게 함.
            cam = viewer_box[0].cam
            if kl == "a": cam.azimuth -= 10
            elif kl == "d": cam.azimuth += 10
            elif kl == "w": cam.elevation = np.clip(cam.elevation + 10, -89, 89)
            elif kl == "s": cam.elevation = np.clip(cam.elevation - 10, -89, 89)
        elif kl == "p":
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids]
            degs = [float(np.degrees(d.qpos[m.jnt_qposadr[j]])) for j in jids]
            print("=== 현재 관절 각도 (deg) ===")
            for jn, jv in zip(names, degs):
                print(f"  {jn}: {jv:+.2f}")
            print("qpos(rad) =", [d.qpos[m.jnt_qposadr[j]] for j in jids])
        elif kl == "n":
            for j in jids:
                d.qpos[m.jnt_qposadr[j]] = 0.0
            mujoco.mj_forward(m, d)
            print("전부 0으로 리셋")

    print(_jog.__doc__)
    print("(마우스 드래그 회전이 안 먹으면 A/D/W/S로 카메라 돌리세요 — 스크롤 줌은 그대로 됩니다)")
    print_status()
    with mujoco.viewer.launch_passive(
        m, d, key_callback=key_cb, show_left_ui=False, show_right_ui=False
    ) as v:
        viewer_box[0] = v
        v.cam.lookat = [0.15, 0, 0.15]
        v.cam.azimuth = 90.0
        v.cam.elevation = 0.0
        v.cam.distance = 0.5
        while v.is_running():
            v.sync()
            time.sleep(0.02)


def view_raw(model_key: str, run_physics: bool):
    m = mujoco.MjModel.from_xml_path(str(MODELS[model_key]))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"[{model_key}] nq={m.nq} nu={m.nu} nbody={m.nbody}  physics={'on' if run_physics else 'off'}")
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            if run_physics:
                mujoco.mj_step(m, d)
            v.sync()
            time.sleep(m.opt.timestep if run_physics else 0.02)


def _teleop(scripted: bool, robot: str = "jaw"):
    # [FIX] 이전엔 --model 인자를 무시하고 항상 SO101GraspEnv(robot="jaw") 기본값을 써서
    # "--model handarm --scripted"로 실행해도 항상 예전 jaw 모델(so101_grasp)이 떴었음.
    from sim.tasks.grasp_env import SO101GraspEnv
    env = SO101GraspEnv(robot=robot, max_steps=10_000)
    obs, info = env.reset(seed=0)
    m, d = env.model, env.data

    state = {"dpos": np.zeros(3), "dyaw": 0.0, "grip": 0.0, "reset": False}
    STEP = 1.0  # 액션 스케일 (1.0 = max_pos_delta per frame)

    def key_cb(keycode):
        k = chr(keycode) if 0 < keycode < 0x110000 else ""
        kl = k.lower()
        if kl == "w": state["dpos"][0] = STEP
        elif kl == "s": state["dpos"][0] = -STEP
        elif kl == "a": state["dpos"][1] = STEP
        elif kl == "d": state["dpos"][1] = -STEP
        elif kl == "r": state["dpos"][2] = STEP
        elif kl == "f": state["dpos"][2] = -STEP
        elif kl == "q": state["dyaw"] = STEP
        elif kl == "e": state["dyaw"] = -STEP
        elif kl == "n": state["reset"] = True
        elif keycode == 32:  # space
            state["grip"] = 0.0 if state["grip"] > 0.5 else 1.0
            print("grip ->", "CLOSE" if state["grip"] > 0.5 else "OPEN")

    scripted_pol = None
    if scripted:
        from sim.scripts.scripted_grasp import ScriptedGraspPolicy, policy_step
        scripted_pol = ScriptedGraspPolicy(robot=robot)   # [FIX] 여기도 robot 안 넘기던 버그
        scripted_pol.reset()

    print(__doc__)
    with mujoco.viewer.launch_passive(m, d, key_callback=key_cb) as v:
        while v.is_running():
            t0 = time.time()
            if state["reset"]:
                obs, info = env.reset()
                if scripted_pol:
                    scripted_pol.reset()
                state["reset"] = False

            if scripted_pol:
                _, obs, rew, term, trunc, info = policy_step(env, scripted_pol, info)
            else:
                action = np.array([state["dpos"][0], state["dpos"][1], state["dpos"][2],
                                   state["dyaw"], state["grip"]], dtype=np.float32)
                obs, rew, term, trunc, info = env.step(action)
            # 관성 없는 조작감을 위해 방향키 입력은 매 프레임 소거 (누르는 동안만 이동)
            state["dpos"][:] = 0.0
            state["dyaw"] = 0.0
            if term or trunc:
                tag = "SUCCESS" if info.get("is_success") else "done"
                print(f"[{tag}] cube_z={info['cube_pos'][2]:.3f} -> reset")
                obs, info = env.reset()
                if scripted_pol:
                    scripted_pol.reset()
            v.sync()
            time.sleep(max(0.0, env.decim * m.opt.timestep - (time.time() - t0)))
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="grasp")
    ap.add_argument("--sim", action="store_true", help="물리 시뮬레이션 실행")
    ap.add_argument("--teleop", action="store_true", help="키보드 EE 조작")
    ap.add_argument("--scripted", action="store_true", help="스크립트 파지 재생")
    ap.add_argument("--jog", action="store_true", help="키보드로 관절 하나씩 각도 조절 (home 자세 잡기)")
    args = ap.parse_args()

    if args.jog:
        _jog(args.model)
    elif args.teleop or args.scripted:
        # view.py의 --model 키(grasp/handarm)를 grasp_env.SO101GraspEnv의 robot 키(jaw/hand)로 매핑.
        robot = {"grasp": "jaw", "handarm": "hand"}.get(args.model)
        if robot is None:
            raise SystemExit(f"--teleop/--scripted 는 --model grasp|handarm 만 지원 (받음: {args.model})")
        _teleop(scripted=args.scripted, robot=robot)
    else:
        view_raw(args.model, run_physics=args.sim)


if __name__ == "__main__":
    main()
