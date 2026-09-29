"""SO-101 팔 + SO-ARM_Interface + AmazingHand(오른손)을 하나의 MJCF로 결합.

MuJoCo MjSpec(attach) API로:
  1) SO-101 공식 MJCF에서 평행 그리퍼(moving_jaw, gripper 조인트/액추에이터, tcp/pad) 제거
  2) 손목 끝(gripper 바디)에 hand_mount 사이트 + SO-ARM_Interface 판 지오메트리
  3) AmazingHand 오른손 MJCF를 그 사이트에 attach (prefix="ah_")
  4) AmazingHand 손끝/손바닥에 충돌 지오메트리 추가 (원본은 visual-only 라 물체 통과함)
  5) 손끝 사이 TCP 사이트, home 키프레임 추가 → scene 파일까지 저장

파지 개폐는 MJCF에서 8모터 그대로 두고, 환경(grasp_env.py, robot="hand")에서
grip 스칼라 1개 → 8모터 목표각(OPEN↔CLOSE 보간)으로 매핑한다 (A안 '잡기/펴기 이진').

    python -m sim.robots.build_combined            # 결합 모델 생성
    python -m sim.robots.build_combined --sweep    # 손목 방향 후보 렌더 비교

MOUNT_EULER 은 gripper 바디 프레임 기준 회전(XYZ, deg). 뷰어에서 눈으로 맞추고 이 값만 고침.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]           # .../sim
SO101 = ROOT / "robots" / "so101" / "so101_new_calib.xml"
HAND = ROOT / "robots" / "amazinghand" / "robot.xml"
# [FIX] 이전엔 AmazingHand-main 저장소의 범용 커뮤니티 STL(사용자 실물과 무관)을 썼음.
# 지금은 사용자의 실제 assembly_whole.STEP을 CAD 상에서 STL로 재export한 진짜 인터페이스
# 부품(전역 어셈블리 좌표계 그대로 보존됨)을 사용 — IFACE_POS/EULER는 실측(XCAF)으로 이미
# identity+(-0,0,-0.006)임을 확인해서 그대로 둠 (아래 iface-rel-gripper 검증 참고).
IFACE_STL_SRC = ROOT / "robots" / "interface_real" / "SO-ARM_Interface_real.stl"
OUT_DIR = ROOT / "robots" / "so101_amazinghand"
OUT_XML = OUT_DIR / "so101_amazinghand.xml"

# --- 손목 마운트 (gripper 바디 로컬 프레임) -------------------------------------
# MOUNT_QUAT 이 있으면 그걸 쓰고, 없으면 MOUNT_EULER 사용.
# 아래 기본값은 팔 home 자세에서 "손바닥이 아래(-z), 손가락이 전방(+x)" 가 되도록
# 계산한 값 (gripper world R 기준). 실제 인터페이스 판의 물리적 장착각으로 교체 가능.
MOUNT_POS = (0.0, 0.0, -0.0068432499999999995)
MOUNT_QUAT = (3.0540179094941497e-16, -0.024337354256020242, 0.9997038027274964, -9.543805967169218e-18)
# [사용자 확정] 방향(MOUNT_QUAT)은 위 축 매핑으로 확정 후, 위치는 인터페이스 바깥면
# 좌표(0,0,-0.00234325, IFACE 두께 계산값)에서 사용자가 Z축(그리퍼 로컬, 마운트 깊이
# 방향) -4.5mm 추가 오프셋을 직접 골라서 확정 — 여러 오프셋 후보를 측면/사선 두 각도로
# 렌더한 그리드에서 선택.
# [사용자 정밀 지정, 검증 완료] CAD 실측 합성값(위 주석의 이전 버전)이 여러 차례 시각
# 확인에도 안 맞아서, 사용자가 손 자체의 로컬축(X/Y/Z, 오른손 법칙)과 월드/베이스 좌표축을
# 나란히 그려서 보여준 뒤 정확한 축 매핑을 직접 지정함:
#   hand local +Z -> world +X,  hand local +X -> world -Z,  hand local +Y -> world +Y
# (qpos=0 기준). gripper 바디의 qpos=0 실제 월드 회전(R_gripper_world, 완전한 항등행렬이
# 아니고 so101 관절 체인 특성상 약 2.8도 정도 기울어짐 포함)을 실측해서
# R_mount = R_gripper_world^T @ R_target 로 정확히 역산 — 그리드 스윕/추정이 전혀 아니고,
# 위 3개 축 매핑을 hand_test 모델에서 d.xmat으로 직접 검증(정확히 일치, 오차 <1e-15) 완료.
# IFACE_POS/QUAT는 그대로 두고 hand-rel-interface 실측 대신 이 hand-rel-gripper 직접
# 지정값으로 완전히 대체함 (이전 "실측 CAD 합성" 접근 자체를 폐기).
# [실험/반려됨] Wrist_Roll_Pitch_SO100 STEP 부품의 로컬 좌표축이 MJCF "gripper" 바디의
# 축 정의와 같다고 가정하고 그 상대회전(-45°)을 그대로 IFACE_EULER에 넣어봤는데, 8개
# 각도(0~315, 45° 간격) 후보를 렌더해서 사용자가 직접 비교한 결과 0°(회전 없음, 즉
# 이 아래 원래 값)가 가장 실물과 비슷하다고 확인함 → 그 "45°"는 실제 장착각이 아니라
# 두 좌표계의 로컬축 정의 차이였던 것으로 결론. 교훈: CAD에서 이름만 보고 고른 부품의
# 월드 회전을 MJCF 바디에 곧바로 갖다붙이면 안 되고, 그 부품의 로컬축이 MJCF 바디 축과
# 같은 컨벤션인지 별도로 검증되지 않는 한 신뢰할 수 없음 — 애매하면 이번처럼 여러 각도
# 후보를 렌더해서 사용자가 직접 고르게 하는 편이 훨씬 빠르고 정확함.
MOUNT_EULER = (-125.0, 0.0, 0.0)                 # deg XYZ (fallback / --sweep 용, 미사용)
IFACE_POS = (0.0, 0.0, -0.0055)
# [실측 검증] trimesh boolean intersection으로 interface.stl vs wrist_roll_pitch_so101_v2.stl
# 간섭을 직접 체크함 (gripper 바디의 wrist-상대 pos/quat는 so101_new_calib.xml 원본 값 사용).
# z=-0.006(원래 값): 간섭 0. z=+0.00316(플러시 시도): 간섭 부피가 인터페이스 전체 부피의
# 10.4%(사용자가 "너무 붙었다"고 정확히 짚어낸 그 상태) → -0.0045부터 간섭 시작, -0.0052
# 부근이 사실상 완전 밀착(간섭≈0)인 한계점 → -0.0055로 약간의 여유를 두고 확정.
IFACE_EULER = (0.0, 0.0, 90.0)   # IFACE_QUAT 있으면 이건 무시됨 (구 fallback)
# [실측/유저 확인] 인터페이스 판이 옆으로 삐져나온 "블레이드"처럼 보이는 문제 — Z축
# 90도 회전(반시계, 사용자 지시)만으로는 판의 넓은 면이 팔 축 옆을 향해서 여전히
# 칼날처럼 튀어나와 보였음. Y축 90도를 추가로 넣어 넓은 면이 팔 끝을 정면으로
# 향하게(캡을 씌운 모양) 만든 뒤, 6가지 후보(X0/X90/X-90/X180/Y90/Y-90)를 렌더해서
# 사용자가 5번(Y+90 적용, Z90 다음에 합성) 선택. 합성 순서: 먼저 Z90, 그다음 Y90
# (mju_mulQuat(q, qY90, qZ90)) → 아래 quat으로 굳힘.
IFACE_QUAT = (0.5, 0.5, 0.5, 0.5)   # wxyz. None이면 IFACE_EULER 사용

# AmazingHand 8 finger-motor 각도 프리셋 (env 가 grip 0→1 로 이 두 상태를 보간)
GRIP_OPEN = np.zeros(8)
#           f1m1  f1m2  f2m1  f2m2  f3m1  f3m2  f4m1(thumb) f4m2
GRIP_CLOSE = np.array([1.15, -1.15, 1.15, -1.15, 1.15, -1.15, 1.00, -1.00])

# 손끝 충돌 캡슐: fingertip 바디 로컬 (tip site 가 [0.0273,-0.0277,0.0022] 인 바디들)
_TIP_BODIES = [f"ah_parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999{s}"
               for s in ("", "_2", "_3", "_4")]


def _try_del(spec, getter, name):
    try:
        obj = getter(name)
    except Exception:
        obj = None
    if obj is not None:
        spec.delete(obj)


def _euler_deg_to_quat(rx, ry, rz):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.deg2rad([rx, ry, rz]), "xyz")
    return q


# 원본 so101_new_calib.xml의 wrist_roll 범위(±90°)는 사람식 캔 옆잡기(손가락 줄 수직 = roll 90°)가
# 관절 한계와 정확히 겹쳐 여유가 0. 실물은 더 돌아간다고 사용자 확인(2026-09-29) — 실측값 나오면 교체.
WRIST_ROLL_LIMIT = np.deg2rad(150.0)


def build(mount_euler=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = OUT_DIR / "assets"
    assets.mkdir(exist_ok=True)
    for src in (ROOT / "robots" / "so101" / "assets", ROOT / "robots" / "amazinghand" / "assets"):
        for stl in src.glob("*.stl"):
            shutil.copy2(stl, assets / stl.name)
    if IFACE_STL_SRC.exists():
        shutil.copy2(IFACE_STL_SRC, assets / "SO-ARM_Interface.stl")
    else:
        print("!! SO-ARM_Interface.stl 못 찾음:", IFACE_STL_SRC)

    abs_assets = str(assets.resolve())
    arm = mujoco.MjSpec.from_file(str(SO101))
    arm.meshdir = abs_assets
    arm.joint("wrist_roll").range = [-WRIST_ROLL_LIMIT, WRIST_ROLL_LIMIT]
    arm.actuator("wrist_roll").ctrlrange = [-WRIST_ROLL_LIMIT, WRIST_ROLL_LIMIT]

    # 1) 평행 그리퍼 제거
    # [FIX] moving_jaw 바디만 지우면 "gripper" 바디에 남아있는 고정 조(jaw) 브라켓
    # 메쉬(wrist_roll_follower_so101_v1, visual+collision 2장)가 그대로 남아 렌더/충돌에
    # 계속 등장했음. 사용자의 실제 assembly_whole.STEP(SO101 Assembly 서브트리)에는
    # Moving_Jaw는 물론 이 follower 브라켓 자체가 아예 없음(마운팅 플레이트 이후 바로
    # 인터페이스로 이어짐) — 실물 하드웨어와 일치시키려면 이것도 지워야 한다.
    # [FIX, 2차] sts3215_03a_v1(손목롤 서보 하우징)도 gripper 바디에서는 제거 — 사용자가
    # "덩그러니 있다"고 지적: 실물에서는 이 노출된 하우징이 인터페이스 안쪽으로 들어가
    # 가려지는 구조라, gripper 바디에 남겨두면 인터페이스와 겹쳐서 이상하게 렌더링됨.
    # 다른 관절(base/shoulder/upper_arm/lower_arm)의 같은 메쉬는 실제 서보라 그대로 둠.
    _try_del(arm, arm.actuator, "gripper")
    _try_del(arm, arm.body, "moving_jaw_so101_v1")
    for s in ("tcp", "gripperframe"):
        _try_del(arm, arm.site, s)
    for g in ("pad_fixed", "pad_moving"):
        _try_del(arm, arm.geom, g)
    grip0 = arm.body("gripper")
    for g in list(grip0.geoms):
        if g.meshname in ("wrist_roll_follower_so101_v1", "sts3215_03a_v1"):
            arm.delete(g)

    # 2) 인터페이스 판 + hand_mount 사이트
    grip = arm.body("gripper")
    mi = arm.add_mesh()
    mi.name, mi.file, mi.scale = "soarm_interface", "SO-ARM_Interface.stl", [1e-3, 1e-3, 1e-3]
    # STL 원점이 판 중심이 아니라 오프셋돼 있음 → 지오메트리 pos 로 상쇄
    try:
        from stl import mesh as _stlmesh
        _v = _stlmesh.Mesh.from_file(str(assets / "SO-ARM_Interface.stl")).vectors.reshape(-1, 3)
        _c = (_v.min(0) + _v.max(0)) / 2.0 * 1e-3
    except Exception:
        _c = np.zeros(3)
    gi = grip.add_geom()
    gi.name, gi.type, gi.meshname = "soarm_interface", mujoco.mjtGeom.mjGEOM_MESH, "soarm_interface"
    if IFACE_QUAT is not None:
        _q_iface = np.array(IFACE_QUAT, float); _q_iface /= np.linalg.norm(_q_iface)
    else:
        _q_iface = _euler_deg_to_quat(*IFACE_EULER)
    _R_iface = np.zeros(9); mujoco.mju_quat2Mat(_R_iface, _q_iface); _R_iface = _R_iface.reshape(3, 3)
    # [FIX] IFACE_EULER가 0이 아니게 된 뒤로는 bbox-center 보정도 회전시켜서 빼야 함
    # (안 그러면 -45° 같은 실측 회전이 들어갔을 때 판이 축을 벗어나 엉뚱한 데 놓임).
    gi.pos = tuple(np.array(IFACE_POS) - _R_iface @ _c)
    gi.quat = _q_iface
    gi.rgba, gi.group = [0.55, 0.55, 0.58, 1.0], 2
    gi.contype = gi.conaffinity = 0

    if mount_euler is not None:
        mq = _euler_deg_to_quat(*mount_euler)
    elif MOUNT_QUAT is not None:
        mq = np.array(MOUNT_QUAT, float); mq /= np.linalg.norm(mq)
    else:
        mq = _euler_deg_to_quat(*MOUNT_EULER)
    mount = grip.add_site()
    mount.name, mount.pos = "hand_mount", MOUNT_POS
    mount.quat, mount.group = mq, 3

    # 3) AmazingHand attach
    hand = mujoco.MjSpec.from_file(str(HAND))
    hand.meshdir = abs_assets
    arm.attach(hand, prefix="ah_", site=arm.site("hand_mount"))

    # 4) 충돌 지오메트리 (원본 손은 전부 visual-only)
    for bn in _TIP_BODIES:
        b = arm.body(bn)
        if b is None:
            print("!! tip body 없음:", bn)
            continue
        cg = b.add_geom()
        cg.name = bn.replace("ah_parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999", "ah_tip_col")
        cg.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        cg.fromto = [0.0, 0.0, 0.0022, 0.024, -0.024, 0.0022]
        cg.size = [0.0065, 0, 0]
        cg.rgba = [0.1, 0.9, 0.1, 0.35]
        cg.group = 3   # [FIX] group 미지정 시 기본 0 → 디버그용 충돌 캡슐이 기본 렌더에
                       # 그대로 노출돼 손끝이 굵은 흰색 캡슐처럼 보이는 원인이었음
        cg.condim, cg.priority = 4, 1
        cg.friction = [1.6, 0.05, 0.001]
        cg.contype = cg.conaffinity = 1
    # 손바닥 판 충돌 박스 (ah_r_wrist_interface 로컬)
    palm = arm.body("ah_r_wrist_interface").add_geom()
    palm.name, palm.type = "ah_palm_col", mujoco.mjtGeom.mjGEOM_BOX
    palm.pos, palm.size = [0.03, 0.0, 0.055], [0.028, 0.04, 0.02]
    palm.rgba, palm.condim = [0.1, 0.5, 0.9, 0.3], 4
    palm.group = 3   # [FIX] 위와 동일한 이유로 기본 렌더에서 숨김
    palm.friction = [1.4, 0.05, 0.001]
    palm.contype = palm.conaffinity = 1

    # 5) TCP 사이트 (손끝 중간, 파지 중심)
    # [실측] 새 MOUNT_POS/QUAT(실제 CAD 기반) 적용 후, qpos0(손가락 벌린 상태)에서
    # 4개 손끝 바디 world 위치의 평균을 ah_r_wrist_interface 로컬 좌표계로 역변환한 값.
    # 이전 값(구 마운트 기준 추정치): pos=(0.05, 0.0, 0.12)
    tcp = arm.body("ah_r_wrist_interface").add_site()
    tcp.name, tcp.pos = "tcp", [0.026, 0.001, 0.119]
    tcp.size, tcp.rgba, tcp.group = [0.005, 0.005, 0.005], [0.1, 1, 0.1, 0.7], 2

    model = arm.compile()
    print(f"compiled: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} neq={model.neq}")
    hand_qpos0 = model.qpos0[5:]
    OUT_XML.write_text(arm.to_xml(), encoding="utf-8")
    _write_scene(hand_qpos0, abs_assets)
    print("saved:", OUT_XML.name, "+ so101_amazinghand_scene.xml")
    return model


SCENE_XML = """<?xml version="1.0" ?>
<mujoco model="so101_amazinghand_scene">
  <include file="so101_amazinghand.xml"/>

  <compiler angle="radian" autolimits="true" meshdir="{assets_dir}"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.15"/>
    <material name="table_mat" rgba="0.75 0.6 0.42 1"/>
    <material name="cube_mat" rgba="0.85 0.2 0.2 1"/>
    <material name="can_mat" rgba="0.8 0.8 0.85 1"/>
    <!-- 실물 외부 카메라 스탠드 (SO-ARM100 레포 부품, bottom+top, 독립 스탠드로 세움) -->
    <mesh name="cam_stand_bot" file="cam_mount_bottom_v2.stl" scale="0.001 0.001 0.001"/>
    <mesh name="cam_stand_top" file="cam_mount_top_v2.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
    <light pos="0.3 0 1.6" dir="0 0 -1" directional="true"/>
    <light pos="0.6 0.5 1.2" dir="-0.4 -0.3 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.001" material="groundplane"/>
    <!-- 책상 [실측 2026-09-29]: 로봇 베이스(pan 축, 월드 x=0.039) 기준 뒤로 65mm ~ 앞으로 467mm(세로 532),
         왼쪽(+y) 1150mm ~ 오른쪽 350mm(가로 1500). -->
    <body name="table" pos="0.240 0.400 -0.02">
      <geom name="table_top" type="box" size="0.266 0.750 0.02" material="table_mat" friction="1 0.01 0.001"/>
    </body>
    <!-- [2026-09-15] 큐브 -> 나랑드사이다 245ml(슬림캔) 원통으로 교체. 사용자 확인:
         슬림캔 규격(지름 약 53mm, 높이 약 115mm) — 정확한 실측치는 아니고 추정값.
         mass=0.25kg(내용물 포함 풀캔 가정), 원통 관성 직접 계산(축=로컬 Z).
         geom/body 이름은 하위호환 위해 "cube"/"cube_geom" 그대로 유지(코드에서 참조 다수). -->
    <body name="cube" pos="0.22 0.00 0.0575">
      <freejoint name="cube_free"/>
      <inertial pos="0 0 0" mass="0.25" diaginertia="3.19e-4 3.19e-4 8.78e-5"/>
      <geom name="cube_geom" type="cylinder" size="0.0265 0.0575" material="can_mat"
            friction="1.0 0.03 0.002" solref="0.008 1" solimp="0.97 0.995 0.001" condim="4" priority="2"/>
    </body>
    <camera name="front" pos="0.60 0.00 0.30" mode="targetbody" target="cube" fovy="50"/>
    <camera name="side"  pos="0.18 -0.55 0.32" mode="targetbody" target="cube" fovy="50"/>
    <camera name="top"   pos="0.18 0.00 0.55" mode="targetbody" target="cube" fovy="46"/>

    <!-- 실물 외부 카메라 스탠드 [실측 2026-09-29]: 스탠드 바닥 중심이 로봇 베이스(pan 축, 월드
         (0.039,0)) 정면 450mm -> 월드 (0.489,0). body 원점은 메시 bbox 기준 (+0.0185, -0.0465) 보정.
         폴 긴 축(메시 로컬 +y)을 world +z로 세우고, 꼭대기 절곡부가 로봇 쪽을 보도록 180° yaw 합성.
         실물처럼 팔이 부딪힐 수 있게 충돌 켬(정적 body라 테이블과는 충돌 안 함). -->
    <body name="cam_stand" pos="0.5075 -0.0465 0" quat="0 0 0.7071068 0.7071068">
      <geom name="cam_stand_bot" type="mesh" mesh="cam_stand_bot" rgba="0.75 0.72 0.15 1" group="2"/>
      <geom name="cam_stand_top" type="mesh" mesh="cam_stand_top" rgba="0.12 0.12 0.12 1" group="2"/>
    </body>
    <!-- 실물 외부 카메라 Innomaker U20CAM-720P [실측 2026-09-29]: 렌즈 높이 375mm, 스탠드 중심보다
         25.5mm 로봇 쪽(STL), 로봇 쪽을 보며 아래로 64.95°. 고정 카메라(실물처럼 캔을 따라가지 않음).
         fovy = 1280x720, 수평화각 102° -> 수직 69.6° (핀홀 — 실물 광각 왜곡은 미반영). -->
    <camera name="ext_cam" pos="0.4635 0 0.375" quat="0.690279 0.153347 0.153347 0.690279" fovy="69.57"/>
  </worldbody>

  <keyframe>
    <!-- [사용자 확정, 2026-09-15, 2차 수정] 1차 시도에서 actuator_gainprm(kp)만 0으로
         만들고 actuator_biasprm은 그대로 둬서, MuJoCo position 액추에이터의 bias 항
         (force = biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel, biasprm[1]=-kp≈-998)이
         여전히 거의 무한강성 스프링처럼 qpos=0을 붙잡고 있었음 — 사용자가 렌더 보고
         "중력 있으면 저렇게 못 서있지"라고 바로 지적해서 발견. gainprm과 biasprm을
         둘 다 0으로 만들고(진짜 무동력) 40000스텝(80초) 시뮬레이션해서 재측정한 진짜
         낙하 평형점으로 교체. 시작자세를 바꿔도 여기로 수렴하는지는 1차 시도 때만
         확인했었고 이번 수정판으로는 재검증 안 함(같은 방법론이라 크게 다르진 않을 것).
         이전(가짜) 값: qpos 앞 5개 = 0 0 0 0 0 -->
    <!-- [사용자 수동 미세조정, 2026-09-15, 3차] 위 물리 시뮬레이션 결과를 사용자가
         뷰어(--jog)에서 직접 보고 "2번(shoulder_lift) 더 내려가야, 4번(wrist_flex)
         더 뒤로 꺾여야" 라고 지시 → 두 관절 다 range 한계까지 밀어붙여서 확정
         (shoulder_lift=-100deg, wrist_flex=-95deg, 둘 다 정확히 관절 한계값). -->
    <key name="home"
         qpos="0.02618 -1.74533 1.56380 -1.65806 -0.112 {hand_qpos0} 0.22 0 0.0575 1 0 0 0"
         ctrl="0.02618 -1.74533 1.56380 -1.65806 -0.112 0 0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def _write_scene(hand_qpos0, assets_dir: str):
    scene = (SCENE_XML
             .replace("{hand_qpos0}", " ".join(f"{v:.6g}" for v in hand_qpos0))
             .replace("{assets_dir}", assets_dir.replace("\\", "/")))
    (OUT_DIR / "so101_amazinghand_scene.xml").write_text(scene, encoding="utf-8")


def sweep():
    """여러 MOUNT_EULER 후보를 home 자세에서 렌더 → media/handarm/sweep_*.png"""
    import imageio.v3 as iio
    cands = {
        "A_-125_0_0": (-125, 0, 0), "B_-90_0_0": (-90, 0, 0), "C_-155_0_0": (-155, 0, 0),
        "D_-125_0_90": (-125, 0, 90), "E_-125_0_-90": (-125, 0, -90), "F_-125_90_0": (-125, 90, 0),
    }
    outdir = (ROOT.parent / "media" / "handarm").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    for name, eul in cands.items():
        build(mount_euler=eul)
        m = mujoco.MjModel.from_xml_path(str(OUT_DIR / "so101_amazinghand_scene.xml"))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 600, 800)
        r.update_scene(d, camera="side")
        iio.imwrite(str(outdir / f"sweep_{name}.png"), r.render())
    print("sweep rendered ->", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if a.sweep:
        sweep()
    else:
        build()
