# SO-101 + AmazingHand 파지 시뮬레이션 (A안 · 초기 환경)

A안 "시뮬레이션 데이터 비율 비교 실험"을 위한 MuJoCo 파지 환경.
목표: 실물 teleop 데이터와 섞어 ACT를 학습할 **시뮬레이션 데모 데이터**를
자동으로 뽑아낼 수 있는 최소 파이프라인.

```
sim/
├─ robots/
│  ├─ so101/
│  │   ├─ so101_new_calib.xml      TheRobotStudio 공식 SO-101 MJCF
│  │   │                            (+ [LOCAL EDIT] 표시된 tcp 사이트 & 손가락 패드)
│  │   ├─ so101_new_calib_camera.xml  손목 카메라 포함 변형본 (아직 미사용)
│  │   ├─ so101_grasp.xml          ← 실제 사용 씬: 팔 + 테이블 + 큐브 + 카메라 + home
│  │   └─ assets/*.stl             팔 메시 15개
│  ├─ amazinghand/
│  │   ├─ scene_ah_only.xml        AmazingHand 전체(8모터 병렬링크) 단독 — 시각 검증용
│  │   ├─ robot.xml / *.xml        pollen-robotics 공식 MJCF (onshape-to-robot)
│  │   └─ assets/*.stl             손 메시 29개
│  ├─ so101_amazinghand/           ← SO-101 + AmazingHand 결합 모델
│  │   ├─ so101_amazinghand.xml        결합 본체 (build_combined.py 가 생성)
│  │   ├─ so101_amazinghand_scene.xml  ← 테이블/큐브/카메라/home 포함 씬
│  │   └─ assets/                      팔+손+인터페이스 메시 전부
│  └─ build_combined.py            SO-101 그리퍼 제거 → SO-ARM_Interface 판 →
│                                  AmazingHand attach. MOUNT_* 값만 고쳐 재실행
├─ tasks/
│  └─ grasp_env.py                 Gymnasium 환경 (SO101GraspEnv, id "SO101Grasp-v0")
├─ scripts/
│  ├─ view.py                      뷰어 + 키보드 텔레오퍼레이션
│  ├─ scripted_grasp.py            스크립트 파지 전문가 (상태기계)
│  └─ record_dataset.py            데모 데이터 수집 → data/<name>/
├─ requirements-sim.txt
└─ README.md
```

## 빠른 시작

```bash
# (Windows, 이 폴더의 .venv 사용)
.venv/Scripts/python.exe -m sim.scripts.view --sim               # SO-101 파지 씬
.venv/Scripts/python.exe -m sim.scripts.view --model handarm --sim   # SO-101 + AmazingHand 결합
.venv/Scripts/python.exe -m sim.scripts.view --model amazinghand     # 손 단독(시각 검증)
.venv/Scripts/python.exe -m sim.scripts.view --scripted          # 스크립트 파지 재생
.venv/Scripts/python.exe -m sim.scripts.view --teleop            # 직접 조작 (W/A/S/D/R/F, Q/E, Space)
.venv/Scripts/python.exe -m sim.scripts.scripted_grasp --episodes 20
.venv/Scripts/python.exe -m sim.scripts.record_dataset --name sim_grasp_v0 --episodes 100
.venv/Scripts/python.exe -m sim.robots.build_combined            # 결합 모델 재생성
```

## 환경 사양 (`SO101GraspEnv`)

| 항목 | 값 |
|---|---|
| `robot="jaw"` | SO-101 5축 + 네이티브 평행 그리퍼 (단순·안정, 대량 롤아웃용) |
| `robot="hand"` | SO-101 5축 + **AmazingHand**(실물과 동일). grip 스칼라 1개 → 8모터 개폐 보간 (A안 '잡기/펴기 이진') |
| 액션 | `[dx, dy, dz, dyaw, grip]` — EE 카테시안 델타(±1→±2cm/step), grip>0.5=닫기 |
| IK | 내부 damped-least-squares (위치 1순위, 방향 널스페이스) |
| 제어 주파수 | 25 Hz (timestep 2 ms × decimation 20) |
| 관측(state) | 28D: `arm_qpos[5] arm_qvel[5] grip[1] tcp_pos[3] tcp_quat[4] cube_pos[3] cube_quat[4] (cube-tcp)[3]` |
| 관측(image) | `image_obs=True` 시 `{"state", "front", "top", ...}` dict |
| 보상 | `dense`(리치+파지+리프트 셰이핑) / `sparse`(성공=1) |
| 성공 | 큐브가 테이블에서 6cm 이상 상승 + TCP 근처 + 파지(턱 닫힘 / 손 충돌 접촉) |
| 큐브 위치 | `cube_pos`(고정) / `cube_grid`(리스트 순환) / `cube_range`(균등박스, 기본) |
| **에피소드 지표** (info) | `is_success`(이진), `grasp_pos_err`(그립 닫는 순간 \|cube_xy − tcp_xy\|, m), `time_to_grasp`(파지 성립 스텝) — A안 비율 비교 실험 로깅용 |

```python
from sim.tasks.grasp_env import SO101GraspEnv
env = SO101GraspEnv(robot="hand", reward_type="sparse",
                    cube_grid=[(0.18,-0.05),(0.20,0.0),(0.22,0.05)])
obs, info = env.reset(seed=0, options={"cube_pos": (0.20, 0.0), "no_cube_yaw": True})
```

## 데이터 수집 결과물

`record_dataset.py` → `data/<name>/`
- `episode_000000.npz` … : `state[T,28] action[T,5] reward[T] done[T] success` (+`--images`시 카메라)
- `meta.json` : fps, control_hz, state/action 레이아웃, 성공률

LeRobotDataset(v2)로 변환하는 코드 스켈레톤은 `record_dataset.py` docstring 참고.

## Windows / Ubuntu

- 이 머신에 **WSL 미설치** (`wsl -l -v` → 미설치). 설치엔 관리자 권한 + 재부팅 필요.
- `sim/` 코드(MJCF · Gymnasium 환경 · 스크립트 데이터 수집 · 렌더링)는 **OS 무관** — Ubuntu에서
  그대로 실행됨. Windows에서 만드는 이유는 파일·venv가 여기 있고 MuJoCo가 Windows에서 잘 돌기 때문.
- **Ubuntu가 꼭 필요한 지점 = Phase 3**: LeRobotDataset 기록 + ACT 학습.
  `torch`/`lerobot` 은 Python 3.14 휠이 없고 lerobot 은 리눅스 우선.
- 권장: 관리자 PowerShell에서 `wsl --install` (재부팅) → Ubuntu + Python 3.11 env 세팅 후 `sim/`
  통째로 이동. 그 전까지는 환경 구축을 Windows에서 계속해도 무방 (코드가 포터블).

## 지금 상태 / 알려진 한계

- ✅ SO-101 공식 모델 로드, 씬·물리 안정, 3개+1(ext_cam) 카메라 정상
- ✅ IK 추종 오차 mm 수준, Gymnasium API·reset 변형·보상·렌더링 동작
- ✅ 스크립트 파지 성공률 (2026-09-11 튜닝 후, 고정좌표 (0.20,0.0) 20회 기준):
  **jaw 100% / jaw 랜덤박스+yaw 70% / hand 45%**. hand는 실패해도 "그냥 못 잡고 넘어감"
  으로 깨끗해짐 (예전엔 자기충돌로 팔이 폭주하는 구조적 버그였음). 큐브를 22→18mm로
  줄이고 손가락 끝에 **패드 박스 지오메트리**(`pad_fixed`/`pad_moving`)를 추가해야
  SO-101 메시 그리퍼로 안정적 파지가 됨 (원본 메시 콜리전은 힌지 근처에서 거의 붙어
  있어 유효 개폐 폭이 없음). 이 편집은 `so101_new_calib.xml`에 `[LOCAL EDIT]`로 표시.

**IK/물리 버그 3개 수정 (2026-09-11)**
1. 큐브 높이 상수가 `0.009`로 남아있었음 — 실제 `cube_geom` half-size는 `0.011`이라
   목표 높이가 테이블 속으로 파고들었음. `scripted_grasp.py`/`grasp_env.py` 둘 다 수정.
2. **hand 로봇 자기충돌**: identity/y90 접근자세로 낮은 높이를 잡으려 하면 `shoulder`
   바디와 `gripper`(손목) 바디가 부딪혀서 토크가 포화되고 하강이 멈춤. Y축 회전각을
   스윕해서 130°가 자기충돌 없이 도달 가능함을 확인 → `GRASP_QUAT0_HAND`로 반영.
3. **IK 반복횟수 폭주**: `_solve_ik` 반복을 너무 많이(24+) 돌리면 근특이점 근방에서
   damped-least-squares가 한 스텝 안에서 shoulder_pan을 엉뚱한 방향으로 크게 틀어버리는
   현상 발견. 로봇별로 스윕해서 jaw=18, hand=10 iters로 고정(`self._ik_iters`).

## SO-101 + AmazingHand 결합 모델 (`handarm`)

`build_combined.py` 가 MuJoCo MjSpec(attach) 으로:
- SO-101 공식 MJCF 에서 평행 그리퍼(moving_jaw + gripper 조인트/액추에이터) 제거
- 손목(gripper 바디)에 `SO-ARM_Interface.stl` 판 지오메트리(mm→m 스케일) + `hand_mount` 사이트
- pollen-robotics **공식** AmazingHand 오른손 MJCF 를 그 사이트에 attach (prefix `ah_`)
- 손끝 사이 `tcp` 사이트, `home` 키프레임 추가
결과: nq=80, nu=13 (팔5 + 손가락8), neq=20. 안정적으로 컴파일·시뮬됨.

**STL/스케일 확인 결과**
- `After_Class/Right_Hand.step`, `SO-ARM_Interface.step` 는 STL 이 아니라 **STEP(CAD)** 파일.
  MuJoCo 직접 사용 불가. `Right_Hand.step` 헤더 좌표 크기가 ~500mm 까지 나와서
  단품 손 치고는 큼 (Enhanced 버전이거나 어셈블리/여분 지오메트리 포함 가능성).
- **정확한 손 모델 = pollen-robotics 공식 MuJoCo 모델** (`robots/amazinghand/`, 이미 사용 중).
  손바닥 폭 ~110mm, 전체 ~150mm → 공개 스펙(4손가락·2관절·400g)과 일치.
- 연결부는 공식 `SO-ARM_Interface.stl` (45×49×6mm 판) 을 그대로 사용.
- ⚠️ 실물이 **"Amazing Hand Enhanced"** 라면 그 브랜치에서 모델을 다시 뽑아야 함:
  https://github.com/pollen-robotics/AmazingHand/tree/Amazing-Hand-Enhanced

**적용된 수정**
- ✅ 손 충돌 지오메트리 추가 (`ah_tip_col*` 4개 + `ah_palm_col`). 원본 AmazingHand MJCF는
  전부 visual-only(충돌 0개)라 물체를 통과했음.
- ✅ 손목 방향을 `MOUNT_EULER`(gripper 프레임 기준 XYZ deg)로 재파라미터화.
  기본 `(-125, 0, 0)` = 손가락 아래·손바닥 전방. `--sweep` 로 후보 비교 렌더 가능.

**추가 반영 (2026-09-11)**
- ✅ **wrist_roll(5번 관절) 가동범위 ±90°로 제한** (원본 -157°~+163°) — `so101_new_calib.xml`
  `[LOCAL EDIT]`. jaw/hand 모델 공통.
- ✅ **손바닥 방향 재계산**: gripper 바디의 world 회전을 구해 "손바닥(+x)→아래, 손가락(+z)→전방"
  이 되는 마운트 쿼터니언을 역산 (`MOUNT_QUAT`). 스윕 대신 계산으로 확정.
- ✅ **외부 카메라 스탠드 설치**: `cam_mount_bottom_v2.stl` + `cam_mount_top_v2.stl` — 사용자가
  SolidWorks 어셈블리에서 정확한 상대 위치로 다시 export 해준 버전 (초판은 두 부품을 별도
  origin으로 export해서 내가 PCA/추측으로 잘못 정렬했었음; v2는 어셈블리 좌표 그대로라
  추가 오프셋 계산 불필요, 두 메시를 그대로 한 body에 넣으면 됨). PCA로 폴의 긴 축이 메시
  로컬 +y 인 걸 확인 → x축 +90° 회전(피치) + 카메라 절곡부가 로봇을 보도록 180° 요(yaw)
  합성 쿼터니언으로 세움. 큐브에서 ~55cm, `ext_cam`으로 등록 (world 최상위에 둬야 함 —
  회전된 스탠드 body 안에 넣으면 부모의 roll을 상속해 수평선이 기울어짐),
  `mode="targetbody"`로 자동 조준. 두 씬(`so101_grasp.xml`, `so101_amazinghand_scene.xml`)
  모두에 있음.

**남은 튜닝** (뷰어에서, `build_combined.py` 상단):
- `MOUNT_EULER` / `MOUNT_POS` — 손 장착 자세·위치 미세조정 (팔 home 자세 기준)
- `GRIP_CLOSE` (8개 각도) — 확실한 핀치가 되도록
- `tcp.pos` (`build()` 안, ah 로컬) — 현재 파지중심이 옆으로 치우침, 손끝 중앙으로
- SO-101 그리퍼 뭉치의 고정 턱이 `wrist_roll_follower` 메시에 붙어 있어 시각적으로 조금 남음

## 다음 단계 (제안 순서)

1. **결합 모델 튜닝**: `view.py --model handarm --sim` 에서 `MOUNT_*` / `GRIP_*` 눈으로 맞추기
2. **hand 파지 환경**: `SO101GraspEnv` 에 `robot="hand"` 분기 추가 (그리퍼 액추에이터 대신
   8개 finger motor 를 하나의 grip 스칼라로 구동) → `handarm` 모델로 파지 데이터 수집
3. **파지 안정화**: `scripted_grasp.py` 파라미터 + 손목 yaw 정렬로 성공률 ↑
4. **텔레오퍼레이션 데이터 경로**: `view.py --teleop` 로그를 `record_dataset` 포맷으로 저장
5. **WSL2 이전**: LeRobotDataset 기록 + ACT 학습 + 실물↔시뮬 비율 스윕
