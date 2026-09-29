# SO-Arm101 + AmazingHand

SO-101 로봇팔에 [AmazingHand](https://github.com/pollen-robotics/AmazingHand)(오른손)를 결합해 캔을 잡는
MuJoCo 파지 시뮬레이션과 관련 CAD 파일 모음.
목표는 실물 teleop 데이터와 섞어 ACT를 학습할 **시뮬레이션 데모 데이터**를 자동 생성하는 것(A안).

<p align="center"><img src="media/hand_side_grasp_thumbup_frames.png" width="720"></p>

- 데모 영상: [`media/hand_side_grasp_thumbup.mp4`](media/hand_side_grasp_thumbup.mp4)

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `sim/` | MuJoCo 모델(MJCF), Gymnasium 환경, 스크립트 파지 전문가, 데이터 수집 코드. 자세한 설명은 [`sim/README.md`](sim/README.md) |
| `project_cad_file/` | SolidWorks 원본 및 STEP 파일 (손, 팔, 인터페이스, 카메라 스탠드) |
| `media/` | 파지 데모 영상과 프레임 |
| `so_arm101.urdf`, `model.xml` | 팔 URDF / MJCF |
| `*.STL`, `*.step`, `*.SLDASM` | 카메라 마운트, 손목 인터페이스, 오른손 CAD |

## 설치

Python 3.11 이상 권장 (개발은 Windows / Python 3.14).

```bash
python -m venv .venv
.venv/Scripts/pip install -r sim/requirements-sim.txt    # Linux/Mac: .venv/bin/pip
```

## 실행

저장소 루트에서 실행합니다. 아래 `python`은 위에서 만든 `.venv`의 것입니다.

```bash
python -m sim.scripts.view --model handarm --sim        # 결합 모델 뷰어
python -m sim.scripts.scripted_grasp --robot hand --episodes 20   # 스크립트 파지 성공률 측정
python -m sim.scripts.record_dataset --robot hand --name sim_grasp_v0 --episodes 100   # 데이터 수집 → data/
python -m sim.robots.build_combined                     # 결합 모델 재생성 (MOUNT_* 값 수정 후)
```

MJCF의 메시 경로는 상대 경로라 어느 위치에 clone해도 바로 로드됩니다.

## 현재 상태 (2026-09-29)

- 파지 방식: 사람처럼 손바닥을 캔 옆면에 대고 손가락으로 감싸는 **옆면 파지**, 엄지가 위쪽.
- 스크립트 전문가로 `hand` 로봇 **60/60 성공** (고정 외부 카메라 기준 캔 배치 아크 r 0.37–0.43 m, ±20–40°).
- 관절 목표 `[q_arm(5), grip]` 형태의 액션과 `obs_state`(6D)를 저장 → LeRobot SO-101 액션 공간과 호환.
- 아직 안 한 것: 학습(ACT), LeRobotDataset 변환(WSL2/Ubuntu 필요), 실물 카메라와의 화각·왜곡 정합.

> `sim/README.md`의 일부 수치(성공률 등)는 초기 개발 시점 기준이라 위 내용이 더 최신입니다.

## 크레딧

- SO-101: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- AmazingHand MJCF: [pollen-robotics/AmazingHand](https://github.com/pollen-robotics/AmazingHand)
