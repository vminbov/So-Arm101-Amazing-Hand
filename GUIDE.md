# 팀원용: Claude와 함께 실행하는 순서

## 0. 준비물 (한 번만)
- Git, Python 3.11 이상, [Claude Code](https://claude.com/claude-code) (데스크톱 앱 Code 탭 또는 CLI)

## 1. 다운로드
터미널에서 아래를 실행합니다. 폴더는 원하는 위치에 만들면 됩니다.

```bash
git clone https://github.com/vminbov/So-Arm101-Amazing-Hand.git
```

Claude Code로 **그 폴더(`So-Arm101-Amazing-Hand`)를 작업 폴더로 열어서** 시작합니다.

## 2. Claude에게 순서대로 말하기

**① 환경 파악**
> README.md와 sim/README.md 읽고 이 프로젝트가 뭔지 3줄로 설명해줘.

**② 설치**
> 이 폴더에 .venv 만들고 sim/requirements-sim.txt 설치해줘. OS는 (Windows / Mac / Ubuntu)야.

**③ 동작 확인 (창 없이)**
> 스크립트 파지를 hand 로봇으로 3 에피소드 돌려서 성공률 알려줘.

성공하면 `2/3 = ...%` 같은 결과가 나옵니다. 성공률이 100%가 아니어도 정상입니다. 시드에 따라 달라질 수 있습니다.

**④ 눈으로 보기 (GUI 필요)**
> SO-101 + AmazingHand 결합 모델 뷰어를 스크립트 파지 재생 모드로 띄워줘.

**⑤ 데이터 수집**
> hand 로봇으로 sim_grasp_v0 이름으로 10 에피소드 데이터 수집해줘. data 폴더에 뭐가 생기는지도 설명해줘.

## 3. 자주 쓰는 요청
| 하고 싶은 것 | Claude에게 |
|---|---|
| 캔 위치 바꾸기 | "grasp_env.py에서 hand 캔 배치 범위가 어디 정의돼 있는지 알려주고 바꾸는 법 설명해줘" |
| 손 장착 자세 수정 | "build_combined.py의 MOUNT_* 값 설명해줘" (수정 후 `python -m sim.robots.build_combined` 재실행) |
| 성공률 측정 | "hand 로봇 스크립트 파지 30 에피소드 성공률 측정해줘" |
| 영상 저장 | "scripted_grasp 옵션 보고 파지 영상을 mp4로 저장해줘" |

## 4. 문제가 생기면
Claude에게 **에러 메시지 전체를 붙여넣고** 이렇게 말하세요.
> 이 에러 원인 찾아서 고쳐줘. 원인 먼저 설명하고, 코드 수정 전에 알려줘.

**주의**
- `git push`는 팀 저장소를 바꾸는 동작입니다. Claude가 "올릴까요?" 하고 물으면 내용을 확인하고 답하세요.
- 코드를 수정했다면 push 전에 `scripted_grasp`를 한 번 돌려서 성공률이 유지되는지 확인하세요.
- 학습(ACT)과 LeRobotDataset 변환은 아직 안 했습니다. WSL2/Ubuntu와 Python 3.11 환경이 필요합니다.
