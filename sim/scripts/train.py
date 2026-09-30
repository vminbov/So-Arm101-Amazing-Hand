"""Windows용 lerobot-train 래퍼 (.venv-train에서 실행, 인자는 lerobot-train과 동일).

    .venv-train/Scripts/python.exe -m sim.scripts.train --dataset.repo_id=local/<name> \
        --dataset.root=data/lerobot/<name> --policy.type=act --policy.device=cuda ...

lerobot은 체크포인트마다 checkpoints/last 심볼릭 링크를 만드는데, Windows는 관리자/개발자 모드가
아니면 심볼릭 링크 생성이 막혀(WinError 1314) 학습이 첫 저장에서 죽는다. 권한이 필요 없는
디렉터리 정션으로 대신 만든다(재개 시 last/ 경로를 그대로 따라감).
"""
import sys
import _winapi
from pathlib import Path

from lerobot.common import train_utils
from lerobot.scripts import lerobot_train


def _update_last_checkpoint(checkpoint_dir: Path) -> None:
    last = checkpoint_dir.parent / train_utils.LAST_CHECKPOINT_LINK
    if last.is_junction() or last.is_symlink():
        last.unlink()
    _winapi.CreateJunction(str(checkpoint_dir.resolve()), str(last.resolve()))


if sys.platform == "win32":
    lerobot_train.update_last_checkpoint = _update_last_checkpoint

if __name__ == "__main__":
    lerobot_train.main()
