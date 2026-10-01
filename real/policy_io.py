"""학습된 정책을 실물 입력으로 돌리는 부분 (step3 / step5 공용).

정책 입력 = 실물 카메라(320x240) + state 6개. state 중 팔 4개는 실물 측정값, wrist_roll·grip 은
'정책이 직전에 지시한 값'(가상)을 넣는다 — 실물은 손목을 못 돌리고 손가락도 안 쓰지만, 정책이
자기가 지시한 대로 됐다고 믿게 해서 학습 때 본 상태와 어긋나지 않게 하려는 것.
"""
import numpy as np
import torch

from common import ROOT, deg_to_sim, sim_to_deg

DEFAULT_POLICY = ROOT / "outputs" / "act_sim_v3" / "checkpoints" / "last" / "pretrained_model"


class RealPolicy:
    def __init__(self, path=None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.common.control_utils import predict_action
        from sim.scripts.scripted_grasp import ScriptedGraspPolicy
        path = str(path or DEFAULT_POLICY)
        cfg = PreTrainedConfig.from_pretrained(path)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = get_policy_class(cfg.type).from_pretrained(path).to(self.dev).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=path)
        self._predict = predict_action
        self.virt0 = np.array([ScriptedGraspPolicy.Q_UNFOLD_TARGET[4], 0.0])   # 시연에서 팔 펴기 직후 값
        self.reset()
        print(f"정책 로드: {path} ({self.dev})")

    def reset(self):
        self.policy.reset(); self.pre.reset(); self.post.reset()
        self.virt = self.virt0.copy()

    def act(self, q_real_deg, image_320x240):
        """-> (팔 4관절 목표[도, 실물 기준], 정책 원출력 6개[시뮬 rad/grip])"""
        st = np.zeros(6, np.float32)
        st[:4] = deg_to_sim(q_real_deg)[:4]
        st[4:] = self.virt
        a = self._predict({"observation.state": st, "observation.images.ext_cam": image_320x240},
                          self.policy, self.dev, self.pre, self.post, use_amp=False,
                          task="grasp the can", robot_type="so101_amazinghand")
        a = (a.squeeze(0) if a.dim() > 1 else a).cpu().numpy()
        self.virt = np.array([a[4], np.clip(a[5], 0, 1)])
        target_deg = sim_to_deg(np.append(a[:4], 0.0))[:4]
        return target_deg, a
