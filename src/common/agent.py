from abc import ABC, abstractmethod

import torch


class BaseAgent(ABC):
    """Common interface of the value-based agents (DDQN, Rainbow, DVN).

    Subclasses must define `self.policy_net`, `self.target_net`,
    `self.optimizer` and `self.device`.
    """

    @abstractmethod
    def select_action(self, state, epsilon) -> int:
        pass

    @abstractmethod
    def update_model(self):
        pass

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_model(self, path, load_optimizer: bool = True):
        # map_location: checkpoints trained on GPU must load on a CPU-only machine.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
