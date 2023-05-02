from pathlib import Path

import torch

from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class ModelSaver(SupervisedPlugin):
    def __init__(self, root: str = None):
        if not root:
            raise Exception("Please define a root to save models in")

        if not Path(root).is_dir():
            raise Exception(
                "Model saving path ill-defined. It does not exist.")

        self.root = root
        return

    def after_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        strategy_name = strategy.__class__.__name__
        model_name = strategy.model.__class__.__name__
        curr_exp = strategy.experience.current_experience

        save_name = [strategy_name, model_name, "Experience", str(curr_exp)]
        save_name = "_".join(save_name)

        PATH = self.root + save_name

        torch.save(strategy.model.state_dict(), PATH)
