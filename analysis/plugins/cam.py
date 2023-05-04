from pathlib import Path

import torch

from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class CAMPlugin(SupervisedPlugin):
    def __init__(self, model, layers, size):
        pass
