from pathlib import Path

import torch

from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class ActivationSaver(SupervisedPlugin):
    def __init__(self, model, layers, size):
        self.layers = layers
        self.model = model
        self.size = size
        self.add_hooks(model, layers)
        self.latest_activations = {}
        self.activations = {}
        return

    def get_name_to_module(self, model):
        name_to_module = {}
        for m in model.named_modules():
            name_to_module[m[0]] = m[1]
        return name_to_module

    def get_activation(self, layer):
        def hook(model, input, output):
            self.latest_activations[layer] = output.cpu().detach()
        return hook

    def add_hooks(self, model, layers):
        name_to_module = self.get_name_to_module(model)
        for layer in layers:
            name_to_module[layer].register_forward_hook(
                self.get_activation(layer)
            )

    def after_eval_forward(self, strategy: "SupervisedTemplate", *args, **kwargs):
        # Iterate through all layers in the latest activation
        for k, v in self.latest_activations.items():
            # If key already exists
            if (k in self.activations):
                previous_activations = self.activations[k]
                # Append latest activation to dict
                self.activations[k] = torch.cat(
                    (previous_activations, v), dim=0)
            # key does not exist
            else:
                self.activations[k] = v

    def after_eval_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        # strategy.activations = self.activations
        strategy.activations = {k: v[:self.size]
                                for k, v in self.activations.items()}
