from copy import deepcopy
from enum import Enum
from functools import partial
import inspect
import torch
import numpy as np
from torch.optim import SGD
import avalanche
from tqdm import tqdm
from training.plugins.activations import ActivationSaver

# Dummy value added to a queue for progress bar update
SENTINEL = 1


def generate_experiences(stream, n_experiences):
    """Generates a list of experiences

    Args:
        stream (_type_): Avalanche stream
        n_experiences (_type_): number of experiences to generate

    Returns:
        list: list of experiences
    """
    it = iter(stream)
    return [next(it) for i in range(n_experiences)]


def adapt_model(model, experiences):
    """Adapts the multi task model to get more classifier heads

    Args:
        model (_type_): Base model to adapt
        experiences (_type_): The experiences to adapt the model with
    """
    for exp in experiences:
        model.classifier.adaptation(exp)


def model_loader(base_model, path):
    """Loads a model with the given weights

    Args:
        base_model (_type_): base model to adapt
        path (_type_): path to weights

    Returns:
        _type_: a deepcopy of the base model with the loaded weights
    """
    copied = deepcopy(base_model)
    copied.load_state_dict(torch.load(path), strict=True)
    copied.eval()

    return copied


class DataStreamEnum(Enum):
    """Given the data type, returns the directory for storage and reading

    Args:
        Enum (_type_): test or train
    """
    train = "train_pickles"
    test = "test_pickles"


class StrategiesEnum(Enum):
    """Selects the strategy to build

    Args:
        Enum (_type_): lwf or mas

    Returns:
        _type_: Returns a reference to the strategy
    """
    lwf = avalanche.training.LwF
    mas = avalanche.training.MAS
    naive = avalanche.training.Naive

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

    def filter_args(self, strategy_args):
        signature = inspect.signature(self.value).parameters
        filtered_args = {k: v for k,
                         v in strategy_args.__dict__.items() if k in signature}
        return filtered_args


def strategy_builder(strat_enum,
                     model,
                     criterion,
                     evaluation_plugin,
                     strategy_args,
                     layers,
                     size=1000):
    """Builds a strategy from Avalanche

    Args:
        strat_enum (StrategiesEnum): see StrategiesEnum
        model (_type_): Model with loaded weights
        criterion (_type_): Torch criterion
        evaluation_plugin (_type_): Avalanche evaluation plugin
        strategy_args (_type_): Arguments for the strategy
        layers (_type_): Which layers to store activations from
        size (int, optional): What fraction of the activations. Defaults to 1.

    Returns:
        _type_: Returns an Avalanche strategy
    """
    activation_saver_plugin = ActivationSaver(
        model=model, layers=layers, size=size)

    device = torch.device(
        f"cuda:{strategy_args.cuda}" if torch.cuda.is_available() else "cpu")

    optimizer = SGD(model.parameters(),
                    lr=strategy_args.learning_rate,
                    momentum=0.9)

    filtered_args = strat_enum.filter_args(strategy_args)

    return strat_enum(model,
                      optimizer,
                      criterion,
                      eval_mb_size=1000,
                      device=device,
                      evaluator=evaluation_plugin,
                      plugins=[activation_saver_plugin],
                      **filtered_args)


def set_conv_dict(i, j, conv_dist, dict):
    """Specifically to write the raw convolution data. Ordering of i,j doesn't matter

    Args:
        i (_type_): i index
        j (_type_): j index
        conv_dist (_type_): the convolution distances from netrep
        dict (_type_): the dictionary to store them
    """
    key = f"({min(i, j)}-{max(i, j)})"
    dict[key] = conv_dist.tolist()


def get_conv_dict(i, j, dict):
    """Specifically to read the raw convolution data. Ordering of i,j doesn't matter

    Args:
        i (_type_): i index
        j (_type_): j index
        dict (_type_): the dictionary to read from. probably from a json file

    Returns:
        _type_: the array of raw convolution distances
    """
    key = f"({min(i, j)}-{max(i, j)})"
    return dict.get(key, None)
