import copy
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import argparse

from analysis.args import (
    collect_activations_args,
    MODEL_DIR,
    ALGOS,
    LAYERS,
    LOAD_EXPERIENCES,
    NUM_EXPERIENCES_TRAINED_ON,
    TRUNCATE
)
from utils import (
    generate_experiences,
    adapt_model,
    model_loader,
    strategy_builder,
    StrategiesEnum,
    DataStreamEnum
)

from training.args import lwf_args, mas_args, naive_args
from training.models.models import MultiHeadVGGSmall

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.benchmarks import SplitTinyImageNet

# initialization
ACTIVATIONS_DATA = {}
SHAPE_DATA = {}
EVALS = len(ALGOS)*len(LOAD_EXPERIENCES)


def initialize_dict():
    global ACTIVATIONS_DATA
    global SHAPE_DATA
    ACTIVATIONS_DATA = {"lwf": {}, "mas": {}, "naive": {}}

    for algo in ALGOS:
        ACTIVATIONS_DATA[algo] = {}
        for exp in LOAD_EXPERIENCES:
            ACTIVATIONS_DATA[algo][str(exp)] = {}
            for layer in LAYERS:
                ACTIVATIONS_DATA[algo][str(exp)][layer] = {}

    SHAPE_DATA = copy.deepcopy(ACTIVATIONS_DATA)


def main():
    # Base model
    base_model = MultiHeadVGGSmall(n_classes=200)

    # Split benchmark
    benchmark = SplitTinyImageNet(
        NUM_EXPERIENCES_TRAINED_ON, return_task_id=True, dataset_root=None)

    # Generate list of experiences

    if (DataStreamEnum[args.split] == DataStreamEnum.train):
        stream = benchmark.train_stream
    elif (DataStreamEnum[args.split] == DataStreamEnum.test):
        stream = benchmark.test_stream

    experiences = generate_experiences(
        stream=stream, n_experiences=benchmark.n_experiences)

    # Directory where all models are saved
    lwf_dir = Path(f"./{MODEL_DIR}/lwf_150_saved_models")
    mas_dir = Path(f"./{MODEL_DIR}/mas_150_saved_models")
    naive_dir = Path(f"./{MODEL_DIR}/naive_150_saved_models")

    # Torch/Avalanche necessities
    criterion = CrossEntropyLoss()
    # interactive_logger = InteractiveLogger()
    evaluation_plugin = EvaluationPlugin(
        # accuracy_metrics(epoch=True, experience=True, stream=True),
        # loss_metrics(epoch=True, experience=True, stream=True),
        # forgetting_metrics(experience=True, stream=True),
        # loggers=[interactive_logger],
    )

    print(f"BUILDING EVALUATION FOR EXPERIENCE {EVALUATE_EXPERIENCE}")

    # Iterate through all experiences to load
    for load_experience in tqdm(LOAD_EXPERIENCES):

        # Adapt base model with experiences
        if (load_experience > 0):
            adapt_model(base_model, experiences)

        # Load trained model paths
        pattern = f"*{load_experience}"
        lwf_path = list(lwf_dir.glob(pattern))[0]
        mas_path = list(mas_dir.glob(pattern))[0]
        naive_path = list(naive_dir.glob(pattern))[0]

        # Instantiate trained models
        lwf_model = model_loader(base_model=base_model, path=lwf_path)
        mas_model = model_loader(base_model=base_model, path=mas_path)
        naive_model = model_loader(base_model=base_model, path=naive_path)

        # Instantiate strategies
        lwf_strategy = strategy_builder(
            strat_enum=StrategiesEnum.lwf,
            model=lwf_model,
            criterion=criterion,
            evaluation_plugin=evaluation_plugin,
            strategy_args=lwf_args,
            layers=LAYERS,
            size=TRUNCATE,
        )

        mas_strategy = strategy_builder(
            strat_enum=StrategiesEnum.mas,
            model=mas_model,
            criterion=criterion,
            evaluation_plugin=evaluation_plugin,
            strategy_args=mas_args,
            layers=LAYERS,
            size=TRUNCATE,
        )

        naive_strategy = strategy_builder(
            strat_enum=StrategiesEnum.naive,
            model=naive_model,
            criterion=criterion,
            evaluation_plugin=evaluation_plugin,
            strategy_args=naive_args,
            layers=LAYERS,
            size=TRUNCATE,
        )

        # Run evaluation
        print(
            f"EVALUATING MODELS TRAINED ON EXPERIENCE {load_experience}")

        lwf_strategy.eval(experiences[EVALUATE_EXPERIENCE])
        mas_strategy.eval(experiences[EVALUATE_EXPERIENCE])
        naive_strategy.eval(experiences[EVALUATE_EXPERIENCE])

        for layer in LAYERS:

            lwf_acts = lwf_strategy.activations[layer].cpu().numpy()
            mas_acts = mas_strategy.activations[layer].cpu().numpy()
            naive_acts = naive_strategy.activations[layer].cpu().numpy()

            ACTIVATIONS_DATA['lwf'][str(load_experience)][layer] = lwf_acts
            ACTIVATIONS_DATA['mas'][str(load_experience)][layer] = mas_acts
            ACTIVATIONS_DATA['naive'][str(load_experience)][layer] = naive_acts

            SHAPE_DATA['lwf'][str(load_experience)][layer] = lwf_acts.shape
            SHAPE_DATA['mas'][str(load_experience)][layer] = mas_acts.shape
            SHAPE_DATA['naive'][str(load_experience)][layer] = naive_acts.shape

        del lwf_strategy, mas_strategy, naive_strategy, lwf_model, mas_model, naive_model

    # Pickle data
    with open(f'{SAVE_DIR}/act_on_exp_{EVALUATE_EXPERIENCE}.pickle', 'wb') as f:
        pickle.dump(ACTIVATIONS_DATA, f)

    with open(f'{SAVE_DIR}/act_on_exp_{EVALUATE_EXPERIENCE}.json', 'w') as f:
        json.dump(SHAPE_DATA, f)


if __name__ == "__main__":
    args = collect_activations_args()
    EVALUATE_EXPERIENCE = args.experience
    SAVE_DIR = DataStreamEnum[args.split].value
    initialize_dict()
    main()
