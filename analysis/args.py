import argparse

MODEL_DIR = "saved_models"
ALGOS = ["lwf", "mas", "naive"]
LAYERS = [
    "vgg.features.3",
    "vgg.features.6",
    "vgg.features.8",
    "vgg.features.11",
    "vgg.features.13"
]
LOAD_EXPERIENCES = list(range(10))
NUM_EXPERIENCES_TRAINED_ON = 10
TRUNCATE = 1000


def base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',
                        required=True,
                        type=str,
                        choices=['train', 'test'],
                        help='Train or test data to use')
    parser.add_argument('--experience',
                        required=True,
                        type=int,
                        help='The experience to evaluate on')
    return parser


def collect_activations_args():
    parser = base_args()
    return parser.parse_args()


def compute_dist_args():
    parser = base_args()
    parser.add_argument('--alpha',
                        required=True,
                        type=float,
                        help='The experience to evaluate on')
    return parser.parse_args()
