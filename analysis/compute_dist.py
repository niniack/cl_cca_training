import pickle
import os
import json
import copy
import enum
from pathlib import Path
import itertools
import multiprocessing
from typing import Literal, Tuple, Optional, List

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from netrep.metrics import LinearMetric
from netrep.conv_layers import convolve_metric

from utils import (
    DataStreamEnum,
    SENTINEL,
    set_conv_dict,
    get_conv_dict
)
from analysis.args import (
    compute_dist_args,
    LAYERS,
    LOAD_EXPERIENCES,
    TRUNCATE
)

HEATMAP_LWF_NAIVE_DATA = {}
HEATMAP_MAS_NAIVE_DATA = {}
HEATMAP_LWF_MAS_DATA = {}
RAW_CONV_DATA = {}
os.environ['OMP_NUM_THREADS'] = '1'


class DistanceCalculator(LinearMetric):

    def __init__(self, alpha: float = 1, center_columns: bool = True, zero_pad: bool = True, score_method: Literal['angular', 'euclidean'] = "angular"):
        super().__init__(alpha, center_columns, zero_pad, score_method)

    def _listener(self, queue, total_steps):
        pbar = tqdm(total=total_steps)
        for item in iter(queue.get, None):
            pbar.update()

    def _compute_distance(self, i, j, X, Y, queue):
        """Computes the distance between X and Y using the convolution method

        Args:
            i (_type_): i index to store in the heatmap, corresponds to experience
            j (_type_): j index to store in the heatmap, corresponds to experience
            X (_type_): activations from a CL method
            Y (_type_): activations from a CL method
            queue (_type_): Manager queue to update for pbar update

        Returns:
            _type_: i,j, mean of the distance, raw convolution distances
        """
        n, c, h, w = X.shape
        Xf = X.reshape(-1, c)
        Yf = Y.reshape(-1, c)

        self.fit(Xf, Yf)
        dist_min = self.score(Xf, Yf)
        queue.put(SENTINEL)

        return i, j, dist_min

    def _compute_distance_conv(self, i, j, X, Y, queue):
        """Computes the distance between X and Y using the convolution method

        Args:
            i (_type_): i index to store in the heatmap, corresponds to experience
            j (_type_): j index to store in the heatmap, corresponds to experience
            X (_type_): activations from a CL method
            Y (_type_): activations from a CL method
            queue (_type_): Manager queue to update for pbar update

        Returns:
            _type_: i,j, mean of the distance, raw convolution distances
        """
        n, c, h, w = X.shape
        Yf = Y.reshape(-1, c)

        dists = np.full((h, w), -1.0)
        for h_idx, w_idx in itertools.product(range(h), range(w)):

            # Apply shift to X tensor, then flatten.
            shifts = (h_idx - (h // 2), w_idx - (w // 2))
            Xf = np.roll(X, shifts, axis=(1, 2)).reshape(-1, c)

            # Fit and evaluate metric.
            self.fit(Xf, Yf)
            dists[h_idx, w_idx] = self.score(Xf, Yf)
            queue.put(SENTINEL)

        dist_mean = np.mean(dists)
        dist_min = np.min(dists)
        return i, j, dist_min, dist_mean, dists

    def _compute_distance_star(self, args):
        """Helper function for multiprocessing.
        Using this allows us to use tqdm to track progress via imap_unordered.
        """
        return self._compute_distance(*args)

    def _compute_distance_conv_star(self, args):
        """Helper function for multiprocessing.
        Using this allows us to use tqdm to track progress via imap_unordered.
        """
        return self._compute_distance_conv(*args)

    def pairwise_distances(
            self,
            n1_data: List[npt.NDArray],
            n2_data: List[npt.NDArray],
            processes: Optional[int] = None,
            verbose: bool = True,
            conv: bool = False,
    ):
        """Computes pairwise distances between all pairs of networks w/ multiprocessing.

        We suggest setting "OMP_NUM_THREADS=1" in your environment variables to avoid oversubscription 
        (multiprocesses competing for the same CPU).

        Parameters
        ----------
        n1_data:  List[npt.NDArray]

        n2_data: List[npt.NDArray], optional

        enable_caching: bool
            Whether to cache pre-transformed data.
        processes: int, optional
            Number of processes to use. If None, defaults to number of CPUs.
        verbose: bool, optional
            Whether to display progress bar.

        Returns
        -------
        D_train: npt.NDArray
            n_networks x n_networks distance matrix.
        """
        shape = n1_data[0].shape
        n_networks = len(n1_data)
        n_dists = n_networks*(n_networks-1)//2 + n_networks

        if (conv):
            total_steps = shape[-1]*shape[-2]*n_dists
        else:
            total_steps = n_dists

        manager = multiprocessing.Manager()
        queue = manager.Queue()

        # create generator of args for multiprocessing
        ij = itertools.combinations_with_replacement(range(n_networks), 2)
        args = ((i, j, n1_data[i], n2_data[j], queue)
                for i, j in ij)

        if verbose:
            print(
                f"Parallelizing {n_dists} distance calculations with {multiprocessing.cpu_count() if processes is None else processes} processes.")

        listener_proc = multiprocessing.Process(
            target=self._listener, args=(queue, total_steps,))
        listener_proc.start()

        with multiprocessing.Pool(processes=processes) as pool:
            results = []
            for result in pool.imap_unordered(self._compute_distance_conv_star if conv else self._compute_distance_star, args):
                results.append(result)

        queue.put(None)
        listener_proc.join()

        D_min = np.zeros((n_networks, n_networks))
        D_mean = np.zeros((n_networks, n_networks))
        D_conv = {}

        if conv:
            for i, j, dist_min, dist_mean, dist_conv in results:
                D_min[i, j], D_min[j, i] = dist_min, dist_min
                D_mean[i, j], D_mean[j, i] = dist_mean, dist_mean
                set_conv_dict(i, j, dist_conv, D_conv)
        else:
            for i, j, dist_min in results:
                D_min[i, j], D_mean[j, i] = dist_min, dist_min

        return D_min


def initialize_dict():
    global HEATMAP_LWF_NAIVE_DATA
    global HEATMAP_MAS_NAIVE_DATA
    global HEATMAP_LWF_MAS_DATA
    global RAW_CONV_DATA

    for layer in LAYERS:
        HEATMAP_LWF_MAS_DATA[layer] = {}

    RAW_CONV_DATA = copy.deepcopy(HEATMAP_LWF_MAS_DATA)
    HEATMAP_LWF_NAIVE_DATA = copy.deepcopy(HEATMAP_LWF_MAS_DATA)
    HEATMAP_MAS_NAIVE_DATA = copy.deepcopy(HEATMAP_LWF_MAS_DATA)


def main(args):

    # Build metric (procrustes)
    # metric = LinearMetric(
    #     alpha=1,
    #     center_columns=True,
    #     score_method="angular",
    # )

    calculator = DistanceCalculator(alpha=ALPHA)

    # Iterate through directory
    for exp_id in EVALUATE_EXPERIENCES:

        # Read the data from the pickle file
        with open(f'{DIR}/act_on_exp_{exp_id}.pickle', 'rb') as f:
            activations_dict = pickle.load(f)

        # Each heatmap is for a single layer
        for layer in LAYERS:

            # Build the input for pairwise calculator
            LWF_LAYER_ACTS = []
            MAS_LAYER_ACTS = []
            NAIVE_LAYER_ACTS = []

            # Pairwise for all models
            for model in LOAD_EXPERIENCES:

                # Flat reshape (n*chw)
                lwf_act = activations_dict['lwf'][str(
                    model)][layer]

                mas_act = activations_dict['mas'][str(
                    model)][layer]

                naive_act = activations_dict['naive'][str(
                    model)][layer]

                LWF_LAYER_ACTS.append(lwf_act[:TRUNCATE])
                MAS_LAYER_ACTS.append(mas_act[:TRUNCATE])
                NAIVE_LAYER_ACTS.append(naive_act[:TRUNCATE])

            lwf_mas_dist = calculator.pairwise_distances(
                LWF_LAYER_ACTS, MAS_LAYER_ACTS, conv=False)

            lwf_naive_dist = calculator.pairwise_distances(
                LWF_LAYER_ACTS, NAIVE_LAYER_ACTS, conv=False)

            mas_naive_dist = calculator.pairwise_distances(
                MAS_LAYER_ACTS, NAIVE_LAYER_ACTS, conv=False)

            HEATMAP_LWF_MAS_DATA[layer] = lwf_mas_dist.tolist()
            HEATMAP_LWF_NAIVE_DATA[layer] = lwf_naive_dist.tolist()
            HEATMAP_MAS_NAIVE_DATA[layer] = mas_naive_dist.tolist()

        # Dump data
        with open(f'{DIR}/lwf_mas_alpha_{ALPHA}_heatmap_on_exp_{exp_id}.json', 'w') as f:
            json.dump(HEATMAP_LWF_MAS_DATA, f)

        with open(f'{DIR}/lwf_naive_alpha_{ALPHA}_heatmap_on_exp_{exp_id}.json', 'w') as f:
            json.dump(HEATMAP_LWF_NAIVE_DATA, f)

        with open(f'{DIR}/mas_naive_alpha_{ALPHA}_heatmap_on_exp_{exp_id}.json', 'w') as f:
            json.dump(HEATMAP_MAS_NAIVE_DATA, f)

        # with open(f'{DIR}/raw_conv_on_exp_{exp_id}.json', 'w') as f:
        #     json.dump(RAW_CONV_DATA, f)


if __name__ == "__main__":
    args = compute_dist_args()
    ALPHA = args.alpha
    DIR = DataStreamEnum[args.split].value
    EVALUATE_EXPERIENCES = [args.experience]

    # DEBUG
    # LAYERS = [
    #     "vgg.features.13"
    # ]
    # LOAD_EXPERIENCES = list(range(1))

    initialize_dict()
    main(args)
