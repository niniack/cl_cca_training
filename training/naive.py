import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

import avalanche as avl
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics
)
from training.utils import set_seed, create_default_args
from training.models.models import MultiHeadVGGSmall
from args import naive_args as args
from plugins.model_saver import ModelSaver

from avalanche.training.plugins import EvaluationPlugin


def naive_stinyimagenet(override_args=None):

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitTinyImageNet(
        10, return_task_id=True, dataset_root=args.dataset_root)
    model = MultiHeadVGGSmall(n_classes=200)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()
    text_logger = avl.logging.TextLogger(
        open(args.save_folder+"naive_log.txt", "a"))

    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(
            epoch=True, experience=True, stream=True
        ),
        loss_metrics(
            epoch=True, experience=True, stream=True
        ),
        forgetting_metrics(
            experience=True, stream=True
        ),
        loggers=[interactive_logger, text_logger])

    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    saver_plugin = ModelSaver(root=args.save_folder)

    # cl_strategy = avl.training.LwF(
    #     model,
    #     optimizer,
    #     criterion,
    #     alpha=args.lwf_alpha, temperature=args.lwf_temperature,
    #     train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
    #     device=device, evaluator=evaluation_plugin, plugins=[saver_plugin])

    cl_strategy = avl.training.Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.train_epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin, plugins=[saver_plugin]
    )

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == "__main__":
    res = naive_stinyimagenet()
    print(res)
