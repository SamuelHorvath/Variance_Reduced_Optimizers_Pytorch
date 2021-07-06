import argparse
from datetime import datetime
import os


def parse_args(args):
    parser = initialise_arg_parser(args, 'Variance Reduction.')

    parser.add_argument(
        "--total-runs",
        type=int,
        default=3,
        help="Number of times to redo run, we increase seed by 1 if deterministic",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to run",
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Initial learning rate (default: .1)'
    )
    parser.add_argument(
        "--tune-lr",
        default=False,
        action='store_true',
        help="Whether to tune step size during optimization procedure, based on single run"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Static batch size for computation, for speed select as large as possible"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Define which method to run"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "mnist", "cifar10", "cifar100", "mushrooms", "w8a", "ijcnn1", "a9a", "phishing"],
        help="Define which dataset to load"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='top_1_acc',
        choices=["loss", "top_1_acc"],
        help="Define which metric to optimize."
    )
    parser.add_argument(
        "--train-metric",
        default=False,
        action='store_true',
        help="Whether to tune train or validation metric"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Define which model to load"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default='CE',
        choices=['CE', 'BCE'],
        help="Define which model to load"
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.,
        help='Weight decay (default: 0.)'
    )
    parser.add_argument(
        "--track-grad-norm",
        default=False,
        action='store_true',
        help="Whether to track grad norm on validation set"
    )
    parser.add_argument(
        "--nc-regularizer",
        default=False,
        action='store_true',
        help="Whether to include non-convex regularizer"
    )
    parser.add_argument(
        "--nc-regularizer-value",
        type=float,
        default=1e-3,
        help="Non-convex regularizer coefficient"
    )

    # SETUP ARGUMENTS
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default='../check_points',
        help="Directory to persist run meta data_preprocess, e.g. best/last models."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/",
        help="Base root directory for the dataset."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Define on which GPU to run the model (comma-separated for multiple). If -1, use CPU."
    )
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        default=4,
        help="Num workers for dataset loading"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Run deterministically for reproducibility."
    )
    parser.add_argument(
        "--manual-seed",
        type=int,
        default=123,
        help="Random seed to use."
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="How often to do validation."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Name of the Experiment (no default)"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        default="INFO"
    )
    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    os.makedirs("../logs/", exist_ok=True)
    parser.add_argument(
        "--logfile",
        type=str,
        default=f"../logs/log_{now}.txt"
    )

    # Evaluation mode, do not run training
    parser.add_argument("--evaluate", action='store_true', default=False, help="Evaluation or Training mode")

    args = parser.parse_args()
    transform_gpu_args(args)

    return args


def initialise_arg_parser(args, description):
    parser = argparse.ArgumentParser(args, description=description)
    return parser


def transform_gpu_args(args):
    if args.gpu == "-1":
        args.gpu = "cpu"
    else:
        gpu_str_arg = args.gpu.split(',')
        if len(gpu_str_arg) > 1:
            args.gpu = sorted([int(card) for card in gpu_str_arg])
        else:
            args.gpu = f"cuda:{args.gpu}"
