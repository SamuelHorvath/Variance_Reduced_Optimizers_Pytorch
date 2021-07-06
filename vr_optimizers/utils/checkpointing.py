import os
import shutil
import json

from utils.logger import Logger
from utils.utils import create_model_dir, get_key


def save_checkpoint(args, is_best, metrics, metric_to_optim):
    """
    Persist checkpoint to disk
    :param args: training setup
    :param is_best: Whether model with best metric
    :param metrics: metrics obtained from evaluation
    :param metric_to_optim: metric to optimize, e.g. top 1 accuracy
    """

    key = get_key(train=False)
    result_text = f"avg_loss={metrics[key + 'loss']}," \
                  f" avg_{metric_to_optim}={metrics[key + metric_to_optim]}"

    model_dir = create_model_dir(args)
    result_filename = os.path.join(model_dir, 'results.txt')

    best_metric_filename = os.path.join(model_dir, 'best_metrics.json')
    last_metric_filename = os.path.join(model_dir, 'last_metrics.json')

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Logger.get().info("Saving checkpoint '{}'".format(last_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result_text)
    save_dict_to_json(metrics, last_metric_filename)
    if is_best:
        Logger.get().info("Found new best.")
        shutil.copyfile(last_metric_filename, best_metric_filename)

    # Remove all previous checkpoints except for the best one for storage savings
    Logger.get().info("Removing redundant files")
    files_to_keep = ['last_metrics.json', 'best_metrics.json', 'results.txt', 'args.json']
    files_to_delete = [file for file in os.listdir(model_dir) if file not in files_to_keep]
    for f in files_to_delete:
        if not os.path.isdir(f):
            os.remove("{}/{}".format(model_dir, f))


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: float(v) if (isinstance(v, float) or isinstance(v, int)) else [float(e) for e in v]
             for k, v in d.items()}
        json.dump(d, f, indent=4)
