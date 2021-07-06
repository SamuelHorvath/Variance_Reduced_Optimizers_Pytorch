import sys
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import time

# import wandb

from opts import parse_args
from optims.utils import get_full_gradient, get_grad_norm
from utils.logger import Logger
from utils.data.data_loaders import load_data
from utils.model_funcs import get_training_elements, evaluate_model, run_one_step, get_vr_method
from utils.checkpointing import save_checkpoint, save_dict_to_json
from utils.utils import create_model_dir, init_metrics_meter, extend_metrics_dict, \
    get_best_lr_and_metric, get_key, create_metrics_dict, loader_kwargs


def main(args):
    # In case of DataParallel for .to() to work
    args.device = args.gpu[0] if type(args.gpu) == list else args.gpu

    # Load validation set
    train_set, test_set = load_data(args.data_path, args.dataset,
                                    load_train_set=True, download=True)

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,  **loader_kwargs)

    if not args.evaluate:  # Training mode
        args.budget = len(train_set) * args.epochs
        # for lr tuning, we grid-searched the optimal learning rate, starting from
        # the range of args.lr * {1., 0.5, 0.25, 0.125}. We evaluate an exponentially-spaced grid of learning
        # rates. If the best performance was ever at one of the extremes of the grid, we would try new grid
        # points so that the best performance was contained in the middle of the parameter.
        lrs = [args.lr * 2**(-i) for i in range(4)] if args.tune_lr else [args.lr]
        if args.tune_lr:
            logger.info(f"Tuning learning rate, initial values: {lrs}.")

        edge_case = True
        best_lr = None
        best_metric = None
        while edge_case:
            for lr in lrs:
                args.lr = lr
                logger.info(f"Running run 1: lr...{lr}")
                init_and_train_model(args, train_set, test_loader)

            if not args.tune_lr:
                edge_case = False
            else:
                best_lr, best_metric, lrs = get_best_lr_and_metric(args)
                logger.info(f"Best step-size: {best_lr}, {args.metric}: {best_metric:.2f}")

                if best_lr == lrs[0]:
                    lrs = [lrs[0] / 2, lrs[0] / 4]
                    logger.info('Optimal learning rate is bottom edge case. Adding lrs: {}'.format(lrs))
                elif best_lr == lrs[-1]:
                    lrs = [lrs[-1] * 2, lrs[-1] * 4]
                    logger.info('Optimal learning rate is upper edge case. Adding lrs: {}'.format(lrs))
                else:
                    edge_case = False

        if args.tune_lr:
            save_dict_to_json(
                {'best_lr': best_lr, args.metric: best_metric, 'train': args.train_metric},
                os.path.join(create_model_dir(args, lr=False), 'best_lr.json'))
            logger.info(f'Running {args.total_runs - 1} extra evaluations for the best step size({best_lr}).')

        args.lr = best_lr if args.tune_lr else args.lr
        for _ in range(args.total_runs - 1):
            args.manual_seed += 1
            init_and_train_model(args, train_set, test_loader)

    else:  # Evaluation mode
        model, current_epoch = get_training_elements(
            args.model, args.loss, args.dataset, args.gpu, args.nc_regularizer, args.nc_regularizer_value)

        metrics = evaluate_model(model, test_loader, args.device, current_epoch,
                                 print_freq=50, metric_to_optim=args.metric)

        logger.info(f'Validation metrics: {metrics}')


def init_and_train_model(args, train_set, test_loader):
    full_metrics = init_metrics_meter()
    model_dir = create_model_dir(args)
    # don't train if setup already exists
    if os.path.isdir(model_dir):
        Logger.get().info(f"{model_dir} already exists.")
        Logger.get().info("Skipping this setup.")
        return
    # create model directory
    os.makedirs(model_dir, exist_ok=True)
    # init wandb tracking
    # wandb.init(project='scaled_grad', entity='samuelhovath',
    #            config=vars(args), name=str(create_model_dir(args, only_setup=True)),
    #            reinit=True)
    # save used args as json to experiment directory
    with open(os.path.join(create_model_dir(args), 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    model, current_epoch = get_training_elements(
        args.model, args.loss, args.dataset, args.gpu, args.nc_regularizer, args.nc_regularizer_value)
    vr_method = get_vr_method(
        args.method, model, train_set, args.batch_size, args.lr, args.device, args.num_workers, args.weight_decay)

    metric_to_optim = args.metric
    best_metric = np.inf if metric_to_optim in ['loss', 'grad_norm'] else -np.inf
    train_time_meter = 0

    start = time.time()
    current_epoch = 0
    spend_budget = 0
    i = 0
    metrics_meter = init_metrics_meter(spend_budget)
    while spend_budget < args.budget:
        if spend_budget >= current_epoch * len(train_set):
            metrics_train = create_metrics_dict(metrics_meter, train=True)
            extend_metrics_dict(full_metrics, metrics_train)
            # wandb.log(just_epoch_key(metrics_train))
            train_time = time.time() - start
            train_time_meter += train_time  # Track timings for across epochs average
            Logger.get().debug(f'Epoch train time: {train_time}')

            if current_epoch % args.eval_every == 0 or current_epoch == (args.epochs - 1):
                metrics_eval = evaluate_model(model, test_loader, args.device, spend_budget / len(train_set),
                                              print_freq=1, metric_to_optim=metric_to_optim)
                grad_norm = None
                if args.track_grad_norm:
                    grad = get_full_gradient(model, test_loader, args.device)
                    grad_norm = get_grad_norm(grad)
                extend_metrics_dict(
                    full_metrics, metrics_eval, grad_norm=grad_norm, add_grad_norm=args.track_grad_norm)
                # wandb.log(just_epoch_key(metrics_eval))
                key = get_key(train=False)
                avg_metric = metrics_eval[key + metric_to_optim]
                # Save model checkpoint
                if metric_to_optim == 'loss':
                    is_best = avg_metric < best_metric
                else:
                    is_best = avg_metric > best_metric
                save_checkpoint(is_best=is_best, args=args, metrics=metrics_eval, metric_to_optim=metric_to_optim)
                if is_best:
                    best_metric = avg_metric

                if np.isnan(metrics_eval[key + 'loss']):
                    Logger.get().info('NaN loss detected, aborting training procedure.')
                    break
            current_epoch += 1
            metrics_meter = init_metrics_meter(spend_budget / len(train_set))
            continue
        spend_budget = run_one_step(i, spend_budget, args.budget, vr_method, metrics_meter)
        i += 1
    Logger.get().debug(f'Average epoch train time: {train_time_meter / args.epochs}')
    with open(os.path.join(create_model_dir(args), 'full_metrics.json'), 'w') as f:
        json.dump(full_metrics, f, indent=4)


if __name__ == "__main__":
    global CUDA_SUPPORT

    args = parse_args(sys.argv)
    Logger.setup_logging(args.loglevel, logfile=args.logfile)
    logger = Logger()

    logger.debug(f"CLI args: {args}")

    if torch.cuda.device_count():
        CUDA_SUPPORT = True
        # torch.backends.cudnn.benchmark = True
    else:
        logger.warning('CUDA unsupported!!')
        CUDA_SUPPORT = False

    if not CUDA_SUPPORT:
        args.gpu = "cpu"

    if args.deterministic:
        import torch.backends.cudnn as cudnn
        import os
        import random

        if CUDA_SUPPORT:
            cudnn.deterministic = args.deterministic
            cudnn.benchmark = not args.deterministic
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)

        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.backends.cudnn.deterministic = True

    main(args)
