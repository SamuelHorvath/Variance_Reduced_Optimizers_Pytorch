import numpy as np
import os
import glob
import json

# os.environ["WANDB_API_KEY"] = '205924dacff241a772a053e251a37bc15ceb90b4'
loader_kwargs = {
    # 'pin_memory': False,
    # 'persistent_workers': False,
    # 'prefetch_factor': 2,
}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def get_val(self):
        return self.val

    def get_avg(self):
        return self.avg


def init_metrics_meter(epoch=None):
    if epoch is not None:
        metrics_meter = {
            'epoch': epoch,
            'loss': AverageMeter(),
            'top_1_acc': AverageMeter(),
        }
    else:
        metrics_meter = {
            'train_epoch': [], 'train_loss': [], 'train_top_1_acc': [],
            'test_epoch': [], 'test_loss': [], 'test_top_1_acc': [], 'test_grad_norm': []
        }
    return metrics_meter


def get_model_str_from_obj(model):
    return str(list(model.modules())[0]).split("\n")[0][:-1]


def create_metrics_dict(metrics, train=True):
    key = get_key(train)
    metrics_dict = {key + 'epoch': metrics['epoch']}
    for k in metrics:
        if k == 'epoch':
            continue
        metrics_dict[key + k] = metrics[k].get_avg()
    return metrics_dict


def extend_metrics_dict(full_metrics, last_metrics, grad_norm=None, add_grad_norm=False):
    k = None
    for k in last_metrics:
        if last_metrics[k] is not None:
            full_metrics[k].append(last_metrics[k])
    if add_grad_norm:
        full_metrics[k.split('_')[0] + '_grad_norm'].append(grad_norm)


def just_epoch_key(metrics):
    for k in metrics:
        if k[-5:] == 'epoch':
            metrics['epoch'] = metrics[k]
            del metrics[k]
            break
    return metrics


def create_model_dir(args, lr=True, only_setup=False):
    model_dataset = '_'.join([args.model, args.dataset])
    run_id = f'id={args.run_id}'
    model_dir = os.path.join(run_id, model_dataset, args.method)
    if not only_setup:
        model_dir = os.path.join(args.checkpoint_dir, model_dir)
    if lr:
        run_hp = os.path.join(f"lr={str(args.lr)}",
                              f"seed={str(args.manual_seed)}")
        model_dir = os.path.join(model_dir, run_hp)

    return model_dir


def get_key(train=True):
    return 'train_' if train else 'test_'


def get_best_lr_and_metric(args, last=True):
    best_arg, best_lookup = (np.nanargmin, np.nanmin) if args.metric in ['loss', 'grad_norm'] \
        else (np.nanargmax, np.nanmax)
    best_metric_nan = np.inf if args.metric in ['loss', 'grad_norm'] else -np.inf
    key = get_key(args.train_metric)
    model_dir_no_lr = create_model_dir(args, lr=False)
    lr_dirs = [lr_dir for lr_dir in os.listdir(model_dir_no_lr)
               if os.path.isdir(os.path.join(model_dir_no_lr, lr_dir))
               and not lr_dir.startswith('.')]

    lrs = np.array([float(lr_dir.split('=')[-1]) for lr_dir in lr_dirs])

    best_runs_metric = list()
    for lr_dir in lr_dirs:
        # /*/ for different seeds
        lr_metric_dirs = glob.glob(model_dir_no_lr + '/' + lr_dir + '/*/full_metrics.json')
        lr_metric = list()
        for lr_metric_dir in lr_metric_dirs:
            with open(lr_metric_dir) as json_file:
                metrics = json.load(json_file)
            metric_values = metrics[key + args.metric]
            metric = metric_values[-1] if last else best_lookup(metric_values)
            lr_metric.append(metric)
        # can be replaced with best_lookup to include best run rather than average
        # best_runs_metric.append(np.nanmean(lr_metric))
        if all(np.isnan(lr_metric)):
            best_runs_metric.append(np.nan)
        else:
            best_runs_metric.append(best_lookup(lr_metric))

    if all(np.isnan(best_runs_metric)):
        return np.min(lrs), best_metric_nan, np.sort(lrs)
    i_best_lr = best_arg(best_runs_metric)
    best_metric = best_runs_metric[i_best_lr]
    best_lr = lrs[i_best_lr]
    return best_lr, best_metric, np.sort(lrs)


def get_best_runs(args_exp, last=True):
    model_dir_no_lr = create_model_dir(args_exp, lr=False)
    best_lr, _, _ = get_best_lr_and_metric(args_exp, last=last)
    model_dir_lr = os.path.join(model_dir_no_lr, f"lr={str(best_lr)}")
    json_dir = 'full_metrics.json'
    metric_dirs = glob.glob(model_dir_lr + '/*/' + json_dir)

    with open(metric_dirs[0]) as json_file:
        metric = json.load(json_file)
    runs = [metric]

    for metric_dir in metric_dirs[1:]:
        with open(metric_dir) as json_file:
            metric = json.load(json_file)
        # ignores failed runs
        if not any(np.isnan(metric['train_loss'])):
            runs.append(metric)
    return runs


def log_epoch_info(logger, spend_budget, budget, metrics_meter, train=True):
    mode_str = 'Train' if train else 'Test'
    logger.get().info("{mode_str} [{current_batch}/{total_batches}]\t"
                      "Loss {loss:.4f}\t"
                      "Prec@1 {prec1:.3f}\t".format(
                        mode_str=mode_str,
                        current_batch=spend_budget,
                        total_batches=budget,
                        loss=metrics_meter['loss'].get_avg(),
                        prec1=metrics_meter['top_1_acc'].get_avg()))
