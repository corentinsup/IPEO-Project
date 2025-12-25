#  metrics and losses
import torch


def precision(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum(dim=[0, 1, 2])
    fp = ((pred == label) & (true != label)).sum(dim=[0, 1, 2])

    result = torch.true_divide(tp, tp + fp)
    result[(tp == fp) & (fp == 0)] = 0

    return result

'''
def tp_fp_fn(pred, true, acm=False, label=1):
    """Retruns tp, fp, fn mean of whole batch or array of summed tp, fp, fn per image"""
    tp = ((pred == label) & (true == label)).sum(dim=[1, 2])
    fp = ((pred == label) & (true != label)).sum(dim=[1, 2])
    fn = ((pred != label) & (true == label)).sum(dim=[1, 2])

    if not acm:
        tp = tp.sum(dim=0)
        fp = fp.sum(dim=0)
        fn = fn.sum(dim=0)

    return tp, fp, fn
'''

def recall(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum(dim=[0, 1, 2])
    fn = ((pred != label) & (true == label)).sum(dim=[0, 1, 2])

    result = torch.true_divide(tp, tp + fn)
    result[(tp == fn) & (fn == 0)] = 0

    return result

def pixel_acc(pred, true):
    correct = (pred == true).sum(dim=[0, 1, 2])
    count = true.shape[0] * true.shape[1] * true.shape[2]
    return torch.true_divide(correct, count)

def IoU(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum(dim=[-2, -1])
    fp = ((pred == label) & (true != label)).sum(dim=[-2, -1])
    fn = ((pred != label) & (true == label)).sum(dim=[-2, -1])
    return torch.true_divide(tp, tp + fp + fn)

def dice(pred, true, label=1):
    
    tp = ((pred == label) & (true == label)).sum(dim=[-2, -1])
    fp = ((pred == label) & (true != label)).sum(dim=[-2, -1])
    fn = ((pred != label) & (true == label)).sum(dim=[-2, -1])
    
    result = torch.true_divide(2 * tp, 2 * tp + fp + fn)
    result[(tp == 0) & (fp == 0) & (fn == 0)] = 0
    
    return result

def metrics(y_hat, y, metrics_opts):
    results = {}
    for k, metric in metrics_opts.items():

        # copy tensors to avoid modifying in-place
        y_hat_metric = y_hat.detach()
        y_metric = y.detach()

        # apply threshold only for this metric
        if "threshold" in metric:
            y_hat_metric = (y_hat_metric > metric["threshold"]).float()

        metric_fun = globals()[k]
        results[k] = metric_fun(y_hat_metric, y_metric)

    return results

def update_metrics(main_metrics, batch_metrics):
    for k, v in batch_metrics.items():
        v = v.detach().cpu().float().view(1,-1)

        if k not in main_metrics:
            main_metrics[k] = v
        else:
            main_metrics[k] = torch.cat((main_metrics[k], v), dim=1)

def agg_metrics(metrics):
    for k, v in metrics.items():
        if v.dim() == 2:
            metrics[k] = v.mean(dim=1)
        else:
            metrics[k] = v.mean()
        metrics[k] = metrics[k].cpu()

