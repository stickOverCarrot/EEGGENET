import numpy as np
import torch
import random
import errno
import os
import sys
import time

from models.utils import normalize_adj


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_save_path(father_path, args):
    father_path = os.path.join(father_path, '{}'.format(time.strftime("%m_%d_%H_%M")))
    mkdir(father_path)
    args.log_path = father_path
    args.tensorboard_path = os.path.join(father_path, 'tensorboard')
    args.model_adj_path = father_path
    args.model_classifier_path = father_path
    return args


def save(checkpoints, save_path):
    torch.save(checkpoints, save_path)


def load_adj(dn='bciciv2a', norm=False):
    if 'hgd' == dn:
        num_node = 44
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(1, 21), (1, 11), (1, 25), (1, 14),
                         (2, 22), (2, 37), (2, 11), (2, 12), (2, 26), (2, 15), (2, 39),
                         (3, 38), (3, 23), (3, 12), (3, 13), (3, 40), (3, 16), (3, 27),
                         (4, 24), (4, 13), (4, 28), (4, 17),
                         (5, 25), (5, 11), (5, 26), (5, 14), (5, 15), (5, 29), (5, 18), (5, 30),
                         (6, 27), (6, 13), (6, 28), (6, 16), (6, 17), (6, 31), (6, 20), (6, 32),
                         (7, 14), (7, 29), (7, 18), (7, 33),
                         (8, 30), (8, 15), (8, 41), (8, 18), (8, 19), (8, 34), (8, 43),
                         (9, 42), (9, 16), (9, 31), (9, 19), (9, 20), (9, 44), (9, 35),
                         (10, 17), (10, 32), (10, 20), (10, 36),
                         (11, 21), (11, 22), (11, 25), (11, 26),
                         (12, 37), (12, 38), (12, 39), (12, 40),
                         (13, 23), (13, 24), (13, 27), (13, 28),
                         (14, 25), (14, 29),
                         (15, 26), (15, 39), (15, 30), (15, 41),
                         (16, 40), (16, 27), (16, 42), (16, 31),
                         (17, 28), (17, 32),
                         (18, 29), (18, 30), (18, 33), (18, 34),
                         (19, 41), (19, 42), (19, 43), (19, 44),
                         (20, 31), (20, 32), (20, 35), (20, 36),
                         (21, 22), (21, 25),
                         (22, 37), (22, 26),
                         (23, 38), (23, 24), (23, 27),
                         (24, 28),
                         (25, 26), (25, 29),
                         (26, 39), (26, 30),
                         (27, 40), (27, 28), (27, 31),
                         (28, 32),
                         (29, 30), (29, 33),
                         (30, 41), (30, 34),
                         (31, 42), (31, 32), (31, 35),
                         (32, 36),
                         (33, 34),
                         (34, 43),
                         (35, 36), (35, 44),
                         (37, 38), (37, 39),
                         (38, 40),
                         (39, 40), (39, 41),
                         (40, 42),
                         (41, 43), (41, 42),
                         (42, 44),
                         (43, 44)]
    elif 'bciciv2a' == dn:
        num_node = 22
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(1, 3), (1, 4), (1, 5),
                         (2, 3), (2, 7), (2, 8), (2, 9),
                         (3, 4), (3, 8), (3, 9), (3, 10),
                         (4, 5), (4, 9), (4, 10), (4, 11),
                         (5, 6), (5, 10), (5, 11), (5, 12),
                         (6, 11), (6, 12), (6, 13),
                         (7, 8), (7, 14),
                         (8, 9), (8, 14), (8, 15),
                         (9, 10), (9, 14), (9, 15), (9, 16),
                         (10, 11), (10, 15), (10, 16), (10, 17),
                         (11, 12), (11, 16), (11, 17), (11, 18),
                         (12, 13), (12, 17), (12, 18),
                         (13, 18),
                         (14, 15), (14, 19),
                         (15, 16), (15, 19), (15, 20),
                         (16, 17), (16, 19), (16, 20), (16, 21),
                         (17, 18), (17, 20), (17, 21),
                         (18, 21),
                         (19, 20), (19, 22),
                         (20, 21), (20, 22),
                         (21, 22)]
    else:
        raise ValueError('cant support {} dataset'.format(dn))
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
    edge = self_link + neighbor_link
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[i, j] = 1.
        A[j, i] = 1.
    if norm:
        A = normalize_adj(torch.tensor(A, dtype=torch.float32), mode='sym')
    return A


def accuracy(output, target, topk=(1,)):
    shape = None
    if 2 == len(target.size()):
        shape = target.size()
        target = target.view(target.size(0))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    if shape:
        target = target.view(shape)
    return ret


class Logger(object):
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    """
    Early stops the training if validation loss
    doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, max_epochs=80):
        """
        patience (int): How long to wait after last time validation
        loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation
        loss improvement.
                        Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        # my addition:
        self.max_epochs = max_epochs
        self.max_epoch_stop = False
        self.epoch_counter = 0
        self.should_stop = False
        self.checkpoint = None

    def __call__(self, val_loss):
        # my addition:
        self.epoch_counter += 1
        if self.epoch_counter >= self.max_epochs:
            self.max_epoch_stop = True

        score = val_loss

        if self.best_score is None:
            print('')
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} '
                  f'out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        # my addition:
        if any([self.max_epoch_stop, self.early_stop]):
            self.should_stop = True
