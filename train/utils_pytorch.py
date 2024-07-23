from functools import partial
import csv

import torch
from torch.optim.optimizer import Optimizer
import torch



def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))


def save_opt(opt, filepath):
    torch.save(opt.state_dict(), filepath)


def load_opt(opt, filepath):
    opt.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))


def save_to_csv(data, filepath):
    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        writer.writerows(data)


class UniversalFactory:
    classes = []

    def __init__(self, classes=None):
        if classes is not None:
            self.classes = classes
        self.classes_names = {class_.__name__: class_ for class_ in self.classes}

    def create_class(self, class_name, *args, **kwargs):
        instance = self.classes_names[class_name](*args, **kwargs)
        return instance


class EarlyStopReduceLROnPlateau:

    def __init__(
        self,
        optimizer,
        generator,
        filepath,
        mode='min',
        factor=0.1,
        patience=10,
        verbose=False,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        min_lr=0.0,
        eps=1e-8,
        patience_stopping=100
    ):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()
        self.last_lr = False
        self.train = True
        self.patience_stopping = patience_stopping

        self.generator = generator
        self.filepath = filepath

        save_to_csv([['epoch', 'new_lr']], self.filepath + '/scheduler.csv')

    def training(self):
        return self.train

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        assert self.is_better is not None
        assert self.num_bad_epochs is not None
        if self.is_better(current, self.best):
            # better or equal -> See function definition
            self.best = current
            self.num_bad_epochs = 0
            is_better = True
        else:
            self.num_bad_epochs += 1
            is_better = False

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            if self.last_lr:
                self.train = False
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience_stopping:
            self.train = False

        return is_better

    def _reduce_lr(self, epoch):
        last_lr = list()
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    save_to_csv(
                        [[str(epoch),str(new_lr)]],
                        self.filepath + '/scheduler.csv'
                    )
                last_lr.append(False)
            else:
                last_lr.append(True)
        if all(last_lr):
            self.last_lr = True

    def load_best_model(self):
        load_model(self.generator, self.filepath + '/generator_best.pth')
        load_opt(self.optimizer, self.filepath + '/optimizer_best.pth')

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a <= best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a <= best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a >= best * rel_epsilon

        else:
            return a >= best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)
