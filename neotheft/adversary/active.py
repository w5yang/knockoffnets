"""Coreset Model Extraction

"""

import numpy as np
from neotheft.utils.utility import parser_dealer
from tqdm import tqdm

import configs.config as cfg
from knockoff.victim.blackbox import Blackbox
from torch.utils.data import Dataset
from torch.nn import Module
from torch import Tensor
import torch
import os
from typing import List, Tuple, Set
import pickle
from neotheft.utils.subset_selection_strategy import RandomSelectionStrategy, KCenterGreedyApproach
import knockoff.utils.model as model_utils
from knockoff.adversary.train import get_optimizer
from models import zoo


class ActiveAdversary(object):
    def __init__(self,
                 blackbox: Blackbox,
                 surrogate: Module,
                 queryset: Dataset,
                 testset: Dataset,
                 model_dir: str,
                 batch_size: int = 50,
                 num_workers: int = 15,
                 strategy: str = 'random',
                 metric: str = 'euclidean',
                 initial_size: int = 100,
                 device: torch.device = torch.device('cpu'),
                 optimizer_choice: str = 'sgdm',
                 **kwargs
                 ):
        self.device = device
        self.blackbox = blackbox
        self.surrogate = surrogate
        self.queryset = queryset
        self.path = model_dir
        self.kwargs = kwargs
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.batch_size = batch_size
        self.num_worker = num_workers
        self.selected: List[Tuple[Tensor, Tensor]] = []  # [(img_tensor, output_tensor)]
        self.evaluation_set: List[Tuple[Tensor, Tensor]] = []
        assert strategy in ('random', 'kcenter')
        if strategy == 'random':
            self.sss = RandomSelectionStrategy(
                dataset=self.queryset,
                model=self.surrogate,
                initial_size=initial_size,
                seed=cfg.DEFAULT_SEED,
                batch_size=self.batch_size
            )
        elif strategy == 'kcenter':
            self.sss = KCenterGreedyApproach(
                dataset=self.queryset,
                model=self.surrogate,
                initial_size=initial_size,
                seed=cfg.DEFAULT_SEED,
                batch_size=self.batch_size,
                metric=metric,
                device=device
            )
        else:
            raise NotImplementedError
        self.optimizer_choice = optimizer_choice
        self.optim = get_optimizer(self.surrogate.parameters(), optimizer_choice, **kwargs)
        self.queried: Set[int] = set()
        self.criterion = model_utils.soft_cross_entropy
        self.query_dataset([sample[0] for sample in testset], argmax=True, train=False)
        self.iterations = 0
        self.query_index(self.sss.selecting)
        self.train()

    def query_dataset(self, training_samples: List[Tensor], argmax: bool = False, train: bool = True):
        with tqdm(total=len(training_samples)) as pbar:
            for t, B in enumerate(range(0, len(training_samples), self.batch_size)):
                x_t = torch.stack([training_samples[i] for i in range(B, min(B + self.batch_size, len(training_samples)))]).to(self.device)
                y_t = self.blackbox(x_t)
                if self.kwargs['argmaxed'] or argmax:
                    y_t = y_t.argmax(1)
                elif self.kwargs['topk'] != 0:
                    v, i = y_t.topk(self.kwargs['topk'], 1)
                    y_t = torch.zeros_like(y_t).scatter(1, i, v)
                for i in range(x_t.size(0)):
                    if train:
                        self.selected.append((x_t[i].cpu(), y_t[i].cpu()))
                    else:
                        self.evaluation_set.append((x_t[i].cpu(), y_t[i].cpu()))
                pbar.update(x_t.size(0))

    def query_index(self, index_set: Set[int]):
        if len(index_set.intersection(self.queried)) > 0:
            raise Exception("Double query.")
        for index in index_set:
            x: Tensor = self.queryset[index][0].unsqueeze(0).to(self.device)
            y = self.blackbox(x)
            if self.kwargs['argmaxed']:
                y = y.argmax(1)
            elif self.kwargs['topk'] != 0:
                v, i = y.topk(self.kwargs['topk'], 1)
                y = torch.zeros_like(y).scatter(1, i, v)
            self.selected.append((x.squeeze(0).cpu(), y.squeeze(0).cpu()))
        self.queried.update(index_set)
        np.random.shuffle(self.selected)

    def train(self):
        # self.surrogate = zoo.get_net(self.kwargs["model_arch"], 'custom_cnn', None, num_classes=43).to(self.device)
        # self.optim = get_optimizer(self.surrogate.parameters(), self.optimizer_choice, **self.kwargs)
        model_utils.train_model(self.surrogate, self.selected, self.path, batch_size=self.batch_size,
                                testset=self.evaluation_set, criterion_train=self.criterion,
                                checkpoint_suffix='.{}.iter'.format(self.iterations), device=self.device,
                                optimizer=self.optim, **self.kwargs)

    def save_selected(self):
        self.sss.merge_selection()
        selected_index_output_path = os.path.join(self.path, 'selection.{}.pickle'.format(len(self.sss.selected)))
        selected_transfer_outpath = os.path.join(self.path, "transferset.{}.pickle".format(len(self.selected)))
        if os.path.exists(selected_index_output_path):
            print("{} exists, override file.".format(selected_index_output_path))
        with open(selected_index_output_path, 'wb') as fp:
            pickle.dump(self.sss.selected, fp)
        print("=> selected {} samples written to {}".format(len(self.sss.selected), selected_index_output_path))
        if os.path.exists(selected_transfer_outpath):
            print("{} exists, override file.".format(selected_transfer_outpath))
        with open(selected_transfer_outpath, 'wb') as fp:
            pickle.dump(self.selected, fp)
        print("=> selected {} samples written to {}".format(len(self.selected), selected_transfer_outpath))

    def step(self, size: int):
        self.sss.get_subset(size)
        self.query_index(self.sss.selecting)
        self.train()
        self.iterations += 1


def main():
    torch.manual_seed(cfg.DEFAULT_SEED)
    params = parser_dealer(
        {
            'transfer': False,
            'active': True,
            'sampling': True,
            'synthetic': False,
            'black_box': True,
            'train': True
        }
    )
    active_adv = ActiveAdversary(**params)
    for i in range(params['iterations']):
        print("{} samples selected.".format(len(active_adv.sss.selected)))
        active_adv.step(params['budget_per_iter'])
    active_adv.save_selected()


if __name__ == '__main__':
    main()
