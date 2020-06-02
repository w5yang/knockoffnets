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
from typing import List, Tuple
import pickle
from neotheft.utils.subset_selection_strategy import RandomSelectionStrategy, KCenterGreedyApproach
import knockoff.utils.model as model_utils
from knockoff.adversary.train import get_optimizer


class ActiveAdversary(object):
    def __init__(self,
                 blackbox: Blackbox,
                 surrogate: Module,
                 queryset: Dataset,
                 testset: Dataset,
                 out_dir: str,
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
        self.path = out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.batch_size = batch_size
        self.num_worker = num_workers
        self.selected_index = set()
        self.selected: List[Tuple[Tensor, Tensor]] = []  # [(img_tensor, output_tensor)]
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
        self.optim = get_optimizer(self.surrogate.parameters(), optimizer_choice, **kwargs)
        self.criterion = model_utils.soft_cross_entropy
        self.evaluation_set = self.query_dataset([sample[0] for sample in testset], argmax=True)
        initial_samples = self.sss.get_selecting_tensor()
        self.iterations = 0
        init_queryset = self.query_dataset(initial_samples)
        self.train(init_queryset)
        self.kwargs = kwargs

    def query_dataset(self, training_samples: List[Tensor], argmax: bool = False) -> List[Tuple[Tensor, Tensor]]:
        training_set = []
        idx_set = set(range(len(training_samples)))
        with tqdm(total=len(training_samples)) as pbar:
            for t, B in enumerate(range(len(training_samples), self.batch_size)):
                idxs = np.random.choice(list(idx_set), replace=False,
                                        size=min(self.batch_size, len(training_samples)))
                idx_set = idx_set - set(idxs)

                x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.device)
                y_t = self.blackbox(x_t)
                if argmax:
                    y_t = y_t.argmax(1)
                for i in range(x_t.size(0)):
                    training_set.append((x_t[i], y_t[i]))
                pbar.update(x_t.size(0))
        return training_set

    def train(self, training_set: List[Tuple[Tensor, Tensor]]):
        model_utils.train_model(self.surrogate, training_set, self.path, self.evaluation_set, self.criterion,
                                checkpoint_suffix='.{}.iter'.format(self.iterations), device=self.device,
                                optimizer=self.optim, **self.kwargs)
        self.iterations += 1

    def save_selected(self):
        selected_output_path = os.path.join(self.path, 'selection.{}.pickle'.format(len(self.selected_index)))
        if os.path.exists(selected_output_path):
            print("{} exists, override file.".format(selected_output_path))
        with open(selected_output_path, 'wb') as fp:
            pickle.dump(self.selected, fp)
        print("=> selected {} samples written to {}".format(len(self.selected_index), selected_output_path))

    def step(self, size: int):
        samples = self.sss.get_subset(size)
        sample_ds = self.query_dataset(samples)
        self.train(sample_ds)


def main():
    torch.manual_seed(cfg.DEFAULT_SEED)
    params = parser_dealer(
        {
            'transfer': False,
            'active': True,
            'synthetic': False,
            'black_box': True,
            'train': True
        }
    )
    active_adv = ActiveAdversary(**params)
    for i in params['iterations']:
        active_adv.step(params['budget-per-iter'])
    active_adv.save_selected()


if __name__ == '__main__':
    main()
