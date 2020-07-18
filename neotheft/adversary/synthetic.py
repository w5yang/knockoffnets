import argparse
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL.Image import Image
from tqdm import tqdm
import os
import os.path as osp
import pickle
import json
from datetime import datetime
import random

from knockoff.victim.blackbox import Blackbox
from neotheft.utils.synthetic_sample_crafter import IFGSM, IFGSMMod, AdversarialExampleCrafter
import configs.config as cfg
import knockoff.utils.model as model_utils
import datasets
import models.zoo as zoo
from knockoff.adversary.train import TransferSetImagePaths, TransferSetImages, get_optimizer

class SyntheticAdversary(object):
    crafter: AdversarialExampleCrafter
    blackbox: Blackbox

    def __init__(self, blackbox: Blackbox, classifier: nn.Module, device: torch.device, **kwargs):
        """ Synthetic Adversary Initializer

        Args:
            blackbox: Blackbox model of victim.
            classifier: Adversary's classifier.
            device: Adversary's environment, cuda or cpu.
            **kwargs:
                method: str: method of adversary, format: direction_method-gradient_method
                directions: int: number of directions.
                eps: float: the maximum change that can be applied to a sample, range(1.0, 255.0)
                min_pixel: float: the lower bound of each torch.Tensor.
                max_pixel: float: the upper bound of each torch.Tensor.
                init_alpha: float: IFGSM-like required.
                num_steps: int: iterative step of IFGSM.
        """
        self.blackbox = blackbox
        self.classifier = classifier
        self.device = device
        method_list = kwargs['method'].split('-')
        self.direction_num = kwargs['directions']
        self.direction_method = method_list[0]
        assert self.direction_method in ('topk', 'rand')
        self.params = kwargs
        self.picker = lambda x: eval('self.{}(x, k={}, exclude_max=True)'
                                     .format(self.direction_method, self.direction_num))
        self.craft_method = method_list[1]
        assert self.craft_method in ('IFGSM', 'IFGSMMod')
        self.crafter = eval(self.craft_method + '(eps={}, min_pixel={}, max_pixel={}, targeted_attack=True)'
                            .format(kwargs['eps'], kwargs['min_pixel'], kwargs['max_pixel']))

    def topk(self, x: tuple, k: int, exclude_max=False) -> torch.Tensor:
        x = x[0]
        x = x.unsqueeze(0)
        x = x.to(self.device)
        self.classifier.eval()
        topk = torch.topk(self.classifier(x), k + 1)[1].view(-1, 1)
        if exclude_max:
            return topk[1:]
        else:
            return topk[:-1]

    def rand(self, x: torch.Tensor, k: int, exclude_max=False) -> torch.Tensor:
        x = x[0]
        x = x.unsqueeze(0)
        x = x.to(self.device)
        self.classifier.eval()
        result = self.classifier(x)
        torch.arange(result.shape[1])
        torch.randperm()

    # def __both(self, x: torch.Tensor, k:int, exclude_max=False) -> torch.Tensor:
    #     topk = torch.topk(self.classifier(x), 1)[1]

    def query(self, transfer_samples: list) -> list:
        """query blackbox"""
        transfer_set = []
        x = torch.stack(transfer_samples)
        y = self.blackbox(x)
        for i in range(len(transfer_samples)):
            transfer_set.append((transfer_samples[i], y[i]))
        return transfer_set

    def synthesize(self, samples: list, transform=None) -> list:
        """Synthesize a transferset from samples with initialized method and params.

        Args:
            samples: a list contains sample tensor or numpy array, image(a transform is needed).
            transform: for an images or numpy array
        Returns:
            a list contains (generated_sample, sample_label)
        """
        synthetic_set = []
        directions = list(map(self.picker, samples))
        for (x, y), direction in tqdm(zip(samples, directions)):
            if isinstance(x, np.ndarray) or isinstance(x, Image):
                x = transform(x)
            elif not isinstance(x, torch.Tensor):
                raise TypeError
            for d in range(self.direction_num):
                crafted_x = self.crafter(model=self.classifier, tensor=x, target=direction[d],
                                         init_alpha=self.params['init_alpha'], num_steps=self.params['num_steps'], targeted_attack=True)
                crafted_x.squeeze(0)
                synthetic_set.append(crafted_x)
        return self.query(synthetic_set)


class TransferSetTensor(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, torch.Tensor):
        return TransferSetTensor(samples[:budget])
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray, torch.tensor)'.format(type(sample_x)))


def train(model, transferset_samples, budget, round):
    transferset = samples_to_transferset(transferset_samples, budget=budget)
    print()
    print('=> Training at budget = {}'.format(len(transferset)))
    checkpoint_suffix = '.{}.{}'.format(budget, round)

    model_utils.train_model(model, transferset, train.model_dir, testset=train.testset,
                            criterion_train=train.criterion_train, checkpoint_suffix=checkpoint_suffix,
                            device=train.device, optimizer=train.optimizer, **train.params)

def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('victim_model_dir', metavar='PATH', type=str, help='Directory of Victim Blackbox')
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    parser.add_argument('--rounds', metavar='R', type=str,
                        help='Comma seperates values of duplication rounds of each budget.')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--init_alpha', type=float, default=1.0, metavar='I',
                        help='initial iteration step (default: 1.0)')
    parser.add_argument('--num_steps', type=int, default=80, metavar='I',
                        help='iteration steps of each crafted sample (default: 80)')
    parser.add_argument('--eps', type=float, default=255.0, metavar='E',
                        help='maximum change that can be done on a image. (default: 255.0)')
    parser.add_argument('--method', type=str, default='topk-IFGSMMod', metavar='METHOD', help='direction_method-gradient_method')
    parser.add_argument('--directions', type=int, default=2, metavar='D', help='directions')
    parser.add_argument('--max_pixel', type=float, default=1.0, metavar='P', help='upper bound')
    parser.add_argument('--min_pixel', type=float, default=0.0, metavar='P', help='lower bound')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    train.model_dir = params['model_dir']

    # ----------- Set up transferset
    transferset_path = osp.join(train.model_dir, 'transferset.pickle')
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples[0][1].size(0)
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # ----------- Clean up transfer (if necessary)
    if params['argmaxed']:
        new_transferset_samples = []
        print('=> Using argmax labels (instead of posterior probabilities)')
        for i in range(len(transferset_samples)):
            x_i, y_i = transferset_samples[i]
            argmax_k = y_i.argmax()
            y_i_1hot = torch.zeros_like(y_i)
            y_i_1hot[argmax_k] = 1.
            new_transferset_samples.append((x_i, y_i_1hot))
        transferset_samples = new_transferset_samples

    # ----------- Set up testset
    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    model = model.to(device)

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Set up train params
    budgets = [int(b) for b in params['budgets'].split(',')]
    rounds = [int(r) for r in params['rounds'].split(',')]
    np.random.seed(cfg.DEFAULT_SEED)
    torch.manual_seed(cfg.DEFAULT_SEED)
    torch.cuda.manual_seed(cfg.DEFAULT_SEED)
    train.optimizer = get_optimizer(model.parameters(), params['optimizer_choice'], **params)
    train.criterion_train = model_utils.soft_cross_entropy
    train.params = params
    train.device = device
    train.testset = testset

    print(params)

    # Set up crafter params
    original_samples = transferset_samples[:]
    adversary = SyntheticAdversary(blackbox=blackbox, classifier=model, device=device, **params)

    for b, r in zip(budgets, rounds):
        if params['pretrained'] is None:
            train(model, original_samples, b, 1)
        total_samples = transferset_samples
        latest_samples = random.sample(total_samples, b)
        for r in range(2, r + 1):
            latest_samples = adversary.synthesize(latest_samples)
            transferset_samples = original_samples[:]
            transferset_samples.extend(latest_samples)
            total_samples.extend(latest_samples)
            train(model, transferset_samples, b, r)
            latest_samples = random.sample(total_samples, b)


    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(train.model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
