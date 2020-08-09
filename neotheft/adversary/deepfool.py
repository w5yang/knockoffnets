import numpy as np
import os

from torch.nn import Module
from tqdm import tqdm
import argparse
from typing import List
import multiprocessing

import datasets
from knockoff.victim.blackbox import Blackbox
from models import zoo
from neotheft.utils.deepfool import deepfool
from neotheft.utils.utility import parser_dealer, query, load_transferset, save_selection_state, device_dealer, \
    load_state
import knockoff.utils.model as model_utils
from knockoff.adversary.train import get_optimizer


def deepfool_active():
    params = parser_dealer({
        'transfer': False,
        'active': False,
        'sampling': False,
        'synthetic': False,
        'black_box': True,
        'train': True
    })
    model_dir = params['model_dir']
    transferset, num_classes = load_transferset(os.path.join(model_dir, 'transferset.pickle'))
    surrogate = params['surrogate']
    blackbox = params['blackbox']
    device = params['device']
    testset = params['testset']
    # todo: make the iteration batch and rounds part of parameters.
    iter_batch = 1000
    rounds = 20
    remnant = set(range(len(transferset)))
    selected = set()
    params['testset'] = query(blackbox, [data[0] for data in testset], len(testset), device=device, argmax=True)
    optimizer = get_optimizer(surrogate.parameters(), params['optimizer_choice'], **params)

    for i in range(rounds):
        batch_samples = []
        batch_permutation = []
        current_round = list(remnant)
        print('round {}: {} object to calculate.'.format(i + 1, len(remnant)))
        for index in current_round:
            permutation, _, _, _, result = deepfool(transferset[index][0], surrogate, num_classes)
            batch_samples.append(result.squeeze(0))
            batch_permutation.append(np.linalg.norm(permutation))
        batch_permutation = np.array(batch_permutation)
        selection = batch_permutation.argsort(0)[:iter_batch]
        training_batch = [batch_samples[i] for i in selection]
        selection = [current_round[i] for i in selection]
        selected.update(selection)
        remnant.difference_update(selection)
        transferset.extend(query(blackbox, training_batch, iter_batch, device=device))
        model_utils.train_model(surrogate, transferset, model_dir,
                                criterion_train=model_utils.soft_cross_entropy,
                                optimizer=optimizer, **params)


def deepfool_choose(target_model: Module, blackbox: Blackbox, queryset, testset, selection: set, transferset: List, indices_list: List, device, **params):
    model_dir = params['model_dir']
    # transferset, num_classes = load_transferset(os.path.join(model_dir, 'transferset.pickle'))
    surrogate = target_model
    evalutation_set = query(blackbox, [data[0] for data in testset], len(testset), device=device, argmax=True)
    optimizer = get_optimizer(surrogate.parameters(), params['optimizer_choice'], **params)
    # todo: make deepfool judge budget criteria direction part of parameter.
    reverse = True
    budget = params['deepfool_budget']
    # batch_samples = []
    batch_permutation = []
    total = set([i for i in range(len(queryset))])
    unselected = list(total - selection)
    num_classes = len(testset.classes)
    print('{} object to calculate.'.format(len(unselected)))
    dm = DeepfoolMappable(surrogate, num_classes, queryset)
    with multiprocessing.Pool(4) as pool:
        results = pool.map(dm, unselected)
    batch_permutation = np.array([item[1] for item in results])
    # for index in tqdm(unselected):
    #     permutation, _, _, _, result = deepfool(queryset[index][0], surrogate, num_classes)
    #     batch_samples.append(result.squeeze(0))
    #     batch_permutation.append(np.linalg.norm(permutation))
    batch_permutation = np.array(batch_permutation)
    current_selection = batch_permutation.argsort(0)[:budget] if not reverse else batch_permutation.argsort(0)[
                                                                          len(unselected) - budget:]
    assert len(current_selection) == budget
    training_batch = [results[i][0] for i in current_selection]
    current_selection = [unselected[i] for i in current_selection]
    indices_list.extend(current_selection)
    selection.update(current_selection)
    transferset.extend(query(blackbox, training_batch, budget, device=device))
    model_utils.train_model(surrogate, transferset, model_dir, testset=evalutation_set,
                            criterion_train=model_utils.soft_cross_entropy,
                            optimizer=optimizer, checkpoint_suffix='deepfool', **params)
    save_selection_state(transferset, selection, indices_list, model_dir)


class DeepfoolMappable(object):
    def __init__(self, surrogate, num_classes, queryset):
        self.surrogate = surrogate
        self.num_classes = num_classes
        self.queryset = queryset

    def __call__(self, index):
        permutation, _, _, _, result = deepfool(self.queryset[index][0], self.surrogate, self.num_classes)
        return result.squeeze(0), np.linalg.norm(permutation)


def main():
    parser = argparse.ArgumentParser(description='Select deepfool images, retrain the target model.')
    parser.add_argument('model_dir', metavar='SUR_DIR', type=str,
                        help='Surrogate Model Destination directory, which may contain selecting state, '
                             'aka, selection.pickle, transferset.pickle, select_indices.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('blackbox_dir', metavar='VIC_DIR', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('sampleset', metavar='DS_NAME', type=str,
                        help='Name of sample dataset in deepfool selecting algorithms')
    parser.add_argument('deepfool_budget', metavar='N', type=int,
                        help='deepfool selection size.')
    parser.add_argument('--state-budget', type=int,
                        help="if > 0, load corresponding budget of selection state.", default=0)
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--topk', metavar='TK', type=int, help='iteration times',
                        default=0)
    parser.add_argument('--testset', metavar='DSET', type=str, help="If using full vector", default=None)
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('-x', '--complexity', type=int, default=64, metavar='N',
                        help="Model conv channel size.")
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('--pretrained', type=str, help='Use pretrained network, or a checkpoint', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    parser.add_argument('--optimizer-choice', type=str, help='Optimizer', default='sgdm',
                        choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-d', '--device-id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-w', '--num-workers', metavar='N', type=int, help='# Worker threads to load data',
                        default=10)
    args = parser.parse_args()
    params = vars(args)
    device = device_dealer(device_id=args.device_id)
    blackbox = Blackbox.from_modeldir(args.blackbox_dir, device)
    assert args.sampleset in datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[args.sampleset]
    transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    queryset = datasets.__dict__[args.sampleset](train=True, transform=transform)

    if args.state_budget > 0:
        selection, transfer, indices_list = load_state(state_dir=args.model_dir)
        selection = set(indices_list[:args.state_budget])
    else:
        selection, transfer, indices_list = set(), [], []
    testset_name = args.testdataset
    assert testset_name in datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[testset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    testset = datasets.__dict__[testset_name](train=False, transform=transform)
    num_classes = len(testset.classes)
    pretrained_path = params['pretrained']
    model_arch = params['model_arch']
    sample = testset[0][0]

    model = zoo.get_net(model_arch, modelfamily, pretrained_path, num_classes=num_classes, channel=sample.shape[0],
                        complexity=params['complexity'])
    model = model.to(device)
    deepfool_choose(model, blackbox, queryset, testset, selection, transfer, indices_list, device, **params)


if __name__ == '__main__':
    main()
