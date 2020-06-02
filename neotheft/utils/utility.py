import argparse
from typing import Dict, Any
import torch
import os
import datasets

from knockoff.victim.blackbox import Blackbox
from neotheft.models import zoo


def parser_dealer(option: Dict[str, bool]) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    if option['transfer']:
        parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training',
                            choices=['random', 'adaptive'])
        parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                            required=True)
        parser.add_argument('--out_dir', metavar='PATH', type=str,
                            help='Destination directory to store transfer set', required=True)
        parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True)

    if option['active']:
        parser.add_argument('strategy', metavar='S', type=str, help='Active Sample Strategy',
                            choices=['kcenter', 'random', 'dfal'])
        parser.add_argument('--metric', metavar="M", type=str, help='K-Center method distance metric',
                            choices=['euclidean', 'manhattan', 'l1', 'l2'], default='euclidean')
        parser.add_argument('sampleset', metavar='DS_NAME', type=str,
                            help='Name of sample dataset in active learning selecting algorithms')
        parser.add_argument('--initial-size', metavar='N', type=int, help='Active Learning Initial Sample Size',
                            default=100)
        parser.add_argument('--budget-per-iter', metavar='N', type=int, help='budget for every iteration',
                            default=100)
        parser.add_argument('--iterations', metavar='N', type=int, help='iteration times',
                            default=10)
    if option['synthetic']:
        parser.add_argument('synthetic_method', metavar='SM', type=str, help='Synthetic Method',
                            choices=['fgsm', 'ifgsm'])
    if option['black_box']:
        parser.add_argument('victim_model_dir', metavar='VIC_DIR', type=str,
                            help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
        parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    if option['train']:
        parser.add_argument('model_dir', metavar='SUR_DIR', type=str, help='Surrogate Model Destination directory')
        parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
        parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
        # Optional arguments
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                            help='number of epochs to train (default: 100)')
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
        parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
        parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
        parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm',
                            choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    # apply to all circumstances
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data',
                        default=10)
    args = parser.parse_args()
    params = vars(args)
    device = device_dealer(**params)
    params['device'] = device
    if option['black_box']:
        blackbox_dir = params['victim_model_dir']
        params['blackbox'] = Blackbox.from_modeldir(blackbox_dir, device)
    if option['active']:
        sample_set_name = params['sampleset']
        assert sample_set_name in datasets.__dict__.keys()
        modelfamily = datasets.dataset_to_modelfamily[sample_set_name]
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        params['queryset'] = datasets.__dict__[sample_set_name](train=True, transform=transform)
    if option['train']:
        testset_name = params['testdataset']
        assert testset_name in datasets.__dict__.keys()
        modelfamily = datasets.dataset_to_modelfamily[testset_name]
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        testset = datasets.__dict__[testset_name](train=False, transform=transform)
        params['testset'] = testset
        num_classes = testset.classes
        pretrained_path = params['pretrained']
        model_arch = params['model_arch']

        model = zoo.get_net(model_arch, modelfamily, pretrained_path, num_classes=num_classes)
        params['surrogate'] = model.to(device)
    return params


def device_dealer(**params) -> torch.device:
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


if __name__ == '__main__':
    # test
    parser_dealer(option={
        'transfer': False,
        'active': True,
        'synthetic': False,
        'black_box': True,
        'train': True
    })
