import argparse
from torch.nn import Module
import torch
import os
from typing import List, Tuple, Union, Iterable
from torch.utils.data import Dataset
from torch import Tensor
import pickle

from knockoff.victim.blackbox import Blackbox
import datasets
from models.zoo import get_net
from knockoff.adversary.train import get_optimizer
from neotheft.utils.utility import query, unpack, load_img_dir, load_state
from knockoff.utils.model import soft_cross_entropy, train_model


class Adversary(object):
    def __init__(self,
                 model_arch: str,
                 state_dir: str,
                 testset: str,
                 pretrained: str = None,
                 sampleset: str = None,
                 blackbox_path: str = None,
                 cuda: bool = True,
                 complexity: int = 64,
                 optimizer_choice: str = 'sgdm',
                 batch_size: int = 64,
                 topk: int = 0,
                 argmax: bool = False,
                 num_workers: int = 16,
                 **kwargs) -> None:
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.state_dir = state_dir  # Store model checkpoint, selected state in Active thief etc.
        self.selection, self.transfer = load_state(state_dir)
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
        self.cuda = cuda
        if blackbox_path is None:
            # if blackbox_path is None, no blackbox model is involved in model training.
            self.blackbox = None
        else:
            self.blackbox = Blackbox.from_modeldir(blackbox_path, self.device)
        modelfamily = datasets.dataset_to_modelfamily[testset]
        # Work around for MNIST. MNISTlike is one channel image and is normalized with specific parameter.
        if testset in ('MNIST', 'KMNIST', 'EMNIST', 'EMNISTLetters', 'FashionMNIST'):
            self.channel = 1
            self.transforms = datasets.MNIST_transform
        else:
            self.channel = 3
            self.transforms = datasets.modelfamily_to_transforms[modelfamily]
        # For absolute accuracy test.
        self.testset = datasets.__dict__[testset](train=False, transform=self.transforms['test'])
        if sampleset is not None:
            self.sampleset = datasets.__dict__[sampleset](train=True, transform=self.transforms['train'])
        else:
            self.sampleset = None
        self.argmax = argmax
        self.batch_size = batch_size
        # For relative accuracy test.
        self.query = lambda data: query(self.blackbox, data, len(data), self.argmax, self.batch_size, self.device,
                                        self.topk)
        self.evaluation_set = query(self.blackbox, unpack(self.testset), len(self.testset), True, self.batch_size,
                                    self.device)
        self.num_classes = len(self.testset.classes)
        self.target_model = get_net(model_arch, modelfamily, pretrained=pretrained,
                                    channel=self.channel, complexity=complexity,
                                    num_classes=self.num_classes).to(self.device)
        self.optim = get_optimizer(self.target_model.parameters(), optimizer_choice, **kwargs)
        self.criterion = soft_cross_entropy
        self.batch_size = batch_size
        self.topk = topk
        self.num_workers = num_workers
        self.kwargs = kwargs

    def train(self, trainset: Union[Dataset, Iterable[Tuple[Tensor, Tensor]], Iterable[Tensor], Iterable[int]],
              to_query: bool = False):
        if self.blackbox is None and to_query:
            raise Exception("Blackbox didn't exists, couldn't query")
        elif to_query:
            if isinstance(trainset[0], Tuple):
                # This means trainset is either Dataset or Iterable[Tuple[Tensor, Tensor]], which needs to unpack.
                data = self.query(unpack(trainset))
            elif isinstance(trainset[0], Tensor):
                # This means trainset has already unpack.
                data = self.query(trainset)
            elif isinstance(trainset[0], int):
                assert self.sampleset is not None
                duplication = self.selection.intersection(trainset)
                if len(duplication) > 0:
                    print('{} samples duplicated.'.format(len(duplication)))
                difference = set(trainset) - duplication
                data = self.query([self.sampleset[i][0] for i in difference])
                self.selection.update(difference)
        else:
            # dataset is baked
            data = trainset
        self.transfer.extend(data)

        train_model(self.target_model, self.transfer, self.state_dir, self.batch_size, self.criterion,
                    testset=self.evaluation_set, device=self.device, num_workers=self.num_workers,
                    optimizer=self.optim, **self.kwargs)


def main():
    parser = argparse.ArgumentParser(description='Train a model in a distillation manner.')
    # Required arguments
    parser.add_argument('blackbox_path', metavar='VIC_DIR', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('state_dir', metavar='DIR', type=str, help='Directory storing state.')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testset', metavar='DS_NAME', type=str, help='Name of test dataset')
    parser.add_argument('train_path', metavar='PATH', type=str,
                        help="Either a transfer.pickle or directory containing images.")
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=0, metavar='N',
                        help='input batch size for training (default: all of containing sample)')
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
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    parser.add_argument('--argmax', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm',
                        choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    parser.add_argument('-x', '--complexity', type=int, default=64, metavar='N',
                        help="Model conv channel size.")
    parser.add_argument('--channel', '-c', type=int, help="Model input image channel", default=1)
    parser.add_argument('-k', '--topk', metavar='K', type=int, help='iteration times',
                        default=0)
    args = parser.parse_args()
    params = vars(args)
    train_path = params['train_path']
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        cuda = True
    else:
        cuda = False
    adversary = Adversary(cuda=cuda, **params)
    if os.path.isdir(train_path):
        samples = load_img_dir(train_path, adversary.transforms['train'])
    else:
        samples = pickle.load(train_path)
    adversary.train(samples, True)


if __name__ == '__main__':
    main()
