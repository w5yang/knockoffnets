import torch
from typing import List, Tuple
from torch import Tensor
import os.path as osp
import time
from datetime import datetime
from collections import defaultdict as dd

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import knockoff.utils.utils as knockoff_utils
from knockoff.adversary.train import get_optimizer

from neotheft.utils.utility import parser_dealer, query, load_transferset


def train_step(model, train_loader, criterion, optimizer, epoch, device, log_interval=10):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        targets, inputs = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        total += targets.size(0)
        # if len(targets.size()) == 2:
        #     # Labels could be a posterior probability distribution. Use argmax as a proxy.
        #     target_probs, target_labels = targets.max(1)
        # else:
        #     target_labels = targets
        # correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        # acc = 100. * correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f})'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    # acc = 100. * correct / total

    return train_loss_batch


def test_step(model, test_loader, criterion, device, epoch=0., silent=False):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            targets, inputs = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)

    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}'.format(epoch, test_loss))

    return test_loss


def train_inversion(model, trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                    device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                    epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None,
                    scheduler=None,
                    **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_test = nn.MSELoss(reduction='mean')
        # criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.MSELoss(reduction='mean')
        # criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_loss, train_acc = 999., -1.
    best_recon_loss, test_acc, test_loss = 999, -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'least loss']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        scheduler.step(epoch)
        train_loss = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        best_train_loss = min(best_train_loss, train_loss)

        if test_loader is not None:
            test_loss = test_step(model, test_loader, criterion_test, device, epoch=epoch)
            best_recon_loss = min(best_recon_loss, test_loss)

        # Checkpoint
        if test_loss <= best_recon_loss:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, best_train_loss]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, best_recon_loss]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model


def inversion():
    params = parser_dealer({
        'transfer': False,
        'active': False,
        'sampling': False,
        'synthetic': False,
        'black_box': True,
        'train': True
    })
    blackbox = params['blackbox']
    device = params['device']
    testset = params['testset']
    testset = query(blackbox, [data[0] for data in testset], len(testset), device=device)
    inversion = params['surrogate']
    model_dir = params['model_dir']
    # ignore parameter optimizer_choice
    optim = torch.optim.Adam(inversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)
    transferset, num_classes = load_transferset(osp.join(model_dir, 'transferset.pickle'))
    train_inversion(inversion, transferset, model_dir, optimizer=optim, batch_size=params['batch_size'], testset=testset, device=device, epochs=params['epochs'], checkpoint_suffix='inversion')

if __name__ == '__main__':
    inversion()