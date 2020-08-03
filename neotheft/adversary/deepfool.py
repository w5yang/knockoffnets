import numpy as np
import os
from tqdm import tqdm

from neotheft.utils.deepfool import deepfool
from neotheft.utils.utility import parser_dealer, query, load_transferset, save_selection_state
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

def deepfool_choose():
    params = parser_dealer({
        'transfer': False,
        'active': False,
        'sampling': True,
        'synthetic': False,
        'black_box': True,
        'train': True
    })
    model_dir = params['model_dir']
    transferset, num_classes = load_transferset(os.path.join(model_dir, 'transferset.7500.pickle'))
    surrogate = params['surrogate']
    blackbox = params['blackbox']
    device = params['device']
    testset = params['testset']
    selected = params['selected']
    params['testset'] = query(blackbox, [data[0] for data in testset], len(testset), device=device, argmax=True)
    optimizer = get_optimizer(surrogate.parameters(), params['optimizer_choice'], **params)
    # todo: make deepfool judge budget criteria direction part of parameter.
    reverse = True
    budget = 5000

    queryset = params['queryset']

    batch_samples = []
    batch_permutation = []
    total = set([i for i in range(len(queryset))])
    unselected = list(total - selected)
    num_classes = len(testset.classes)
    print('{} object to calculate.'.format(len(unselected)))
    for index in tqdm(unselected):
        permutation, _, _, _, result = deepfool(queryset[index][0], surrogate, num_classes)
        batch_samples.append(result.squeeze(0))
        batch_permutation.append(np.linalg.norm(permutation))
    batch_permutation = np.array(batch_permutation)
    selection = batch_permutation.argsort(0)[:budget] if not reverse else batch_permutation.argsort(0)[len(unselected) - budget:]
    assert len(selection) == budget
    training_batch = [batch_samples[i] for i in selection]
    selection = [unselected[i] for i in selection]
    selected.update(selection)
    transferset.extend(query(blackbox, training_batch, budget, device=device))
    model_utils.train_model(surrogate, transferset, model_dir,
                            criterion_train=model_utils.soft_cross_entropy,
                            optimizer=optimizer, checkpoint_suffix='deepfool', **params)
    save_selection_state(transferset, selected, model_dir)

def main():
    deepfool_choose()


if __name__ == '__main__':
    main()
