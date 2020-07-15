import numpy as np
import os

from neotheft.utils.deepfool import deepfool
from neotheft.utils.utility import parser_dealer, query, load_transferset
import knockoff.utils.model as model_utils
from knockoff.adversary.train import get_optimizer

def main():
    params = parser_dealer({
        'transfer': False,
        'active': False,
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
    iter_batch = 1000
    rounds = 20
    remnant = set(range(len(transferset)))
    selected = set()
    evaluation_set = query(blackbox, [data[0] for data in testset], len(testset), device=device)
    optimizer = get_optimizer(surrogate.parameters(), params['optimizer_choice'], **params)

    for i in range(rounds):
        batch_samples = []
        batch_permutation = []
        current_round = list(remnant)
        print('round {}: {} object to calculate.'.format(i + 1, len(remnant)))
        for index in current_round:
            permutation, _, _, result = deepfool(transferset[index], surrogate, num_classes)
            batch_samples.append(result)
            batch_permutation.append(np.linalg.norm(permutation))
        batch_permutation = np.array(batch_permutation)
        selection = batch_permutation.argsort(0)[:iter_batch]
        training_batch = [batch_samples[i] for i in selection]
        selection = [current_round[i] for i in selection]
        selected.update(selection)
        remnant.difference_update(selection)
        transferset.extend(query(blackbox, training_batch, iter_batch, device=device))
        model_utils.train_model(surrogate, transferset, model_dir, params['batch_size'],
                                criterion_train=model_utils.soft_cross_entropy, testset=evaluation_set,
                                optimizer=optimizer, **params)

if __name__ == '__main__':
    main()
