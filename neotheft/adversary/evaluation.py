import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from typing import List, Tuple, Dict
from torch.nn import Module
from torch import Tensor

from neotheft.utils.synthetic_sample_crafter import SyntheticSampleCrafter, FGSM, IFGSM, MIFGSM
from knockoff.victim.blackbox import Blackbox


def transferability(blackbox: Blackbox,
                    surrogate: Module,
                    data: List[Tuple[Tensor, int]],
                    method: str = "ifgsm",
                    targeted: bool = False,
                    targets_dict: Dict = None,
                    option: Dict = None,
                    batch_size: int = 64,
                    num_workers: int = 8) -> float:
    if option is None:
        option = {
            "eps": 64,
            "min_pixel": 0.0,
            "max_pixel": 1.0,
            "is_cuda": True
        }
    cuda = lambda x: x.cuda() if option["is_cuda"] else lambda x: x.cpu()
    surrogate = cuda(surrogate)
    if targets_dict is None:
        targets_dict = dict()
        sample = data[0][0]
        surrogate.eval()
        sample_result = surrogate(cuda(sample.unsqueeze(0)))
        num_classes = sample_result.shape[1]
        for i in range(num_classes):
            targets_dict[i] = i
    assert method in ("ifgsm", "mifgsm")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if method == "ifgsm":
        crafter = IFGSM(targeted_attack=targeted, **option)
    elif method == "mifgsm":
        crafter = MIFGSM(targeted_attack=targeted, **option)
    else:
        raise NotImplementedError

    num_steps = 40
    initial_alpha = option['eps'] / num_steps

    total = len(data)
    agreement = 0
    transfer = 0

    for inputs, _ in loader:
        # surrogate output
        outputs_sur = surrogate(Variable(cuda(inputs), requires_grad=False))
        labels_sur = torch.max(outputs_sur, 1)[1]
        # blackbox output
        outputs_bb = blackbox(cuda(inputs))
        labels_bb = torch.max(outputs_bb, 1)[1]
        # target directions
        targets = torch.tensor([targets_dict[int(i)] for i in labels_sur])
        x_adv = crafter(surrogate, inputs, targets, initial_alpha, num_steps)
        adv_output_sur = surrogate(Variable(cuda(x_adv), requires_grad=False))
        adv_labels_sur = adv_output_sur.max(1)[1]
        adv_output_bb = blackbox(cuda(x_adv))
        _, adv_labels_bb = torch.max(adv_output_bb, 1)
        agreement += torch.sum(labels_bb.cpu() == labels_sur.cpu()).int()
        transfer += torch.sum(adv_labels_bb.cpu() == targets.cpu()).int() if targeted else torch.sum(adv_labels_bb.cpu() != targets.cpu()).int()

    print("Agreement: {}".format(agreement / total))
    print("Transferability: {}".format(transfer / total))
    return transfer / total

if __name__ == '__main__':
    # this block of code is only for temporary test.
    from datasets import GTSRB
    from datasets import modelfamily_to_transforms
    transform = modelfamily_to_transforms['custom_cnn']['train']
    dataset = GTSRB(False, transform)
    from knockoff.victim.blackbox import Blackbox
    import torch
    device = torch.device('cuda')
    blackbox = Blackbox.from_modeldir('results/models/victim/gtsrb', device)
    from models import zoo
    surrogate = zoo.get_net('CNN32','custom_cnn', 'results/models/adversary/manhattan/checkpoint.28.iter.pth.tar', num_classes=43)
    transfer = transferability(blackbox, surrogate, dataset, targeted=False)