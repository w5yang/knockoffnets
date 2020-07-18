import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


def img_range(tensor):
    return (tensor.min(), tensor.max())


def loader_subset(loader, num, _batch_size=50):
    '''Returns a loader subsets (first num of each label) '''
    from collections import Counter
    x_y = []
    c = Counter()
    for xb, yb in loader:
        for x, y in zip(xb, yb):
            if c[y] >= num:
                continue
            else:
                x_y += [(x, y)]
                c[y] += 1

    loader_subset = DataLoader(x_y, batch_size=_batch_size)
    return loader_subset


def unique(tensor1d):
    t, idx = np.unique(tensor1d.numpy(), return_inverse=True)
    class_names = {}

    for k in t:
        class_names[int(k)] = k
    return class_names


class SyntheticSampleCrafter(object):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1.):
        assert eps >= 1
        eps = eps / 255
        # normalize to pixel range
        self.eps = eps * (max_pixel - min_pixel)
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        return


class AdversarialExampleCrafter(SyntheticSampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        super().__init__(eps, min_pixel, max_pixel)
        self.targeted_attack = targeted_attack
        return

    def __call__(self, model: nn.Module, tensor: torch.Tensor, target: int, targeted_attack: bool, init_alpha=1.,
                 num_steps=40) -> torch.Tensor:
        raise NotImplementedError


# Random L_inf perturbation around tensor
class Unit(SyntheticSampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1.):
        super(Unit, self).__init__(eps, min_pixel, max_pixel)

    def __call__(self, model, tensor, target, init_alpha=1., num_steps=1):
        return tensor


# Fixed L_inf perturbation around tensor
class RandomColorPert(AdversarialExampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        super(RandomColorPert, self).__init__(eps, min_pixel, max_pixel, targeted_attack=True)

    def __call__(self, model, tensor, target, init_alpha=1., num_steps=1):
        assert tensor.dim() == 4
        # second dim should be color dim
        e = torch.zeros(tensor.shape)
        for i in range(tensor.shape[1]):
            e[:, i, :, :] = (2 * torch.rand(1).item() - 1) * self.eps

        if tensor.is_cuda:
            e = e.cuda()

        return torch.clamp(tensor + e, self.min_pixel, self.max_pixel)

    # Fixed L_inf perturbation around tensor


class IncreaseLuminosityPert(AdversarialExampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        super(IncreaseLuminosityPert, self).__init__(eps, min_pixel, max_pixel, targeted_attack=True)

    def __call__(self, model, tensor, target, init_alpha=1., num_steps=1):
        e = self.eps * torch.sign(torch.rand(tensor.shape))
        if tensor.is_cuda:
            e = e.cuda()

        return torch.clamp(tensor + e, self.min_pixel, self.max_pixel)

        # Random L_inf perturbation around tensor


class RandPert(AdversarialExampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        super(RandPert, self).__init__(eps, min_pixel, max_pixel, targeted_attack=True)

    def __call__(self, model, tensor, target, init_alpha=1., num_steps=1):
        e = self.eps * torch.sign(torch.randn(tensor.shape))
        if tensor.is_cuda:
            e = e.cuda()

        return torch.clamp(tensor + e, self.min_pixel, self.max_pixel)


class RandBoundPert(AdversarialExampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        super(RandBoundPert, self).__init__(eps, min_pixel, max_pixel, targeted_attack=True)

    def __call__(self, model, tensor, target, init_alpha=1., num_steps=1):
        r = self.max_pixel - self.min_pixel
        print(r)
        b = (r) * torch.rand(tensor.shape)
        print(b)
        b += self.min_pixel
        e = self.eps * b
        print(e)
        if tensor.is_cuda:
            e = e.cuda()

        return torch.clamp(tensor + e, self.min_pixel, self.max_pixel)


# Random Gaussian perturbation
class RandNormPert(AdversarialExampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        super(RandNormPert, self).__init__(eps, min_pixel, max_pixel, targeted_attack=True)

    def __call__(self, model, tensor, target, init_alpha=1., num_steps=1):
        e = self.eps * (torch.randn(tensor.shape))
        if tensor.is_cuda:
            e = e.cuda()
        return torch.clamp(tensor + e, self.min_pixel, self.max_pixel)


class FGSM(AdversarialExampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        super(FGSM, self).__init__(eps, min_pixel, max_pixel, targeted_attack=targeted_attack)

    def __call__(self, model, tensor, target, init_alpha=1., num_steps=1):
        model.eval()
        tensor_var = Variable(tensor, requires_grad=True)
        target_var = Variable(target, requires_grad=False)
        outputs = model(tensor_var)
        loss = F.nll_loss(outputs, target_var)

        loss.backward()  # obtain gradients on x
        x_grad_sign = torch.sign(tensor_var.grad.data)  # Add perturbation
        # gradient descent if targeted,
        if self.targeted_attack:
            return torch.clamp(tensor_var.data - self.eps * x_grad_sign, self.min_pixel, self.max_pixel)
        # gradient ascent if untargeted
        else:
            return torch.clamp(tensor_var.data + self.eps * x_grad_sign, self.min_pixel, self.max_pixel)


class PGD(AdversarialExampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        self.transform = lambda x: x
        super(PGD, self).__init__(eps, min_pixel, max_pixel, targeted_attack=targeted_attack)

    def __call__(self, model, tensor, target, init_alpha=1., num_steps=40):
        for name, param in model.state_dict().items():
            param.require_grads = False

        model.eval()

        x_adv_var = Variable(tensor, requires_grad=True)
        target_var = Variable(target, requires_grad=False)

        below_var = Variable(tensor - self.eps, requires_grad=False)
        above_var = Variable(tensor + self.eps, requires_grad=False)

        for i in range(num_steps):
            x_adv_var = Variable(x_adv_var.data, requires_grad=True)
            x_adv_var = self._step(model, x_adv_var, target_var, alpha=init_alpha, below_var=below_var,
                                   above_var=above_var)

        for name, param in model.state_dict().items():
            param.require_grads = True

        # return x_adv_var.cpu().data
        return x_adv_var.data

    def _step(self, model, tensor_var, target_var, alpha, below_var, above_var):
        outputs = model(tensor_var)
        loss = F.nll_loss(outputs, target_var)
        loss.backward(retain_graph=True)  # obtain gradients on x

        x_grad = tensor_var.grad  # Add perturbation
        # gradient descent if targeted,
        if self.targeted_attack:
            x_adv = tensor_var - alpha * x_grad
        # gradient ascent if untargeted
        else:
            x_adv = tensor_var + alpha * x_grad

        x_adv = torch.min(torch.max(x_adv, below_var), above_var)
        x_adv = torch.clamp(x_adv, self.min_pixel, self.max_pixel)
        return x_adv


class IFGSM(AdversarialExampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        self.transform = lambda x: x
        super(IFGSM, self).__init__(eps, min_pixel, max_pixel, targeted_attack=targeted_attack)

    def __call__(self, model: nn.Module, tensor: torch.Tensor, target: torch.Tensor, targeted_attack: bool,
                 init_alpha=1.,
                 num_steps=40) -> torch.Tensor:
        self.targeted_attack = targeted_attack
        for name, param in model.state_dict().items():
            param.require_grads = False

        model.eval()

        x_adv_var = Variable(tensor, requires_grad=True)
        target_var = Variable(target, requires_grad=False)

        below_var = Variable(tensor - self.eps, requires_grad=False)
        above_var = Variable(tensor + self.eps, requires_grad=False)

        for i in range(num_steps):
            x_adv_var = Variable(x_adv_var.data, requires_grad=True)
            x_adv_var = self._step(model, x_adv_var, target_var, alpha=init_alpha, below_var=below_var,
                                   above_var=above_var)

        for name, param in model.state_dict().items():
            param.require_grads = True

        # return x_adv_var.cpu().data
        return x_adv_var.data

    def _step(self, model: nn.Module, tensor_var: Variable, target_var: Variable, alpha: float, below_var: Variable,
              above_var: Variable) -> torch.Tensor:
        outputs = model(tensor_var)
        loss = F.nll_loss(outputs, target_var)
        loss.backward(retain_graph=True)  # obtain gradients on x

        x_grad = tensor_var.grad  # Add perturbation
        # gradient descent if targeted,
        if self.targeted_attack:
            x_adv = tensor_var - alpha * torch.sign(x_grad)
        # gradient ascent if untargeted
        else:
            x_adv = tensor_var + alpha * torch.sign(x_grad)

        x_adv = torch.min(torch.max(x_adv, below_var), above_var)
        x_adv = torch.clamp(x_adv, self.min_pixel, self.max_pixel)
        return x_adv


class IFGSMMod(AdversarialExampleCrafter):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., targeted_attack=True):
        self.transform = lambda x: x
        super(IFGSMMod, self).__init__(eps, min_pixel, max_pixel, targeted_attack=targeted_attack)

    def __call__(self, model: nn.Module, tensor: torch.Tensor, target: int, targeted_attack: bool, init_alpha=1.,
                 num_steps=40) -> torch.Tensor:
        self.targeted_attack = targeted_attack
        for name, param in model.state_dict().items():
            param.require_grads = False

        model.eval()

        x_adv_var = Variable(tensor, requires_grad=True)
        target_var = Variable(target, requires_grad=False)

        below_var = Variable(tensor - self.eps, requires_grad=False)
        above_var = Variable(tensor + self.eps, requires_grad=False)
        previous_label = model((x_adv_var.unsqueeze(0)).to(torch.device('cuda'))).data.max(1)[1]
        distant = False
        if previous_label == target:
            distant = True

        for i in range(num_steps):
            x_adv_var = Variable(x_adv_var.data, requires_grad=True)
            alt_adv_var = self._step(model, x_adv_var, target_var, alpha=init_alpha, below_var=below_var,
                                     above_var=above_var)
            model.eval()
            alt_sample_lbl = model((alt_adv_var.unsqueeze(0)).cuda())
            if torch.argmax(alt_sample_lbl).cpu() == target.cpu():
                if distant:
                    x_adv_var = alt_adv_var
                else:
                    x_adv_var = alt_adv_var
                    break
            elif torch.argmax(alt_sample_lbl).cpu() == previous_label.cpu():
                x_adv_var = alt_adv_var
            else:
                if distant:
                    break
                else:
                    init_alpha /= 2

        for name, param in model.state_dict().items():
            param.require_grads = True

        return x_adv_var.data

    def _step(self, model: nn.Module, tensor_var: Variable, target_var: Variable, alpha: float, below_var: Variable,
              above_var: Variable) -> torch.Tensor:
        outputs = model((tensor_var.unsqueeze(0)).to(torch.device('cuda')))
        loss = F.nll_loss(outputs, target_var)
        loss.backward(retain_graph=True)  # obtain gradients on x

        x_grad = tensor_var.grad  # Add perturbation
        # gradient descent if targeted,
        if self.targeted_attack:
            x_adv = tensor_var - alpha * torch.sign(x_grad)
        # gradient ascent if untargeted
        else:
            x_adv = tensor_var + alpha * torch.sign(x_grad)
        x_adv = torch.min(torch.max(x_adv, below_var), above_var)
        x_adv = torch.clamp(x_adv, self.min_pixel, self.max_pixel)
        return x_adv


# Abstract class
class TransferabilityAttack(object):
    def __init__(self, model, min, max, x_initial, num_channels, eps=64, is_cuda=False, targeted_attack=True):
        self.model = model
        self.min = min
        self.max = max

        assert eps >= 1
        eps = eps / 255 * (self.max - self.min)
        self.eps = eps

        self.num_channels = num_channels
        self.is_cuda = is_cuda
        self.x_initial = x_initial
        if self.is_cuda:
            self.x_initial = self.x_initial.device()

        # global lower/upper bounds
        self.lb, self.ub = self.calculate_bounds(self.x_initial)
        self.targeted_attack = targeted_attack

    def constrain(self, x_adv, below_var, above_var):

        x_adv.data = torch.max(x_adv.data, self.lb.data)
        x_adv = torch.min(x_adv, above_var)
        if self.num_channels == 3:
            for c in range(3):
                x_adv[:, c, :, :] = torch.clamp(x_adv[:, c, :, :], self.min[c], self.max[c])
        else:
            x_adv = torch.clamp(x_adv, self.min, self.max)
        return x_adv

    def upper_bound_var(self, tensor):
        return Variable(tensor + self.eps, requires_grad=False)

    def lower_bound_var(self, tensor):
        return Variable(tensor - self.eps, requires_grad=False)

    def calculate_bounds(self, tensor):  # , eps=None):
        assert tensor.dim() == 4
        return (self.lower_bound_var(tensor), self.upper_bound_var(tensor))

    def __call__(self):
        raise NotImplementedError()

    def to_cpu(self, *args):
        res = []
        for arg in args:
            res += [arg.cpu()]
        return tuple(res)

    def to_cuda(self, *args):
        res = []
        for arg in args:
            res += [arg.device()]
        return tuple(res)


class MIFGSM(TransferabilityAttack):
    def __init__(self, model, min, max, x_initial, num_channels, eps=64, is_cuda=False, momentum=1.0,
                 targeted_attack=True):
        super(MIFGSM, self).__init__(model, min, max, x_initial, num_channels, eps, is_cuda, targeted_attack)
        self.prev_grad = torch.zeros(x_initial.shape)
        if self.is_cuda:
            self.prev_grad = self.prev_grad.cuda()
        self.momentum = momentum

    def __call__(self, tensor, target, init_alpha, num_steps=10):
        if self.is_cuda:
            tensor, target = self.to_cuda(tensor, target)
        else:
            tensor, target = self.to_cpu(tensor, target)

        for name, param in self.model.state_dict().items():
            param.require_grads = False

        self.model.eval()

        x_adv_var = Variable(tensor, requires_grad=True)

        for i in range(num_steps):
            x_adv_var, target_var = Variable(x_adv_var.data, requires_grad=True), Variable(target, requires_grad=False)
            x_adv_var.data = self.step(x_adv_var, target_var, alpha=init_alpha).data

        for name, param in self.model.state_dict().items():
            param.require_grads = True

        return x_adv_var.data

    def step(self, tensor_var, target_var, alpha):
        outputs = self.model(tensor_var)
        loss = F.nll_loss(outputs, target_var)
        loss.backward(retain_graph=True)  # obtain gradients on x

        grad_increment = self.momentum * Variable(self.prev_grad, requires_grad=False)
        x_grad = grad_increment + tensor_var.grad / torch.norm(tensor_var.grad, 1)

        # create x_adv to mimic a class target_var
        if self.targeted_attack:
            x_adv = tensor_var - alpha * torch.sign(x_grad)
        # create x_adv to cause misclassification -- any other than target_var
        else:
            x_adv = tensor_var + alpha * torch.sign(x_grad)
        self.prev_grad = x_grad.data

        self.lb, self.ub = self.calculate_bounds(self.x_initial)
        x_adv = self.constrain(x_adv, self.lb, self.ub)

        return x_adv


def copy_config(config):
    import configparser
    copy = configparser.ConfigParser()
    for section in config:
        if section != 'DEFAULT':
            copy.add_section(section)
        for k, v in config[section].items():
            copy[section][k] = v
    return copy
