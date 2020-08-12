# This module is meant to expand a inversion model outputs to a scale of valid training dataset

from torch import Tensor
from torch.nn import Module
import torch
import os
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image

from models.zoo import get_net
from neotheft.utils.utility import augment, save_npimg
import datasets
from knockoff.victim.blackbox import Blackbox


def get_vector(shape, index: int, noise: str = None) -> Tensor:
    if noise == 'gauss':
        vector = torch.from_numpy(np.abs(np.random.normal(0, 1e-5, shape).astype('Float32')))
    elif noise is None:
        vector = torch.zeros(shape)
    else:
        raise NotImplemented
    for i in range(shape[0]):
        vector[i][index] = np.random.normal(0.95, 0.1)
    return vector.clamp(0, 1)


def get_imgs(model: Module, path: str, quantity: int, num_classes: int = 10) -> None:
    if not os.path.exists(path):
        os.mkdir(path)
    elif not os.path.isdir(path):
        raise NotADirectoryError
    model.eval()
    with torch.no_grad():
        for i in range(num_classes):
            vector = get_vector([1, num_classes], i)
            img_vec = model(vector)
            img_array = augment(img_vec, quantity)
            for j in range(quantity):
                save_npimg(img_array[i], os.path.join(path, "{}_{}.bmp".format(i, j)))


def main():
    """Inversion Generator just works for Inversion model

    Argparse structure is different from other running profile.
    """
    parser = argparse.ArgumentParser(description='Generate inversion images')
    parser.add_argument('modelpath', metavar='P', type=str, help="Path of Inversion model")
    parser.add_argument('--expansion', '-e', metavar='E', type=int, help="Image expansion factor", default=200)
    parser.add_argument('--save-path', '-s', type=str, help="Path of generated image, optional")
    parser.add_argument('--channel', '-c', type=int, help="Inversion model output image channel", default=1)
    parser.add_argument('--num-classes', '-n', type=int, help="Inversion classifier input classes", default=10)
    parser.add_argument('--complexity', '-x', type=int, help="Inversion model conv channel size.", default=64)
    parser.add_argument('--blackbox', '-b', type=str, help="Full vector", default=None)
    parser.add_argument('--testset', metavar='DSET', type=str, help="If using full vector", default=None)
    parser.add_argument('-d', '--device-id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)

    args = parser.parse_args()
    model = get_net('Inversion', 'custom_cnn', pretrained=args.modelpath, num_classes=args.num_classes,
                    channel=args.channel, complexity=args.complexity)
    if args.device_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = os.path.join(os.path.dirname(args.modelpath), 'generated')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    if args.testset is None:
        get_imgs(model, save_path, args.expansion, args.num_classes)
    else:
        blackbox = Blackbox.from_modeldir(args.blackbox, device=device)
        assert args.testset in datasets.__dict__.keys()
        modelfamily = datasets.dataset_to_modelfamily[args.testset]
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        testset = datasets.__dict__[args.testset](train=False, transform=transform)
        results = []
        dataloader = DataLoader(testset, 128, False)
        total = 0
        img_vectors = []
        for inputs, targets in tqdm(dataloader):
            vector = blackbox(inputs)
            imgs = model(vector.to(device)).cpu()
            img_vectors.append(imgs)
            for i in range(imgs.shape[0]):
                img_vectors.append(imgs[i])
                # save_image(imgs[i], os.path.join(save_path, "{}.{}.bmp".format(targets[i], total + i)))
            total += imgs.shape[0]
        np.random.shuffle(img_vectors)
        for i in range(args.expansion):
            save_image(img_vectors[i], os.path.join(save_path, "{}.bmp".format(total + i)))



if __name__ == '__main__':
    main()
