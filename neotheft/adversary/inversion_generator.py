# This module is meant to expand a inversion model outputs to a scale of valid training dataset

from torch import Tensor
from torch.nn import Module
import torch
import os
import numpy as np
import argparse

from models.zoo import get_net
from neotheft.utils.utility import augment, save_npimg


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
    parser.add_argument('expansion', metavar='E', type=int, help="Image expansion factor")
    parser.add_argument('--save-path', '-s', type=str, help="Path of generated image, optional")
    parser.add_argument('--channel', '-c', type=int, help="Inversion model output image channel", default=1)
    parser.add_argument('--num-classes', '-n', type=int, help="Inversion classifier input classes", default=10)
    parser.add_argument('--complexity', '-x', type=int, help="Inversion model conv channel size.", default=64)

    args = parser.parse_args()
    model = get_net('Inversion', 'custom_cnn', pretrained=args.modelpath, num_classes=args.num_classes,
                    channel=args.channel, complexity=args.complexity)

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = os.path.join(os.path.dirname(args.modelpath), 'generated')

    get_imgs(model, save_path, args.expansion, args.num_classes)


if __name__ == '__main__':
    main()

