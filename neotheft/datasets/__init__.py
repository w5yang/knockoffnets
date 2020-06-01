from torchvision import transforms

from neotheft.datasets.gtsrb import GTSRB
from neotheft.datasets.imagenet64 import ImageNet64

# Create a mapping of dataset -> dataset_type
# This is helpful to determine which (a) family of model needs to be loaded e.g., imagenet and
# (b) input transform to apply

dataset_to_modelfamily = {
    'GTSRB': 'custom_cnn',
    'Imagenet64': 'custom_cnn'
}

# Transforms
modelfamily_to_transforms = {
    'custom_cnn': {
        'train': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    }
}
