from __future__ import annotations
from warnings import warn

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms


class CSMNISTDataset(Dataset):
    """
    A torch Dataset for CSMNIST.
    
    Args:
        root (str): Root directory where the dataset is saved.
        train (bool): If True, creates the dataset from the training set, otherwise from the test set.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        sequences (list, optional): A list of pre-generated sequences. If not provided, generator must be provided.
        generator (Generator, optional): A generator to use for generating sequences on the fly. Ignored if sequences is provided.
        seed (int, optional): Seed for the random number generator.

    """
    def __init__(
        self,
        mnist_root,
        train=True,
        mnist_download=True,
        transform=None,
        sequences=None,
        generator=None,
        seed=None
    ):
        self.mnist = MNIST(mnist_root, train=train, download=mnist_download)
        self.transform = transform or transforms.ToTensor()
        self.sequences = sequences
        self.generator = generator

        self.digits = self._group_by_digit()
        self.rng = torch.Generator().manual_seed(seed)

        if self.sequences is None and self.generator is None:
            raise ValueError("For sequences to be generated on the fly, a generator must be provided.")

        if self.sequences is not None and self.generator is not None:
            # raise ValueError("Only one of `sequences` or `generator` should be provided.")
            warn("Only one of `sequences` or `generator` should be provided.")

    def _group_by_digit(self):
        digits = [[] for _ in range(10)]
        for img, label in self.mnist:
            digits[label].append(img)

        self.num_digits = [len(d) for d in digits]
        return digits

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        if self.sequences:
            sequence = self.sequences[idx]
        else:
            sequence = self.generator.generate()
        
        images = list()
        for d in sequence:
            random_idx = torch.randint(0, self.num_digits[d], (1,), generator=self.rng)
            images.append(self.digits[d][random_idx])

        stacked_image = torch.cat([self.transform(img) for img in images], dim=2)
        label = torch.tensor(sequence)
        
        return stacked_image, label