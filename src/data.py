import json
import os

import numpy as np
import math
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate

import pdb


class RotatedMNIST(MNIST):
    """ If nrotations = None, will take to be random continuous rotations """
    def __init__(self, *args, nangles = 4, target_type='categorical', **kwargs):
        super(RotatedMNIST, self).__init__(*args, **kwargs)
        self.nangles = nangles
        self.target_type = target_type
        if nangles == 4:
            angles = [-90, 0, 90, 180]
        elif nangles == 8:
            angles = [-135, -90, -45, 0, 45, 90, 135, 180]
        else:
            angles = None
        self.angles = angles
        #pdb.set_trace()

    def _draw_angle(self):
        if self.angles is None:
            angle = (2*torch.rand(1) -1)*180
        else:
            angle = torch.tensor([np.random.choice(self.angles)])

        if self.target_type == 'degree': #[-180, 180]
            target = angle
        elif self.target_type == 'radian': #[-π, π]
            target = math.pi*angle/180.0
        elif self.target_type == 'scalar': #[-1, 1]
            target = angle/180.0
        elif self.target_type == 'categorical':
            target  = torch.tensor(self.angles.index(angle)).long()
            #print(target)
        return angle, target

    def __getitem__(self, i):
        input, _ = super(RotatedMNIST, self).__getitem__(i)
        #pdb.set_trace()
        angle, target = self._draw_angle()
        input = transforms.ToTensor()(rotate(input, angle))
        input = transforms.Normalize((0.1307,), (0.3081,))(input)
        return input, target


class AnnotatedMolecules(Dataset):
    def __init__(self, path_to_data, train_on, annotation, task, threshold, loss_functional):
        # Load annotated molecules json
        assert os.path.isfile(path_to_data), f'{path_to_data} not found.'
        annotated_molecules = torch.load(path_to_data)
        self.train_on = train_on
        if train_on == 'original':
            self._annotated_molecules = annotated_molecules
            self.annotation = annotation
        else:
            # Filter out invalid decoded SMILES
            self._annotated_molecules = [am for am in annotated_molecules if am['decoded_valid']]
            self.annotation = f'decoded_{annotation}'
        self.task = task
        self.threshold = threshold
        self.loss_functional = loss_functional
        print(f'\t Finished loading {path_to_data}...')

    def __getitem__(self, index):
        if self.task == 'classification':
            # For our setting, we flip positive and negative classes so that minimizing the functional will be convex
            if self.loss_functional == 'mse':
                label = -1 if self._annotated_molecules[index][self.annotation] >= self.threshold else 1
            else:  # For CE labels need to be 0-1
                label = 0 if self._annotated_molecules[index][self.annotation] >= self.threshold else 1
        else:
            label = self._annotated_molecules[index][self.annotation]
        return self._annotated_molecules[index]['embedding'], torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self._annotated_molecules)


class AnnotatedMoleculesDataModules(pl.LightningDataModule):
    def __init__(self, path_to_data, annotation, task, threshold, loss_functional, train_on,
                 batch_size=32, train_file=None, val_file=None, test_file=None):
        super().__init__()
        self.data_dir = path_to_data
        self.annotation = annotation
        self.task = task
        self.threshold = threshold
        self.loss_functional = loss_functional
        self.train_on = train_on
        self.batch_size = batch_size
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        print('In the setup phase...')
        if self.train_file:
            if stage == 'fit' or stage is None:
                assert self.val_file, 'If you pass a specific training data file, you need to pass a specific val data file.'
                print('Loading train and val...')
                self.train = AnnotatedMolecules(os.path.join(self.data_dir, self.train_file),
                                                train_on=self.train_on, annotation=self.annotation, task=self.task,
                                                threshold=self.threshold, loss_functional=self.loss_functional)
                self.val = AnnotatedMolecules(os.path.join(self.data_dir, self.train_file),
                                              train_on=self.train_on, annotation=self.annotation, task=self.task,
                                              threshold=self.threshold, loss_functional=self.loss_functional)
            if (stage == 'test' or stage is None) and self.test_file:
                print('Loading test...')
                self.test = AnnotatedMolecules(os.path.join(self.data_dir, self.test_file),
                                               train_on=self.train_on, annotation=self.annotation, task=self.task,
                                               threshold=self.threshold, loss_functional=self.loss_functional)
        else:
            data = AnnotatedMolecules(self.data_dir, train_on=self.train_on, annotation=self.annotation, task=self.task,
                                      threshold=self.threshold, loss_functional=self.loss_functional)
            self.train, self.val = random_split(data, [int(len(data) * 0.8) + 1, int(len(data) * 0.2)])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        if self.test:
            return DataLoader(self.test, batch_size=self.batch_size, num_workers=0, shuffle=False, pin_memory=True)
        else:
            return None
