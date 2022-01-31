"""
MIT License

Copyright (c) 2022 SLAMcore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from .SLAMcore_helper import read_slamcore

class SLAMcoreDataset(Dataset):

    def __init__(self, data_root, npy_root, val_split, test_split, transform, phase):

        super(SLAMcoreDataset, self).__init__()

        self.data_root = data_root
        self.npy_root = npy_root
        self.phase = phase
        self.val_split = val_split
        self.test_split = test_split
        self.transform = transform
        self.min_depth = .05 # in meters
        self.max_depth = 4.0
        self.focal_length = 4.2711225733701440e+02
        self.baseline = 0.05008 # in meters
        self.curr_index = 0 # current_index 
        self.crop = True

        self.left_imgs, self.right_imgs, self.deps, self.test_left_imgs, self.test_right_imgs, self.test_deps = read_slamcore(self.data_root)

        assert len(self.left_imgs) == len(self.right_imgs) == len(self.deps), 'Invalid training dataset!'
        assert len(self.test_left_imgs) == len(self.test_right_imgs) == len(self.test_deps), 'Invalid testing dataset!'

        test_data_num = len(self.test_left_imgs)
        
        self.nb_train = len(self.left_imgs)
        self.nb_val = int(self.val_split * test_data_num)
        self.nb_test = test_data_num
        
        train_npy = os.path.join(self.npy_root, 'train.npy')
        val_npy = os.path.join(self.npy_root, 'val.npy')
        test_npy = os.path.join(self.npy_root, 'test.npy')

        test_idcs = np.random.permutation(test_data_num)
        self.val_list = test_idcs[0:self.nb_val]
        self.val_list.sort()
        
    def __len__(self):

        if self.phase == 'train':
            return self.nb_train
        elif self.phase == 'val':
            return self.nb_val
        elif self.phase == 'test':
            return self.nb_test


    def __getitem__(self, index):
        
        self.curr_index = index
        if self.phase == 'train':
            left_image = self._read_image(self.left_imgs[index])
            right_image = self._read_image(self.right_imgs[index])
            left_disp = self._read_as_disp(self.deps[index])

        elif self.phase == 'val':
            index = self.val_list[index]
            left_image = self._read_image(self.test_left_imgs[index])
            right_image = self._read_image(self.test_right_imgs[index])
            left_disp = self._read_as_disp(self.test_deps[index])

        elif self.phase == 'test':
            left_image = self._read_image(self.test_left_imgs[index])
            right_image = self._read_image(self.test_right_imgs[index])
            left_disp = self._read_as_disp(self.test_deps[index])
            
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        
        left_disp = torch.Tensor(left_disp)
        right_disp = torch.Tensor(left_disp)
        return left_image, right_image, left_disp, right_disp

    
    def _read_image(self, filename):

        attempt = True
        while attempt:
            try:
                with open(filename, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                attempt = False
            except IOError as e:
                print('[IOError] {}, keep trying...'.format(e))
                attempt = True
        img = np.asarray(img)
        if self.crop:
            img = img[:, 16:, :]                
        return img

    def _read_as_disp(self, filename):

        attempt = True
        while attempt:
            try:
                with open(filename, 'rb') as f:
                    dep = Image.open(f).convert('F')
                attempt = False
            except IOError as e:
                print('[IOError] {}, keep trying...'.format(e))
                attempt = True
        dep = np.asarray(dep) / 1000.0 # convert mm to meters
        if self.crop:
            dep = dep[:, 16:]        
        disp = self._dep_to_disp(dep)
        return disp

    def _get_seqname(self):
        if self.phase == 'train':
            seq = (self.left_imgs[self.curr_index].split('/')[-4] + '_' + self.left_imgs[self.curr_index].split('/')[-1])
        elif self.phase == 'test':
            seq = (self.test_left_imgs[self.curr_index].split('/')[-4] + '_' + self.test_left_imgs[self.curr_index].split('/')[-1])
        return seq

    def _dep_to_disp(self, dep):

        mask = (dep > self.min_depth) * (dep < self.max_depth)

        disp = np.zeros_like(dep)
        disp[mask] = self.focal_length * self.baseline / dep[mask]
        disp = disp[np.newaxis,:,:].copy()
        return disp

    def _disp_to_dep(self, disp):
        if type(disp) != torch.Tensor:
            raise NotImplementedError('Disparity needs to be a torch tensor')
        mask = disp > 0.0

        dep = torch.zeros_like(disp)
        dep[mask] = self.focal_length * self.baseline / disp[mask]
        return dep
