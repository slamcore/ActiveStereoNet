import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from .SLAMcore import SLAMcoreDataset
from .TartanAir import TartanAirDataset

def get_loader(config):

    dset = config['dataset_name'].lower()
    if dset in ['slamcore', 'tartanair']:
        return get_dataset_loader(config, dset)
    else:
        raise NotImplementedError('Dataset [{:s}] is not supported.'.format(dset))

def get_dataset_loader(config, dset):

    cfg_mode = config['mode'].lower()
    if cfg_mode == 'train':
        train_loader = DataLoader(
            create_dataset(config['data'], 'train', dset),
            batch_size=config['solver']['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            create_dataset(config['data'], 'val', dset),
            batch_size=config['solver']['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        return train_loader, val_loader
    elif cfg_mode == 'test':
        test_loader = DataLoader(
            create_dataset(config['data'], 'test', dset),
            batch_size=config['solver']['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        return test_loader
    else:
        raise NotImplementedError('Mode [{:s}] is not supported.'.format(cfg_mode))

def create_dataset(cfg_data, mode, dset):
    
    data_root = cfg_data['data_root']
    npy_root = cfg_data['npy_root']
    test_split = cfg_data['test_split']
    val_split = cfg_data['val_split']
    
    if dset == 'slamcore':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        ])
        return SLAMcoreDataset(data_root, npy_root, val_split, test_split, transform, mode)
    elif dset == 'tartanair':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        ])
        return TartanAirDataset(data_root, npy_root, val_split, test_split, transform, mode)
    else:
        raise NotImplementedError("Can't create dataset [{:s}].".format(dset))
