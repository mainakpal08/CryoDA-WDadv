import numpy as np
import h5py
import torch
import shutil


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_h5dataset(opt)

    def read_h5dataset(self, opt):

        self.numclass = opt.numclass

        #-----------------------SOURCE-----------------------#
        #train
        with h5py.File(opt.source_train_path, 'r') as scan_h5:
            source_train_subtom = scan_h5['subtom'][:]
            source_train_label = scan_h5['label'][:]

        self.source_train_subtom = torch.from_numpy(source_train_subtom).float()
        self.source_train_label = torch.from_numpy(source_train_label).long()
        self.source_ntrain = len(self.source_train_subtom)
        
        #test
        with h5py.File(opt.source_test_path, 'r') as scan_h5:
            source_test_subtom = scan_h5['subtom'][:]
            source_test_label = scan_h5['label'][:]
    
        self.source_test_subtom = torch.from_numpy(source_test_subtom).float()
        self.source_test_label = torch.from_numpy(source_test_label).long()
        self.source_ntest = len(self.source_test_subtom)


        #-----------------------TARGET-----------------------#
        #train
        with h5py.File(opt.target_train_path, 'r') as scan_h5:
            target_train_subtom = scan_h5['subtom'][:]
            target_train_label = scan_h5['label'][:]

        self.target_train_subtom = torch.from_numpy(target_train_subtom).float()
        self.target_train_label = torch.from_numpy(target_train_label).long()
        self.target_ntrain = len(self.target_train_subtom)

        #test
        with h5py.File(opt.target_test_path, 'r') as scan_h5:
            target_test_subtom = scan_h5['subtom'][:]
            target_test_label = scan_h5['label'][:]
    
        self.target_test_subtom = torch.from_numpy(target_test_subtom).float()
        self.target_test_label = torch.from_numpy(target_test_label).long()
        self.target_ntest = len(self.target_test_subtom)


    def source_next_batch(self, batch_size):
        idx = torch.randperm(self.source_ntrain)[0:batch_size]
        batch_tomo = self.source_train_subtom[idx]
        batch_label = self.source_train_label[idx]

        return batch_tomo, batch_label

    def target_next_batch(self, batch_size):
        idx = torch.randperm(self.target_ntrain)[0:batch_size]
        batch_tomo = self.target_train_subtom[idx]
        batch_label = self.target_train_label[idx]

        return batch_tomo, batch_label


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

