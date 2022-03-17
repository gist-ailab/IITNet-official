import os
import torch
import numpy as np
from torch.utils.data import Dataset


class EEGDataLoader(Dataset):

    def __init__(self, config, fold, mode='train'):

        self.mode = mode
        self.fold = fold

        self.sr = 100
        self.config = config
        self.dataset = config['dataset']
        self.seq_len = config['seq_len']
        self.target_idx = config['target_idx']
        self.signal_type = config['signal_type']
        self.n_splits = config['n_splits']

        self.dataset_path = os.path.join('./datasets', self.dataset)
        self.inputs, self.labels, self.epochs = self.split_dataset()
        
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        n_sample = 30 * self.sr * self.seq_len
        file_idx, idx, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][idx:idx+seq_len]

        #inputs = inputs.reshape(1, n_sample)
        inputs = torch.from_numpy(inputs).float()
        
        labels = self.labels[file_idx][idx:idx+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]
        
        return inputs, labels

    def split_dataset(self):

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.signal_type)
        data_fname_list = sorted(os.listdir(data_root))
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load('idx_{}.npy'.format(self.dataset))
        
        assert len(split_idx_list) == self.n_splits
        
        if self.dataset == 'Sleep-EDF':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx == self.fold - 1:
                    data_fname_dict['test'].append(data_fname_list[i])
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict['val'].append(data_fname_list[i])
                else:
                    data_fname_dict['train'].append(data_fname_list[i])
                    
        elif self.dataset == 'MASS' or self.dataset == 'SHHS':
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.mode]:
                    data_fname_dict[self.mode].append(data_fname_list[i])
                else:
                    raise NotImplementedError("idx '{}' does not exist in split idx list".format(i))
            
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))
        
        for data_fname in data_fname_dict[self.mode]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            for i in range(len(npz_file['y']) - self.seq_len + 1):
                epochs.append([file_idx, i, self.seq_len])
            file_idx += 1
        
        return inputs, labels, epochs
