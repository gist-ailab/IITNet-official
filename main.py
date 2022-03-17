import os
import json
import argparse
import warnings
import matplotlib
import numpy as np
import sklearn.metrics as skmet

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils import *
from models.main_models import *
from loader import EEGDataLoader


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(os.path.basename(args.config)))

        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_path = os.path.join('checkpoints', os.path.basename(args.config))
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=config['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.config['early_stopping_mode'])
        
        self.dataset_args = {'config': self.config, 'fold': self.fold}
        self.dataloader_args = {'batch_size': self.config['batch_size'], 'num_workers': 4*len(self.args.gpu.split(","))}

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
    def build_model(self):
        model = MainModel(self.config)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        train_dataset = EEGDataLoader(mode='train', **self.dataset_args)
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, **self.dataloader_args)
        val_dataset = EEGDataLoader(mode='val', **self.dataset_args)
        val_loader = DataLoader(dataset=val_dataset, shuffle=True, **self.dataloader_args)
        test_dataset = EEGDataLoader(mode='test', **self.dataset_args)
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, **self.dataloader_args)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    def train_one_epoch(self, epoch):
        self.model.train()
        correct, total, train_loss = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()

            progress_bar(i, len(self.loader_dict['train']), 'Lr: %.4e | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (get_lr(self.optimizer), train_loss / (i + 1), 100. * correct / total, correct, total))
            
    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.config['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            eval_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), 'Lr: %.4e | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (get_lr(self.optimizer), eval_loss / (i + 1), 100. * correct / total, correct, total))

        if mode == 'val':
            return 100. * correct / total, eval_loss
        elif mode == 'test':
            return y_true, y_pred
        else:
            raise NotImplementedError
    
    def run(self):
        for epoch in range(self.config['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            val_acc, val_loss = self.evaluate(mode='val')
            self.early_stopping(val_acc, val_loss, self.model)
            if self.early_stopping.early_stop:
                break
        
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred = self.evaluate(mode='test')
        print('')

        return y_true, y_pred

def main():
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, default="IITNetV2", help='.json')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open('configs/' + args.config + '.json') as config_file:
        config = json.load(config_file)
    config['config_name'] = os.path.basename(args.config)

    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['num_classes']))

    for fold in range(1, config['n_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
    
        summarize_result(config, fold, Y_true, Y_pred)
    

if __name__ == "__main__":
    main()
