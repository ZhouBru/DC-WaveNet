import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import sklearn

from sklearn.mixture import GaussianMixture
from loss import ASDLoss
import utils
from torch.utils.data import DataLoader
from dataset import ASDDataset


class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.criterion = ASDLoss().to(self.args.device)
        self.transform = kwargs['transform']
        self.batch_size =self.args.batch_size
        self.test_batch_size =64
        
        
    def train(self, train_loader):
        # self.test(save=False)
        model_dir = os.path.join(self.writer.log_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        start_valid_epoch = self.args.start_valid_epoch
        
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        self.test_batch_size =64
        best_metric = 0
        no_better_epoch = 0

        for epoch in range(0, epochs + 1):
            # train
            sum_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')
            for (x_wavs, x_mels, labels) in train_bar:
                # forward
                x_wavs, x_mels = x_wavs.float().to(self.args.device), x_mels.float().to(self.args.device)
                labels = labels.reshape(-1).long().to(self.args.device)
                logits, _ = self.net(x_wavs, x_mels, labels)

                loss = self.criterion(logits, labels)
                train_bar.set_postfix(loss=f'{loss.item():.5f}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # visualization
                self.writer.add_scalar(f'train_loss', loss.item(), self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1
            avg_loss = sum_loss / num_steps
            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            if (epoch - start_valid_epoch) % valid_every_epochs == 0 and epoch >= start_valid_epoch:
                avg_auc, avg_pauc = self.test(save=False, gmm_n=False)
                self.writer.add_scalar(f'auc', avg_auc, epoch)
                self.writer.add_scalar(f'pauc', avg_pauc, epoch)
                if avg_auc + avg_pauc >= best_metric:
                    no_better_epoch = 0
                    best_metric = avg_auc + avg_pauc
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                    self.logger.info(f'Best epoch now is: {epoch:4d}')
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break
            # save last 10 epoch state dict
            if epoch >= self.args.start_save_model_epochs:
                if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                    model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                    utils.save_model_state_dict(model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                    
    
   
    def test(self):
        sum_auc, sum_pauc, num = 0, 0, 0
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(self.args.test_dirs), sorted(self.args.train_dirs))):
            machine_type = target_dir.split('/')[-2]
            # result csv
            performance = []
            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                meta = machine_type + '-' + id_str
                label = self.args.meta2label[meta]
                test_files, y_true = utils.create_test_file_list(target_dir, id_str, dir_name='test')
                test_dataset =ASDDataset(self.args, test_files, load_in_memory=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=24)
                anomaly_score_list = []
                y_pred = []
                testbar = tqdm(test_loader, total=len(test_loader), ncols=100)
                for (x_wavs, x_mels, labels) in testbar:
                    with torch.no_grad():
                        x_wavs, x_mels = x_wavs.float().to(self.args.device), x_mels.float().to(self.args.device)
                        labels = labels.reshape(-1).long().to(self.args.device)
                        predict_ids, _ = net(x_wavs, x_mels, labels)
                    probs = - torch.log_softmax(predict_ids, dim=1).squeeze().cpu().numpy()
                    for i, ids in enumerate(labels.cpu().numpy()):
                        y_pred.append(probs[i][ids])
                    testbar.set_description('Validating') 
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]            
            self.logger.info(f'{machine_type}\t\tAUC: {mean_auc*100:.3f}\tpAUC: {mean_p_auc*100:.3f}')
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1
        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        self.logger.info(f'Total average:\t\tAUC: {avg_auc*100:.3f}\tpAUC: {avg_pauc*100:.3f}')
        result_path = os.path.join(result_dir, 'result.csv')
        return avg_auc, avg_pauc

