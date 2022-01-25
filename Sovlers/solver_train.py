import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Data import get_loader
from Models import get_model
from Losses import get_losses
from Metrics.metrics import epe_metric
from Metrics.metrics import tripe_metric
from Metrics.metrics import epe_metric_non_zero

import numpy as np
from matplotlib import pyplot as plt

class TrainSolver(object):

    def __init__(self, config):

        self.config = config
        self.cfg_solver = config['solver']
        self.cfg_dataset = config['data']
        self.cfg_model = config['model']
        self.reloaded = True if self.cfg_solver['resume_iter'] > 0 else False

        self.max_disp = self.cfg_model['max_disp']
        self.loss_name = self.cfg_model['loss']
        self.train_loader, self.val_loader = get_loader(self.config)
        self.model = get_model(self.config)

        self.crit = get_losses(self.loss_name, max_disp=self.max_disp, lcn_weight=self.cfg_solver['lcn_weight'], occluded_weight=self.cfg_solver['cross_entropy_weight'])

        if self.cfg_solver['optimizer_type'].lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.cfg_solver['lr_init'])
        elif self.cfg_solver['optimizer_type'].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg_solver['lr_init'])
        else:
            raise NotImplementedError('Optimizer type [{:s}] is not supported'.format(self.cfg_solver['optimizer_type']))
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg_solver['milestones'], gamma=self.cfg_solver['gamma'])
        self.global_step = 1
        self.save_val = self.cfg_solver['save_eval']

    def save_checkpoint(self):

        ckpt_root = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'models')
        
        if not os.path.exists(ckpt_root):
            os.makedirs(ckpt_root)
        
        ckpt_name = 'iter_{:d}.pth'.format(self.global_step)
        states = {
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        ckpt_full = os.path.join(ckpt_root, ckpt_name)
        
        torch.save(states, ckpt_full)
    
    def load_checkpoint(self):

        ckpt_root = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'models')

        ckpt_name = 'iter_{:d}.pth'.format(self.cfg_solver['resume_iter'])
        
        ckpt_full = os.path.join(ckpt_root, ckpt_name)

        states = torch.load(ckpt_full, map_location=lambda storage, loc: storage)

        self.global_step = states['global_step']
        self.model.load_state_dict(states['model_state'])
        self.optimizer.load_state_dict(states['optimizer_state'])
        self.scheduler.load_state_dict(states['scheduler_state'])

    def run(self):
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))

        if self.cfg_solver['resume_iter'] > 0:
            self.load_checkpoint()
            print('[{:d}] Model loaded.'.format(self.global_step))
        
        data_iter = iter(self.train_loader)
        brk_step = 5000000
        print_info = 1000
        tot_loss = 0.0
        tot_EPE_ref = 0.0
        tot_EPE_coarse = 0.0
        sample_weight = 1.0
        refine_weight = self.cfg_solver['refine_head_weight']
        self.optimizer.zero_grad()
        while True:
            try:
                data_batch = data_iter.next()
            except StopIteration:
                data_iter = iter(self.train_loader)
                data_batch = data_iter.next()

            if self.global_step > self.cfg_solver['max_steps']:
                break
            
            brk = False
            
            if self.global_step % brk_step == 0:
                brk = True

            sample_wei = sample_weight
            start_time = time.time()
            
            self.model.train()
            imgL, imgR, disp_L, _ = data_batch
            imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()
            

            disp_pred_ref_left, disp_pred_coarse_left, disp_pred_ref_right, disp_pred_coarse_right = self.model(imgL, imgR, disp_L, True)
            
            loss = (refine_weight * self.crit(imgL, imgR, disp_pred_ref_left, disp_pred_ref_right, disp_L, brk, sample_wei)) + ((1 - refine_weight) * self.crit(imgL, imgR, disp_pred_coarse_left, disp_pred_coarse_right, disp_L, False, sample_wei))
            loss_hist = loss.item()
            tot_loss += loss_hist
            loss /= self.cfg_solver['accumulate']
            loss.backward()
            
            if self.global_step % self.cfg_solver['accumulate'] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            elapsed = time.time() - start_time
            train_EPE_left_ref = epe_metric(disp_L.detach(), disp_pred_ref_left.detach(), self.max_disp)
            train_3PE_left = tripe_metric(disp_L.detach(), disp_pred_ref_left.detach(), self.max_disp)
            train_EPE_left_coarse = epe_metric(disp_L.detach(), disp_pred_coarse_left.detach(), self.max_disp)
           
            tot_EPE_ref += train_EPE_left_ref
            tot_EPE_coarse += train_EPE_left_coarse
            print(
                    '[{:d}/{:d}] Train Loss = {:.6f}, Avg. Train Loss = {:.3f}, EPE_ref = {:.3f} px, EPE_coarse = {:.3f}, Avg. EPE_ref = {:.3f} px, Avg. EPE_coarse = {:.3f} px, 3PE = {:.3f}%, time = {:.3f}s.'.format(
                    self.global_step, self.cfg_solver['max_steps'],
                    loss_hist,
                    tot_loss / ((self.global_step % print_info) + 1),
                    train_EPE_left_ref, 
                    train_EPE_left_coarse,
                    tot_EPE_ref / ((self.global_step % print_info) + 1),
                    tot_EPE_coarse / ((self.global_step % print_info) + 1),
                    train_3PE_left * 100,
                    elapsed
                ), end='\r'
            )
            self.scheduler.step()
              
            if self.global_step % self.cfg_solver['save_steps'] == 0 and not self.reloaded:
                self.save_checkpoint()
                print('')
                print('[{:d}] Model saved.'.format(self.global_step))
            
            
            if self.global_step % self.cfg_solver['eval_steps'] == 0 and not self.reloaded:
                elapsed = 0.0
                
                self.model.eval()
                with torch.no_grad():
                    EPE_metric_left = 0.0
                    val_EPE_metric_left = 0.0
                    val_TriPE_metric_left = 0.0
                    N_total = 0
                    valid = 1e-6
                    fig_root = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'results', 'val', str(self.global_step).zfill(6))
                    if self.save_val and (not os.path.exists(fig_root)):
                        os.makedirs(fig_root)

                    for val_batch in self.val_loader:
                        imgL, imgR, disp_L, _= val_batch
                        imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()

                        N_curr = imgL.shape[0]

                        start_time = time.time()
                        ref_pred_left, coarse_pred_left, _, _ = self.model(imgL, imgR, disp_L, False)
                        elapsed += (time.time() - start_time)
                        N_total += N_curr
                        is_valid = (disp_L > 0).float().mean() > 0.5

                        if is_valid:
                            if self.cfg_solver['refine_head_weight'] > -0.5:
                                EPE_metric_left = epe_metric_non_zero(disp_L, ref_pred_left, self.max_disp) * N_curr 
                                val_TriPE_metric_left += tripe_metric(disp_L, ref_pred_left, self.max_disp) * N_curr
                            else:
                                EPE_metric_left = epe_metric_non_zero(disp_L, coarse_pred_left, self.max_disp) * N_curr
                                val_TriPE_metric_left += tripe_metric(disp_L, coarse_pred_left, self.max_disp) * N_curr

                            val_EPE_metric_left += EPE_metric_left
                            valid += N_curr

                        if self.save_val:
                            fig, ax = plt.subplots(3,3)
                            fig.set_size_inches(10,10)
                            if is_valid:
                                fig.suptitle('EPE error: ' + str(EPE_metric_left))
                            else:
                                fig.suptitle('Invalid')
                            ax[0,0].imshow(imgL[0].permute(1,2,0).cpu().detach().numpy()); ax[0,0].set_title('IR0')
                            ax[0,1].imshow(imgR[0].permute(1,2,0).cpu().detach().numpy()); ax[0,1].set_title('IR1')
                            ax[0,2].imshow(disp_L[0,0].cpu().detach().numpy(), vmin=0, vmax=self.max_disp); ax[0,2].set_title('GT disparity')
                            ax[1,0].imshow(ref_pred_left[0,0].cpu().detach().numpy(), vmin=0, vmax=self.max_disp); ax[1,0].set_title('Refine head')
                            ax[1,1].imshow(coarse_pred_left[0,0].cpu().detach().numpy(), vmin=0, vmax=self.max_disp); ax[1,1].set_title('Coarse head')
                            ax[1,2].imshow(np.abs(ref_pred_left[0,0].cpu().detach().numpy() - coarse_pred_left[0,0].cpu().detach().numpy()), vmin=0, vmax=5); ax[1,2].set_title('Diff: Ref - Coarse')
                            ax[2,0].imshow(np.abs(ref_pred_left[0,0].cpu().detach().numpy() - disp_L[0,0].cpu().detach().numpy()), vmin=0, vmax=10); ax[2,0].set_title('Diff: Ref - GT')
                            ax[2,1].imshow(np.abs(coarse_pred_left[0,0].cpu().detach().numpy() - disp_L[0,0].cpu().detach().numpy()), vmin=0, vmax=10); ax[2,1].set_title('Diff: Coarse - GT')
                            ax[2,2].axis('off')
                            fig_nm = fig_root + '/' + str(N_total).zfill(5) + '.png'
                            fig.savefig(fig_nm)                           

                        if N_total % 1 == 0:
                            plt.close('all')
                        print(
                            '[{:d}/{:d}] Validation : valid = {:d}, EPE = {:.6f} px, Avg. EPE = {:.3f} px, Avg. 3PE = {:.3f} %, time = {:.3f} s.'.format(
                            N_total, len(self.val_loader),
                            int(valid),
                            EPE_metric_left,
                            val_EPE_metric_left / valid, 
                            val_TriPE_metric_left * 100 / valid, 
                            elapsed / N_total
                            ), end='\r'
                        )

                    plt.close('all')
                    val_EPE_metric_left /= valid
                    val_TriPE_metric_left /= valid

                    print(
                        '[{:d}/{:d}] Validation : valid = {:d}, EPE = {:.6f} px, 3PE = {:.3f} %, time = {:.3f} s.'.format(
                            N_total, len(self.val_loader),
                            int(valid),
                            val_EPE_metric_left, 
                            val_TriPE_metric_left * 100, 
                            elapsed / N_total
                        )
                    )

            if self.global_step % print_info == 0:
                print('Total updates: {:d}, Avg loss = {:.3f}, Avg. EPE_ref = {:.3f} px, Avg. EPE_coarse = {:.3f} px\n'.format(self.global_step, 
                    tot_loss / print_info,
                    tot_EPE_ref / print_info,
                    tot_EPE_coarse / print_info))
                tot_loss = 0.0
                tot_EPE_ref = 0.0
                tot_EPE_coarse = 0.0
            self.global_step += 1

            self.reloaded = False
