"""
MIT License

Copyright (c) 2020 linjc16

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
import torch
import os
import time
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from Data import get_loader
from Models import get_model
from Losses import get_losses
from Metrics.metrics import epe_metric_non_zero
from Metrics.metrics import tripe_metric
from Utility.utility import map_gray_to_colour
from Utility.utility import put_text
from Sovlers.invalidation import LRInvalidation


class TestSolver(object):
    def __init__(self, config):
        self.config = config
        self.cfg_solver = config['solver']
        self.cfg_dataset = config['data']
        self.cfg_model = config['model']
        
        self.max_disp = self.cfg_model['max_disp']
        self.model = get_model(self.config)
        self.test_loader = get_loader(self.config)
        self.imshow = config['imshow']
        self.save_depth = config['save']
        self.video = config['video']
        self.lr_invalidation = config['lr_invalidation']
        self.res_root = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'results', 'val', str(self.cfg_solver['resume_iter']).zfill(10))        
        if self.video and (not os.path.exists(self.res_root+'/video')):
            os.makedirs(self.res_root+'/video')
        if self.save_depth and (not os.path.exists(self.res_root+'/depth')):
            os.makedirs(self.res_root+'/depth')
        if self.lr_invaldation:
            invalidation = LRInvalidation()

    def load_checkpoint(self):
        ckpt_root = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'models')
        ckpt_name = 'iter_{:d}.pth'.format(self.cfg_solver['resume_iter'])
        ckpt_full = os.path.join(ckpt_root, ckpt_name)
        states = torch.load(ckpt_full, map_location=lambda storage, loc: storage)

        self.model.load_state_dict(states['model_state'])

    def show_results(self, output, target):

        for i in range(output.shape[0]):
            
            outcmap = output[i]
            tarcmap = target[i]
            outcmap = outcmap.cpu().numpy().astype(np.uint8).squeeze()
            tarcmap = tarcmap.cpu().numpy().astype(np.uint8).squeeze()

            outcmap = cv2.applyColorMap(outcmap, cv2.COLORMAP_RAINBOW)
            tarcmap = cv2.applyColorMap(tarcmap, cv2.COLORMAP_RAINBOW)
            
            plt.figure(figsize=(640, 840))
            plt.subplot(1,2,1)
            plt.imshow(tarcmap)
            plt.axis('off')
            plt.title('G.T')
    

            plt.subplot(1,2,2)
            plt.imshow(outcmap)
            plt.axis('off')
            plt.title('Prediction')

            plt.show()

    def save_results(self, pred, mask):
        fNm = self.test_loader.dataset._get_seqname()
        img_nm = self.res_root + '/depth/' + fNm.replace('.png','_depth.png')

        pred = self.test_loader.dataset._disp_to_dep(pred)
        pred = pred[0,0].cpu().detach().numpy()
        pred = pred * 1000  # convert to meter
        cv2.imwrite(img_nm, pred.astype("uint16"))

        if mask is not None:
            mask = mask[0,0].cpu().detach().numpy()
            mask = mask * 255
            cv2.imwrite(img_nm.replace('_depth','_mask'), pred)

    def make_video(self, img, disp_gt, disp_pred, loss, N_total):

        _, _, h, w = img.shape
        img_txt = put_text(img[0].permute(1,2,0).cpu().detach().numpy() * 255, 'IR0')

        disp_gt_colour = map_gray_to_colour(disp_gt[0,0].cpu().detach().numpy(), 70)
        disp_gt_colour = put_text(disp_gt_colour, 'Realsense Disp.')
        disp_gt_colour = put_text(disp_gt_colour, 'Max Disp.: 70 px', 'top')

        disp_pred_colour = map_gray_to_colour(disp_pred[0,0].cpu().detach().numpy(), 70)
        disp_pred_colour = put_text(disp_pred_colour, 'Predicted Disp.')
        disp_pred_colour = put_text(disp_pred_colour, 'Max Disp.: 70 px', 'top')

        diff = (disp_pred - disp_gt).abs()
        diff_colour = map_gray_to_colour(diff[0,0].cpu().detach().numpy(), 10)
        diff_colour = put_text(diff_colour, 'Mean Abs. px error: ' + str(np.round(loss,3)))
        diff_colour = put_text(diff_colour, 'Max Disp.: 10 px', 'top')

        frame = np.zeros(((h * 2) + 10, (w * 2) + 10, 3)).astype('uint8')
        frame[:h, :w, :] = img_txt.astype('uint8')
        frame[:h, -w:, :] = disp_gt_colour
        frame[-h:, :w, :] = disp_pred_colour
        frame[-h:, -w:, :] = diff_colour
        img_nm = self.res_root + '/video/' + str(N_total).zfill(5) + '.png'        
        cv2.imwrite(img_nm, frame)


    def run(self):
        self.model = nn.DataParallel(self.model)
        self.model.cuda()

        if self.cfg_solver['resume_iter'] > 0:
            self.load_checkpoint()
            print('Model loaded.')
        
        self.model.eval()


        with torch.no_grad():
            elapsed = 0.0
            EPE_metric = 0.0
            TriPE_metric = 0.0
            N_total = 0
            for test_batch in self.test_loader:
                imgL, imgR, disp_L, _ = test_batch
                imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()

                N_curr = imgL.shape[0]
                if self.lr_invalidation:
                    start_time = time.time()
                    disp_pred, _, disp_predR, _ = self.model(imgL, imgR, disp_L, True)
                    torch.cuda.synchronize()
                    elapsed += (time.time() - start_time)
                    mask = invalidation(disp_pred, disp_predR)
                else:
                    start_time = time.time()
                    disp_pred, _, _, _ = self.model(imgL, imgR, disp_L, False)
                    torch.cuda.synchronize()
                    elapsed += (time.time() - start_time)
                    mask = None
                
                EPE_metric_curr = epe_metric_non_zero(disp_L, disp_pred, self.max_disp) * N_curr
                EPE_metric += EPE_metric_curr
                TriPE_metric += tripe_metric(disp_L, disp_pred, self.max_disp) * N_curr
                if self.imshow:
                    self.show_results(disp_pred, disp_L)
                if self.save_depth:
                    self.save_results(disp_pred, mask)
                if self.make_video:
                    self.make_video(imgL, disp_L, disp_pred, EPE_metric_curr.cpu().numpy(), N_total)

                N_total += N_curr
                print(
                            '[{:d}/{:d}] Validation : EPE = {:.6f} px, Avg. EPE = {:.3f} px, Avg. 3PE = {:.3f} %, time = {:.3f} s.'.format(
                            N_total, len(self.test_loader),
                            EPE_metric_curr,
                            EPE_metric / N_total, 
                            TriPE_metric * 100 / N_total, 
                            elapsed / N_total
                            ), end='\r'
                )
            EPE_metric /= N_total
            TriPE_metric /= N_total


        print(
            'Test: EPE = {:.6f} px, 3PE = {:.3f} %, time = {:.3f} s.'.format(
                EPE_metric, TriPE_metric * 100, elapsed / N_total
            )
        )




