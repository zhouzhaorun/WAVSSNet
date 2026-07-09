import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from skimage.metrics import peak_signal_noise_ratio as psnr
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.postion_embedding = None

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
            
        if train_opt.get('mwmse_opt'):
            mwmse_type = train_opt['mwmse_opt'].pop('type')
            cri_mwmse_cls = getattr(loss_module, mwmse_type) # L1Loss
            self.cri_mwmse = cri_mwmse_cls(**train_opt['mwmse_opt']).to(self.device)
        else:
            raise ValueError('mwmse loss are None.')
        
        if train_opt.get('fft_opt'):
            fft_type = train_opt['fft_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type) # L1Loss
            self.cri_fft = cri_fft_cls(**train_opt['fft_opt']).to(self.device)
        else:
            raise ValueError('fft loss are None.')


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.comp = data['comp'].to(self.device)
        self.mask = data['mask'].to(self.device)
        if 'real' in data:
            self.real = data['real'].to(self.device)

    def feed_data(self, data):
        self.comp = data['comp'].to(self.device)
        self.mask = data['mask'].to(self.device)
        if 'real' in data:
            self.real = data['real'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.comp, self.mask) 
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1] * self.mask + (1 - self.mask) * self.comp

        loss_dict = OrderedDict()

        # mwmse loss
        l_mwmse = 0.
        for pred in preds:
            l_mwmse += self.cri_mwmse(pred, self.real, self.mask)
        loss_dict['cri_mwmse'] = l_mwmse

        # fft loss
        l_fft = 0.
        for pred in preds:
            l_fft += self.cri_fft(pred, self.real)
        loss_dict['cri_fft'] = l_fft

        l_total = l_mwmse + l_fft
        l_total.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.comp.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.comp, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        msk = F.pad(self.mask, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img, msk)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None, mask=None):
        if img is None:
            img = self.comp 
            mask = self.mask  

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img, mask)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img, mask)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        # if window_size:
        #     test = partial(self.pad_test, window_size)
        # else:
        test = self.nonpad_test

        cnt = 0

        os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name), exist_ok=True)

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['comp_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_tensor = visuals['result']
            cp_tensor = visuals['comp']
            mk_tensor = visuals['mask']
            rl_tensor = visuals['real']
            
            sr_tensor = sr_tensor * mk_tensor + rl_tensor *(1 - mk_tensor)

            sr_img = tensor2img([sr_tensor], rgb2bgr=rgb2bgr)
            cp_img = tensor2img([cp_tensor], rgb2bgr=rgb2bgr)
            mk_img = tensor2img([mk_tensor], rgb2bgr=False)

            if 'real' in visuals:
                real_img = tensor2img([rl_tensor], rgb2bgr=rgb2bgr)
                del self.real

            # tentative for out of GPU memory
            del self.comp
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                imwrite(sr_img  , osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_harmonized.jpg'))
                imwrite(cp_img  , osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_comp.jpg')) 
                imwrite(mk_img  , osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_mask.png')) 
                imwrite(real_img, osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_real.jpg')) 

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(osp.join(self.opt['path']['visualization'], dataset_name), img_name)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(visuals['result'], visuals['real'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
                    
            txt_save_path = os.path.join(self.opt['path']['models'], f'{metric}.txt')
            with open(txt_save_path, 'a') as f:
                f.write(f'{current_iter},{value:.2f}\n')

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['comp'] = self.comp.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['mask'] = self.mask.detach().cpu()
        if hasattr(self, 'real'):
            out_dict['real'] = self.real.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

