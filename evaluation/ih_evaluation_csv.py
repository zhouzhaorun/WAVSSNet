from PIL import Image
import numpy as np
import os
import torch
import argparse
import pytorch_ssim
import torchvision.transforms.functional as tf
import torchvision
import torch.nn.functional as f
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataroot', type=str, default='', help='dataset_dir')
    parser.add_argument('--result_root', type=str, default='', help='dataset_dir')
    parser.add_argument('--dataset_name', type=str, default='ihd', help='dataset_name')
    parser.add_argument('--evaluation_type', type=str, default="our", help='evaluation type')
    parser.add_argument('--ssim_window_size', type=int, default=11, help='ssim window size')
    return parser.parse_args()

def main(dataset_name = None):
    cuda = True if torch.cuda.is_available() else False
    IMAGE_SIZE = np.array([256,256])
    opt.dataset_name = dataset_name
    if opt.dataset_name == "IHD":
        files = os.path.join(opt.dataroot, opt.dataset_name+'_'+opt.phase+'.txt')
    else:
        files = os.path.join(opt.dataroot, dataset_name, opt.dataset_name+'_'+opt.phase+'.txt')

    comp_paths = []
    harmonized_paths = []
    mask_paths = []
    real_paths = []
    
    with open(files,'r') as f:
        for line in f.readlines():
            name_str = line.rstrip()
            if opt.evaluation_type == 'our':
                if opt.dataset_name == "IHD":
                    name_str = name_str.replace("composite_images/", "")
                    harmonized_path = os.path.join(opt.result_root, name_str.replace(".jpg", "_harmonized.jpg"))
                    if os.path.exists(harmonized_path):
                        real_path = os.path.join(opt.result_root, name_str.replace(".jpg", "_real.jpg"))
                        mask_path = os.path.join(opt.result_root, name_str.replace(".jpg", "_mask.jpg"))
                        comp_path = os.path.join(opt.result_root, name_str.replace(".jpg", "_comp.jpg"))
                else:
                    harmonized_path = os.path.join(opt.result_root, opt.dataset_name, name_str.replace(".jpg", "_harmonized.jpg"))
                    if os.path.exists(harmonized_path):
                        real_path = os.path.join(opt.result_root, opt.dataset_name, name_str.replace(".jpg", "_real.jpg"))
                        mask_path = os.path.join(opt.result_root, opt.dataset_name, name_str.replace(".jpg", "_mask.jpg"))
                        comp_path = os.path.join(opt.result_root, opt.dataset_name, name_str.replace(".jpg", "_comp.jpg"))

            elif opt.evaluation_type == 'ori':
                comp_path = os.path.join(opt.dataroot, 'composite_images', line.rstrip())
                harmonized_path = comp_path
                if os.path.exists(comp_path):
                    real_path = os.path.join(opt.dataroot,'real_images',line.rstrip())
                    name_parts=real_path.split('_')
                    real_path = real_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
                    mask_path = os.path.join(opt.dataroot,'masks',line.rstrip())
                    mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')

            real_paths.append(real_path)
            mask_paths.append(mask_path)
            comp_paths.append(comp_path)
            harmonized_paths.append(harmonized_path)

    count = 0
    mse_scores = 0
    sk_mse_scores = 0
    fmse_scores = 0
    psnr_scores = 0
    fpsnr_scores = 0
    ssim_scores = 0
    fssim_scores = 0
    fore_area_count = 0
    image_metric_list = []
    image_size = 256

    for i, harmonized_path in enumerate(tqdm(harmonized_paths)):
        count += 1

        harmonized = Image.open(harmonized_path).convert('RGB')
        real = Image.open(real_paths[i]).convert('RGB')
        mask = Image.open(mask_paths[i]).convert('1')
        if mask.size[0] != image_size:
            harmonized = tf.resize(harmonized,[image_size,image_size], interpolation=Image.BICUBIC)
            mask = tf.resize(mask, [image_size,image_size], interpolation=Image.BICUBIC)
            real = tf.resize(real,[image_size,image_size], interpolation=Image.BICUBIC)

        harmonized_np = np.array(harmonized, dtype=np.float32)
        real_np = np.array(real, dtype=np.float32)
        
        harmonized = tf.to_tensor(harmonized_np).unsqueeze(0).cuda()
        real = tf.to_tensor(real_np).unsqueeze(0).cuda()
        mask = tf.to_tensor(mask).unsqueeze(0).cuda()

        mse_score = mse(harmonized_np, real_np)
        psnr_score = psnr(real_np, harmonized_np, data_range=255)

        fore_area = torch.sum(mask)
        fmse_score = torch.nn.functional.mse_loss(harmonized*mask,real*mask)*256*256/fore_area

        mse_score = mse_score.item()
        fmse_score = fmse_score.item()
        fore_area_count += fore_area.item()
        fpsnr_score = 10 * np.log10((255 ** 2) / fmse_score)

        ssim_score, fssim_score = pytorch_ssim.ssim(harmonized, real, window_size=opt.ssim_window_size, mask=mask)
        ssim_score = float(ssim_score)
        fssim_score = float(fssim_score)
        foreground_area = float(fore_area.item())
        foreground_ratio = foreground_area / float(image_size * image_size)

        psnr_scores += psnr_score
        mse_scores += mse_score
        fmse_scores += fmse_score
        fpsnr_scores += fpsnr_score
        ssim_scores += ssim_score
        fssim_scores += fssim_score

        image_metric_info = {
            'image_name': os.path.basename(harmonized_path),
            'mse': round(mse_score, 4),
            'psnr': round(psnr_score, 4),
            'fmse': round(fmse_score, 4),
            'fpsnr': round(fpsnr_score, 4),
            'ssim': round(ssim_score, 6),
            'fssim': round(fssim_score, 6),
            'foreground_area': round(foreground_area, 4),
            'foreground_ratio': round(foreground_ratio, 6)
        }
        image_metric_list.append(image_metric_info)

    mse_scores_mu = mse_scores/count
    psnr_scores_mu = psnr_scores/count
    fmse_scores_mu = fmse_scores/count
    fpsnr_scores_mu = fpsnr_scores/count
    ssim_scores_mu = ssim_scores/count
    fssim_score_mu = fssim_scores/count

    with open(os.path.join(opt.result_root, 'log.txt'), 'a', encoding='utf-8') as log_file:
        print(f"{opt.dataset_name}: {count}")
        print(f"{opt.dataset_name}: {count}", file=log_file)
        mean_sore = "%s MSE %0.2f | PSNR %0.2f | SSIM %0.4f |fMSE %0.2f | fPSNR %0.2f | fSSIM %0.4f" % (opt.dataset_name,mse_scores_mu, psnr_scores_mu,ssim_scores_mu,fmse_scores_mu,fpsnr_scores_mu,fssim_score_mu)
        print(mean_sore)
        print(mean_sore, file=log_file)

        os.makedirs(opt.result_root, exist_ok=True)
        save_path = os.path.join(opt.result_root, '%s_per_image_metrics.csv' % opt.dataset_name)
        save_per_image_metrics(save_path, image_metric_list)
        print('Per-image metrics saved to %s' % save_path)
        print('Per-image metrics saved to %s' % save_path, file=log_file)

    return mse_scores_mu,fmse_scores_mu, psnr_scores_mu,fpsnr_scores_mu

def generstr(dataset_name='ALL'): 
    datasets = ['HCOCO','HAdobe5k','HFlickr','Hday2night','IHD']
    if dataset_name == 'newALL':
        datasets = ['HCOCO','HAdobe5k','HFlickr','Hday2night','HVIDIT','newIHD']
    for i, item in enumerate(datasets):
        print(item)
        mse_scores_mu,fmse_scores_mu, psnr_scores_mu,fpsnr_scores_mu = main(dataset_name=item)


import csv
def save_per_image_metrics(save_path, image_metric_list):
    fieldnames = [
        'image_name',
        'mse',
        'psnr',
        'fmse',
        'fpsnr',
        'ssim',
        'fssim',
        'foreground_area',
        'foreground_ratio'
    ]

    with open(save_path,'w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        writer.writeheader()
        for item in image_metric_list:
            writer.writerow(item)



if __name__ == '__main__':
    opt = parse_args()
    if opt is None:
        exit()
    if opt.dataset_name == "ALL":
        generstr()
    elif opt.dataset_name == "newALL":
        generstr(dataset_name='newALL')
    else:
        main(dataset_name=opt.dataset_name)


