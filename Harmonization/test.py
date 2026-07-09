import os
import argparse
from tqdm import tqdm
from basicsr.data.data_util import paired_paths_from_harmony_folder_ori
import torch.nn as nn
import torch
import cv2
from basicsr.models.archs.wavssnet_arch import wavssnet
from basicsr.utils.img_util import tensor2img, img2tensor
from basicsr.utils import FileClient, imfrombytes, img2tensor

# from skimage.metrics import mean_squared_error
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser(description='Image Harmonization using Restormer')
parser.add_argument('--weights', default='wavssnet.pth', type=str, help='Path to weights')
parser.add_argument('--yaml_file', default='wavssnet.yml', type=str, help='Path to yaml_file')
parser.add_argument('--dataset_root', default=None, type=str, help='Directory of input images or path of single image')
parser.add_argument('--datasets', nargs='+', type=str, default=['HAdobe5k', 'HCOCO', 'HFlickr', 'Hday2night'], help='Datasets used for evaluation')
parser.add_argument('--result_dir', default=None, type=str, help='Directory for restored results')
parser.add_argument('--image_size', type=int, default=256, help='Tile size (e.g 720). None means testing on the original resolution image')
args = parser.parse_args()

yaml_file = args.yaml_file
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
model_restoration = wavssnet(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets_metrics = []

for dataset in args.datasets:
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)
    testfile = os.path.join(args.dataset_root, dataset, dataset + '_test.txt') 
    test_image_paths = []
    with open(testfile,'r') as f:
        for line in f.readlines():
            test_image_paths.append(os.path.join(args.dataset_root, dataset, 'composite_images', line.rstrip())) 
    paths = paired_paths_from_harmony_folder_ori(test_image_paths)

    io_backend_opt = {'type': 'disk'}
    print(f"dataset: {dataset}, number: {len(paths)}")

    with torch.no_grad():
        for file_ in tqdm(paths):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            file_client = FileClient()
            image_name = os.path.splitext(os.path.split(file_['comp_path'])[-1])[0]
            img_bytes = file_client.get(file_['comp_path'], 'comp')
            try:
                img_comp = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("comp path {} not working".format(file_['comp_path']))
            img_bytes = file_client.get(file_['mask_path'], 'mask')
            try:
                img_mask = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("mask path {} not working".format(file_['mask_path']))
            img_bytes = file_client.get(file_['real_path'], 'real')
            try:
                img_real = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("real path {} not working".format(file_['real_path']))

            img_comp_np = cv2.resize(img_comp, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
            img_mask_np = cv2.resize(img_mask, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST) 
            img_real_np = cv2.resize(img_real, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_comp = img2tensor(img_comp_np, bgr2rgb=True, float32=True).unsqueeze(0).cuda() 
            img_mask = img2tensor(img_mask_np, bgr2rgb=False, float32=True).unsqueeze(0).cuda() 
            img_real = img2tensor(img_real_np, bgr2rgb=True, float32=True).unsqueeze(0).cuda() 

            restored = model_restoration(img_comp, img_mask)  
            restored = img_mask * restored + (1 - img_mask) * img_real

            harm_img = tensor2img([restored], rgb2bgr=True)
            real_img = tensor2img([img_real], rgb2bgr=True)
            mask_img = tensor2img(img_mask, rgb2bgr=False)
            comp_img = tensor2img(img_comp, rgb2bgr=True)

            cv2.imwrite(os.path.join(result_dir, image_name + '_harmonized.' + 'jpg'), harm_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) 
            cv2.imwrite(os.path.join(result_dir, image_name + '_real.' + 'jpg'), real_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(os.path.join(result_dir, image_name + '_mask.' + 'jpg'), mask_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) 
            cv2.imwrite(os.path.join(result_dir, image_name + '_comp.' + 'jpg'), comp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) 



