import os
import numpy as np
import glob
import cv2
import tqdm
from evaluation.psnr import calculate_psnr
from evaluation.ssim import calculate_ssim
from evaluation.niqe import calculate_niqe
from evaluation.lpips import calculate_lpips
from evaluation.fid import calculate_fid

def get_image_pair_func(sr_list,hr_list):
    '''
    用于重新组织图像对，使得两个列表中的图像一一对应
    一般情况，如果命名规则一致，直接 sort 就可以
    '''
    sr_list_new = []
    hr_list_new = []
    sr_root = os.path.dirname(sr_list[0])
    for i in range(len(hr_list)):
        cur_key_hr = os.path.basename(hr_list[i]).split('.')[0]
        sr_path_with_key = os.path.join(sr_root,cur_key_hr+'x4_SwinIR.png')
        if sr_path_with_key in sr_list:
            sr_list_new.append(sr_path_with_key)
            hr_list_new.append(hr_list[i])
        else:
            print(f"Can't find {sr_path_with_key}")
    return sr_list_new,hr_list_new

if __name__ == '__main__':
    hr_path = "asserts/Set5/HR"
    sr_path = "asserts/Set5/SR_x4"
    
    hr_fnames = glob.glob(os.path.join(hr_path, "*.png"))
    sr_fnames = glob.glob(os.path.join(sr_path, "*.png"))
    sr_fnames, hr_fnames = get_image_pair_func(sr_fnames,hr_fnames)
    
    psnr_rgb_list = []
    ssim_rgb_list = []
    psnr_y_list = []
    ssim_y_list = []
    niqe_y_list = []    
    lpips_list = []
    bar = tqdm.tqdm(range(len(sr_fnames)))
    for sr_name, hr_name in zip(sr_fnames, hr_fnames):
        sr = cv2.imread(sr_name)
        hr = cv2.imread(hr_name)
        bar.update(1)
        bar.set_description(f"Processing {sr_name}, {hr_name}")
        crop_border = 4
        # 对于 y_channel_test，使用 bgr2y_channel
        psnr_rgb = calculate_psnr(sr, hr,crop_border=crop_border,test_y_channel=False) 
        ssim_rgb = calculate_ssim(sr, hr,crop_border=crop_border,test_y_channel=False)
        psnr_rgb_list.append(psnr_rgb)
        ssim_rgb_list.append(ssim_rgb)
        
        # 在 rgb 通道计算，输入 bgr 或者 rgb 都可以，结果一致
        psnr_y = calculate_psnr(sr, hr,crop_border=crop_border,test_y_channel=True)
        ssim_y = calculate_ssim(sr, hr,crop_border=crop_border,test_y_channel=True)
        psnr_y_list.append(psnr_y)
        ssim_y_list.append(ssim_y)
        
        niqe = calculate_niqe(sr, crop_border=0, input_order='HWC', convert_to='y')
        niqe_y_list.append(niqe)
        
        current_lpips = calculate_lpips(sr, hr, crop_border=crop_border, input_order='HWC', test_y_channel=False,net='vgg')
        lpips_list.append(current_lpips)
    
    # 如果图像大小尺寸不一致，建议使用 batchsize=1，否则需要单独处理
    fid = calculate_fid(hr_path, sr_path, batchsize=1, save_status=True)
        
    psnr_mean_rgb = np.mean(psnr_rgb_list)
    ssim_mean_rgb = np.mean(ssim_rgb_list)
    
    psnr_mean_y = np.mean(psnr_y_list)
    ssim_mean_y = np.mean(ssim_y_list)
    
    niqe_mean_y = np.mean(niqe_y_list)
    lpips_mean = np.mean(lpips_list)    
    
    print("RGB: PSNR: {:.2f} dB, SSIM: {:.4f}".format(psnr_mean_rgb, ssim_mean_rgb))
    print("Y: PSNR: {:.2f} dB, SSIM: {:.4f}".format(psnr_mean_y, ssim_mean_y))
    print("Y: NIQE: {:.4f}".format(niqe_mean_y))
    print("LPIPS: {:.4f}".format(lpips_mean))
    print("FID: {:.4f}".format(fid))
    # RGB: PSNR: 30.99 dB, SSIM: 0.8758
    # Y: PSNR: 32.92 dB, SSIM: 0.9044
    # Y: NIQE: 7.0217
    # FID: 51.5310
