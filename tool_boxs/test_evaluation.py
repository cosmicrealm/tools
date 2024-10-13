import os
import numpy as np
import glob
import cv2
import tqdm
import multiprocessing as mp
from evaluation.psnr import calculate_psnr
from evaluation.ssim import calculate_ssim
from evaluation.niqe import calculate_niqe
from evaluation.lpips import calculate_lpips
from evaluation.fid import calculate_fid

def get_image_pair_func(sr_list, hr_list):
    """
    Organizes image pairs, ensuring corresponding images in both lists.
    """
    sr_list_new = []
    hr_list_new = []
    sr_root = os.path.dirname(sr_list[0])
    for i in range(len(hr_list)):
        cur_key_hr = os.path.basename(hr_list[i])
        sr_path_with_key = os.path.join(sr_root, cur_key_hr)
        if sr_path_with_key in sr_list:
            sr_list_new.append(sr_path_with_key)
            hr_list_new.append(hr_list[i])
        else:
            print(f"Can't find {sr_path_with_key}")
    return sr_list_new, hr_list_new

def process_images(image_pair):
    """
    Processes a single image pair and calculates the metrics.
    """
    sr_name, hr_name = image_pair
    sr = cv2.imread(sr_name)
    hr = cv2.imread(hr_name)

    # Initialize lists for metrics
    results = {}
    crop_border = 0

    # Calculate metrics
    psnr_rgb = calculate_psnr(sr, hr, crop_border=crop_border, test_y_channel=False)
    ssim_rgb = calculate_ssim(sr, hr, crop_border=crop_border, test_y_channel=False)
    
    psnr_y = calculate_psnr(sr, hr, crop_border=crop_border, test_y_channel=True)
    ssim_y = calculate_ssim(sr, hr, crop_border=crop_border, test_y_channel=True)

    niqe = calculate_niqe(sr, crop_border=0, input_order='HWC', convert_to='y')
    
    lpips = calculate_lpips(sr, hr, crop_border=crop_border, input_order='HWC', test_y_channel=False, net='vgg')

    # Append results to the dictionary
    results['psnr_rgb'] = psnr_rgb
    results['ssim_rgb'] = ssim_rgb
    results['psnr_y'] = psnr_y
    results['ssim_y'] = ssim_y
    results['niqe'] = niqe
    results['lpips'] = lpips

    return results

def process_in_parallel(sr_fnames, hr_fnames, num_processes=40):
    """
    Splits the work across multiple processes for parallel execution.
    """
    with mp.Pool(processes=num_processes) as pool:
        image_pairs = list(zip(sr_fnames, hr_fnames))
        results = list(tqdm.tqdm(pool.imap(process_images, image_pairs), total=len(image_pairs)))
    
    return results

if __name__ == '__main__':
    sr_path = "/mnt/hwdata/cv/users/zhangjinyang/workspace/IR-Adapter/results/CelebA-Test/CelebA-Test/decoded"
    hr_path = "/mnt/hwdata/cv/users/zhangjinyang/datasets/image_restoration/celeba_512_validation"
    
    hr_fnames = glob.glob(os.path.join(hr_path, "*.png"))
    sr_fnames = glob.glob(os.path.join(sr_path, "*.png"))
    hr_fnames.sort()
    sr_fnames.sort()
    sr_fnames, hr_fnames = get_image_pair_func(sr_fnames, hr_fnames)
    
    # Parallel processing
    num_processes = 40
    results = process_in_parallel(sr_fnames, hr_fnames, num_processes=num_processes)
    
    # Aggregate results
    psnr_rgb_list = [res['psnr_rgb'] for res in results]
    ssim_rgb_list = [res['ssim_rgb'] for res in results]
    psnr_y_list = [res['psnr_y'] for res in results]
    ssim_y_list = [res['ssim_y'] for res in results]
    niqe_y_list = [res['niqe'] for res in results]
    lpips_list = [res['lpips'] for res in results]
    
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