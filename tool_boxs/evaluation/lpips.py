from torchvision.transforms.functional import normalize
import lpips
import numpy as np
from evaluation.utils import reorder_image, to_y_channel,img2tensor
import torch

# Ensure loss_fn_vgg is globally available
loss_fn_vgg = None

def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, net = "vgg", **kwargs):
    """Calculate LPIPS.
    Ref: https://github.com/xinntao/BasicSR/pull/367
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: LPIPS result.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    # Start calculating LPIPS metrics
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if loss_fn_vgg exists globally, if not, create it
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = lpips.LPIPS(net=net, verbose=False).to(DEVICE)  # Initialize only once

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    img_gt = img2 / 255.
    img_restored = img / 255.

    img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
    
    # Normalize to [-1, 1]
    normalize(img_gt, mean, std, inplace=True)
    normalize(img_restored, mean, std, inplace=True)

    # Calculate LPIPS
    img_gt = img_gt.to(DEVICE)
    img_restored = img_restored.to(DEVICE)
    
    loss_fn_vgg.eval()
    lpips_val = loss_fn_vgg(img_restored.unsqueeze(0), img_gt.unsqueeze(0))

    return lpips_val.detach().cpu().numpy().mean()
