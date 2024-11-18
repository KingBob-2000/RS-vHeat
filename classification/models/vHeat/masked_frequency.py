"""
SFM: Stochastic Frequency Masking
"""
from PIL import Image

import torchvision.transforms as T
import random
from scipy.fftpack import dct, idct
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch_dct
def fully_random_drop_mask(w=256, h=256, radius=0, p=0.5):
    mask_random = np.random.rand(w, h)
    mask = np.ones((w, h))
    mask = np.ones((w, h))
    mask[mask_random < p] = 0

    return mask


def circular_random_drop_mask(w=256, h=256, SFM_center_radius_perc=-1, SFM_center_sigma_perc=0.05):
    '''
    (w,h) are the dimensions of the mask

    IF (SFM_center_radius_perc=-1)
        the masked regions are selected randomly in circular shape, with the maximum at "radius"
        when "radius" is 0, it is set to the max default value

    ELSE
        the masked regions are always centered at "SFM_center_radius_perc*radius", and stretch inwards and
        outwards with a Gaussian probability, with sigma=SFM_center_sigma_perc*radius
    '''

    radius = np.sqrt(w * w + h * h)  # 362
    # SFM_center_sigma_perc = random.gauss(0.2, SFM_center_sigma_perc)
    # SFM_center_sigma_perc = 0.2, SFM_center_sigma_perc

    SFM_center_sigma = SFM_center_sigma_perc * radius
    SFM_center_radius = SFM_center_radius_perc * radius

    X, Y = np.meshgrid(np.linspace(0, h - 1, h), np.linspace(0, w - 1, w))
    D = np.sqrt(X * X + Y * Y)

    # random SFM (SFM_center_radius 0) vs SFM around a center of given distance
    if SFM_center_radius_perc == -1:
        a1 = random.random() * radius
        a2 = random.random() * radius
        if (a1 > a2):
            tmp = a2;
            a2 = a1;
            a1 = tmp
        mask = np.ones((w, h))
        mask[(D > a1) & (D < a2)] = 0

    else:
        if SFM_center_radius > radius or SFM_center_radius < 0:
            raise Exception('SFM_center_radius out of bounds.')

        a1 = 0
        a2 = SFM_center_sigma

        a1 = abs(a1)
        a2 = abs(a2)

        mask1 = np.ones((w, h))
        mask2 = np.ones((w, h))
        mask1[(D > a1) & (D < a2)] = 0  # low
        mask2[D > a2] = 0  # high

    return mask1,mask2


def random_drop_tensor(img, mode=1, SFM_center_radius_perc=-1, SFM_center_sigma_perc=0.05):
    ''' mode=0:fully random drop, mode=1: circular random drop, mode=2 sweeping mode

        **sweeping mode**:
            SFM_center_radius_perc: determines the center of the band to be erased
                                    it is a percentage of the max radius
            SFM_center_sigma_perc:  determines the sigma for the width of the band
                                    sigma=radius*SFM_center_sigma_perc
    '''
    c, h, w = img.shape
    padded_h = 1 << (h - 1).bit_length()  # find nearest power of two for height
    padded_w = 1 << (w - 1).bit_length()  # find nearest power of two for width
    img = F.pad(img, (0, padded_w - w, 0, padded_h - h))  # pad to nearest power of two
    new_c, new_h, new_w = img.shape
    if mode == 0:
        mask = fully_random_drop_mask(w, h)
    if mode == 1:
        mask = circular_random_drop_mask(w, h)
    if mode == 2:
        mask1, mask2 = circular_random_drop_mask(new_h, new_w, SFM_center_radius_perc, SFM_center_sigma_perc)
    mask1 = torch.tensor(mask1).to(img.device).to(img.dtype)
    mask2 = torch.tensor(mask2).to(img.device).to(img.dtype)
    img_dct = torch_dct.dct_2d(img)
    tmp1 = img_dct * mask1
    tmp2 = img_dct * mask2
    img = torch_dct.idct_2d(tmp1)
    img2 = torch_dct.idct_2d(tmp2)
    img = img[:, :h, :w]
    img2 = img2[:, :h, :w]
    return (img, img2, mask1, mask2)

######

# def save_tensor_as_image(tensor, path):
#     # 将张量转换为图像并保存
#     image_tensor = (tensor + 1) / 2.0
#     image_tensor = torch.clamp(image_tensor, 0, 1)
#     transform = T.ToPILImage()
#     img = transform(image_tensor.cpu())
#     img.save(path)
#
#
# def random_drop_tensor(img, mode=1, SFM_center_radius_perc=-1, SFM_center_sigma_perc=0.05):
#     c, h, w = img.shape
#     padded_h = 1 << (h - 1).bit_length()  # find nearest power of two for height
#     padded_w = 1 << (w - 1).bit_length()  # find nearest power of two for width
#     print(img)
#     save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_ori.png')
#     img = F.pad(img, (0, padded_w - w, 0, padded_h - h))  # pad to nearest power of two
#     new_c, new_h, new_w = img.shape
#     if mode == 0:
#         mask = fully_random_drop_mask(w, h)
#     if mode == 1:
#         mask = circular_random_drop_mask(w, h)
#     if mode == 2:
#         mask1, mask2 = circular_random_drop_mask(new_h, new_w, SFM_center_radius_perc, SFM_center_sigma_perc)
#     img2 = img.clone().detach()
#     mask1 = torch.tensor(mask1).to(img.device).to(img.dtype)
#     mask2 = torch.tensor(mask2).to(img.device).to(img.dtype)
#
#     save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_step1.png')
#     save_tensor_as_image(mask1, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/mask1.png')
#     save_tensor_as_image(mask2, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/mask2.png')
#
#     if c == 3:
#         img_dct = torch_dct.dct_2d(img)
#         save_tensor_as_image(img_dct, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_dct_1.png')
#         img_dct = img_dct * mask1
#         save_tensor_as_image(img_dct, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_dct_mask_1.png')
#         img = torch_dct.idct_2d(img_dct)
#         save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_idct_1.png')
#
#         img_dct = torch_dct.dct_2d(img2)
#         save_tensor_as_image(img_dct, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_dct_2.png')
#         img_dct = img_dct * mask2
#         save_tensor_as_image(img_dct, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_dct_mask_2.png')
#         img2 = torch_dct.idct_2d(img_dct)
#         save_tensor_as_image(img2, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_idct_2.png')
#     elif c == 1:
#         img_dct = torch_dct.dct_2d(img)
#         img_dct = img_dct * mask1
#         img = torch_dct.idct_2d(img_dct)
#         save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_step3.png')
#
#         img_dct = torch_dct.dct_2d(img2)
#         img_dct = img_dct * mask2
#         img2 = torch_dct.idct_2d(img_dct)
#         save_tensor_as_image(img2, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img2_step3.png')
#
#     img = img[:, :h, :w]
#     img2 = img2[:, :h, :w]
#     save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img_final.png')
#     save_tensor_as_image(img2, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat0612/img2_final.png')
#     img = new_w.shape
#     return (img, img2, mask1, mask2)

######

def random_drop(img, mode=1, SFM_center_radius_perc=-1, SFM_center_sigma_perc=0.05):
    ''' mode=0:fully random drop, mode=1: circular random drop, mode=2 sweeping mode

        **sweeping mode**:
            SFM_center_radius_perc: determines the center of the band to be erased
                                    it is a percentage of the max radius
            SFM_center_sigma_perc:  determines the sigma for the width of the band
                                    sigma=radius*SFM_center_sigma_perc
    '''
    (c, w, h) = np.shape(img)
    if mode == 0:
        mask = fully_random_drop_mask(w, h)
    if mode == 1:
        mask = circular_random_drop_mask(w, h)
    if mode == 2:
        mask1, mask2 = circular_random_drop_mask(w, h, SFM_center_radius_perc, SFM_center_sigma_perc)
    img = img.detach().cpu().numpy()
    img2 = img.copy()
    if c == 3:
        img0_dct = dct(dct(img[0, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img1_dct = dct(dct(img[1, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img2_dct = dct(dct(img[2, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img0_dct1 = img0_dct * mask1
        img1_dct1 = img1_dct * mask1
        img2_dct1 = img2_dct * mask1

        img0_dct2 = img0_dct * mask2
        img1_dct2 = img1_dct * mask2
        img2_dct2 = img2_dct * mask2
        img[0, :, :] = idct(idct(img0_dct1, axis=0, norm='ortho'), axis=1, norm='ortho')
        img[1, :, :] = idct(idct(img1_dct1, axis=0, norm='ortho'), axis=1, norm='ortho')
        img[2, :, :] = idct(idct(img2_dct1, axis=0, norm='ortho'), axis=1, norm='ortho')
        img2[0, :, :] = idct(idct(img0_dct2, axis=0, norm='ortho'), axis=1, norm='ortho')
        img2[1, :, :] = idct(idct(img1_dct2, axis=0, norm='ortho'), axis=1, norm='ortho')
        img2[2, :, :] = idct(idct(img2_dct2, axis=0, norm='ortho'), axis=1, norm='ortho')
    elif c == 1:
        img_dct = dct(dct(img[0, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img_dct = img_dct * mask1
        img[0, :, :] = idct(idct(img_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        img_dct = img_dct * mask2
        img2[0, :, :] = idct(idct(img_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    img = torch.from_numpy(img)
    img2 = torch.from_numpy(img2)
    return (img, img2, mask1, mask2)


def DCT(img):
    ''' mode=0:fully random drop, mode=1: circular random drop, mode=2 sweeping mode

        **sweeping mode**:
            SFM_center_radius_perc: determines the center of the band to be erased
                                    it is a percentage of the max radius
            SFM_center_sigma_perc:  determines the sigma for the width of the band
                                    sigma=radius*SFM_center_sigma_perc
    '''
    (c, w, h) = np.shape(img)
    dct_img = img.copy()
    if c == 3:
        img0_dct = dct(dct(img[0, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img1_dct = dct(dct(img[1, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img2_dct = dct(dct(img[2, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_img[0, :, :] = img0_dct
        dct_img[1, :, :] = img1_dct
        dct_img[2, :, :] = img2_dct
    elif c == 1:
        img_dct = dct(dct(img[0, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_img = img_dct
    return dct_img

import torch
import torch.nn.functional as F

def pad_to_power_of_two(x):
    original_size = x.size(-1)
    padded_size = 1 << (original_size - 1).bit_length()  # 找到最接近的二的幂次
    pad_size = padded_size - original_size
    x = F.pad(x, (0, pad_size, 0, pad_size))  # 填充到最近的二的幂次
    return x

def dct_2d(x):
    x = x.float()  # ensure input is float32
    X1 = torch.fft.fftn(x, dim=(-2, -1))  # apply FFT on last two dimensions
    X1_real = torch.real(X1)  # take real part
    return X1_real
def apply_dct(img_tensor,type="none"):
    """ Apply 2D DCT on a batch of image tensors """
    # if type =="sar":
    #     img_tensor = img_tensor.to(torch.float32)  # 提高数据精度
    #     c, h, w = img_tensor.shape  # get channel, height, width
    #     padded_h = 1 << (h - 1).bit_length()  # find nearest power of two for height
    #     padded_w = 1 << (w - 1).bit_length()  # find nearest power of two for width
    #     # 计算最小值和最大值
    #     min_val = img_tensor.min()
    #     max_val = img_tensor.max()
    #     # 归一化到 [0, 1] 之间
    #     img_tensor = (img_tensor - min_val) / (max_val - min_val)
    #     # 调整范围到 [-1, 1] 之间
    #     img_tensor = img_tensor * 2 - 1
    #     img_tensor = F.pad(img_tensor, (0, padded_w - w, 0, padded_h - h))  # pad to nearest power of two
    #     # print("img_tensor",img_tensor)
    #     dct_img = torch_dct.dct_2d(img_tensor) / 1000
    #     # print("dct_img",dct_img)
    #     return dct_img[:, :h, :w]  # trim back to original size after DCT
    img_tensor = img_tensor.to(torch.float32)  # 提高数据精度
    c, h, w = img_tensor.shape  # get channel, height, width
    padded_h = 1 << (h - 1).bit_length()  # find nearest power of two for height
    padded_w = 1 << (w - 1).bit_length()  # find nearest power of two for width
    # if type != "image":
    #     # 计算最小值和最大值
    #     min_val = img_tensor.min()
    #     max_val = img_tensor.max()
    #     # 归一化到 [0, 1] 之间
    #     img_tensor = (img_tensor - min_val) / (max_val - min_val)
    #     # 调整范围到 [-1, 1] 之间
    #     img_tensor = img_tensor * 2 - 1
    img_tensor = F.pad(img_tensor, (0, padded_w - w, 0, padded_h - h))  # pad to nearest power of two
    dct_img = torch_dct.dct_2d(img_tensor) / 1000
    # 筛选出 inf 值
    inf_mask = torch.isinf(dct_img)
    # 检查是否有任何 True 值
    if inf_mask.any():
        # 打印出包含无穷大值的索引
        print("!!inf!!!before max:", dct_img.max())
        print("!!inf!!!before min:", dct_img.min())
        dct_img[inf_mask] = 0
        print("!!inf!!!after max:", dct_img.max())
        print("!!inf!!!after min:", dct_img.min())
    # if type == "image":
    #     print("image before max:", dct_img.max())
    #     print("image before min:", dct_img.min())
    # else:
    #     print("after max:", dct_img.max())
    #     print("after min:", dct_img.min())
    if type != "image":
        dct_img = torch.clamp(dct_img, min=-1000, max=1000)
    # 将 inf 值替换为零（或者你认为合适的其他值）
    return dct_img[:, :h, :w]  # trim back to original size after DCT

def get_mask_low_high(w=256, h=256, radius_perc=-1, mask_mode=-1):
    '''
    (w,h) are the dimensions of the mask
    if mask_mode==1 low frequencies are cut off
    if mask_mode==2 high frequencies are cut off

    returns a binary mask of low or of high frequencies, cut-off at radius_perc*radius
    '''

    if radius_perc < 0:
        raise Exception('radius_perc must be positive.')

    radius = np.sqrt(w * w + h * h)
    center_radius = radius_perc * radius

    X, Y = np.meshgrid(np.linspace(0, h - 1, h), np.linspace(0, w - 1, w))
    D = np.sqrt(X * X + Y * Y)

    if mask_mode == 1:
        a1 = 0
        a2 = center_radius
    elif mask_mode == 2:
        a1 = center_radius
        a2 = radius
    else:
        raise Exception('mask_mode must be 1 or 2.')

    mask = np.ones((w, h))
    mask[(D >= a1) & (D <= a2)] = 0

    return mask


