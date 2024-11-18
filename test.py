import torch
import torchvision.transforms as T
import random
from scipy.fftpack import dct, idct
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch_dct
import torch.nn.functional as F
from PIL import Image

def save_tensor_as_image(tensor, path):
    # 将张量转换为图像并保存
    image_tensor = (tensor + 1) / 2.0
    image_tensor = torch.clamp(image_tensor, 0, 1)
    transform = T.ToPILImage()
    img = transform(image_tensor.cpu())
    img.save(path)

def fully_random_drop_mask(h, w):
    # 创建一个全随机的丢弃掩码
    return np.random.rand(h, w) > 0.5


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
    SFM_center_sigma_perc = random.gauss(0.1, SFM_center_sigma_perc)
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
    c, h, w = img.shape
    padded_h = 1 << (h - 1).bit_length()  # find nearest power of two for height
    padded_w = 1 << (w - 1).bit_length()  # find nearest power of two for width
    print(img)
    save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_ori.png')
    img = F.pad(img, (0, padded_w - w, 0, padded_h - h))  # pad to nearest power of two
    new_c, new_h, new_w = img.shape
    if mode == 0:
        mask = fully_random_drop_mask(w, h)
    if mode == 1:
        mask = circular_random_drop_mask(w, h)
    if mode == 2:
        mask1, mask2 = circular_random_drop_mask(new_h, new_w, SFM_center_radius_perc, SFM_center_sigma_perc)
    print(mask1)
    img2 = img.clone().detach()
    mask1 = torch.tensor(mask1).to(img.device).to(img.dtype)
    mask2 = torch.tensor(mask2).to(img.device).to(img.dtype)

    save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_step1.png')
    save_tensor_as_image(mask1, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/mask1.png')
    save_tensor_as_image(mask2, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/mask2.png')

    if c == 3:
        img_dct = torch_dct.dct_2d(img)
        save_tensor_as_image(img_dct/1000, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_dct_1.png')
        img_dct = img_dct * mask1
        save_tensor_as_image(img_dct/1000, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_dct_mask_1.png')
        img = torch_dct.idct_2d(img_dct)
        save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_idct_1.png')

        img_dct = torch_dct.dct_2d(img2)
        save_tensor_as_image(img_dct/1000, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_dct_2.png')
        img_dct = img_dct * mask2
        save_tensor_as_image(img_dct/1000, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_dct_mask_2.png')
        img2 = torch_dct.idct_2d(img_dct)
        save_tensor_as_image(img2, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_idct_2.png')
    elif c == 1:
        img_dct = torch_dct.dct_2d(img)
        img_dct = img_dct * mask1
        img = torch_dct.idct_2d(img_dct)
        save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_step3.png')

        img_dct = torch_dct.dct_2d(img2)
        img_dct = img_dct * mask2
        img2 = torch_dct.idct_2d(img_dct)
        save_tensor_as_image(img2, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img2_step3.png')

    img = img[:, :h, :w]
    img2 = img2[:, :h, :w]
    save_tensor_as_image(img, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img_final.png')
    save_tensor_as_image(img2, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/img2_final.png')
    img = new_w.shape
    return (img, img2, mask1, mask2)

# 示例展示部分
if __name__ == "__main__":
    # 加载图像并转换为张量
    image_path = "/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat_old/hhy_test/airplane98.tif"
    img = Image.open(image_path).convert('RGB')
    transform = T.ToTensor()
    img_tensor = transform(img)

    # 调用 random_drop_tensor 函数
    img, img2, mask1, mask2 = random_drop_tensor(img_tensor, mode=2, SFM_center_radius_perc=0.85, SFM_center_sigma_perc=0.3)

    # 显示结果图像
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_tensor.permute(1, 2, 0).cpu().numpy())

    plt.subplot(2, 2, 2)
    plt.title("Masked Image 1")
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())

    plt.subplot(2, 2, 3)
    plt.title("Masked Image 2")
    plt.imshow(img2.permute(1, 2, 0).cpu().numpy())

    plt.subplot(2, 2, 4)
    plt.title("Mask 1")
    plt.imshow(mask1.cpu().numpy(), cmap='gray')

    plt.show()
