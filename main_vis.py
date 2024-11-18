# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from utils.config import get_config
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

from fvcore.nn import FlopCountAnalysis, flop_count_str

from timm.utils import ModelEma as ModelEma
from utils.utils_ema import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema
# print(f"||{torch.multiprocessing.get_start_method()}||", end="")
torch.multiprocessing.set_start_method("spawn", force=True)


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # parser.add_argument("--local-rank", type=int, required=True, help='local rank for DistributedDataParallel')
    # 在torch1.13时用local_rank
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    # parser.add_argument('--fused_window_process', action='store_true',
                        # help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    
    torch.cuda.empty_cache()
    
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if not (config.EVAL_MODE and config.MODEL.TYPE == 'vHeat'):
        try:
            logger.info(flop_count_str(FlopCountAnalysis(model, (dataset_val[0][0][None],))))
        except Exception as e:
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")
            if hasattr(model, 'flops'):
                flops = model.flops()
                logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)


    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_accuracy_ema = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        model_without_ddp, max_accuracy, max_accuracy_ema = load_checkpoint_ema(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger, model_ema)
        if config.EVAL_MODE and config.MODEL.TYPE == 'vHeat':
            try:
                logger.info(flop_count_str(FlopCountAnalysis(model_without_ddp.cpu(), (dataset_val[0][0][None],))))
                model_without_ddp.cuda()
            except Exception as e:
                logger.info(str(model))
                n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"number of params: {n_parameters}")
                if hasattr(model, 'flops'):
                    flops = model.flops()
                    logger.info(f"number of GFLOPs: {flops / 1e9}")
            acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        if model_ema is not None:
            throughput(data_loader_val, model_ema.ema, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        acc1, acc5, loss = validate(config, data_loader_val, model)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        
        if dist.get_rank() == 0 and acc1 > max_accuracy:
            save_checkpoint(config,
                            epoch,
                            model,
                            acc1,
                            optimizer,
                            lr_scheduler,
                            loss_scaler,
                            logger,
                            best='best')
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
            
            if dist.get_rank() == 0 and acc1_ema > max_accuracy_ema:
                save_checkpoint(config,
                                epoch,
                                model_ema.ema,
                                acc1_ema,
                                optimizer,
                                lr_scheduler,
                                loss_scaler,
                                logger,
                                best='ema_best')
            
            max_accuracy_ema = max(max_accuracy_ema, acc1_ema)
            logger.info(f'Max accuracy ema: {max_accuracy_ema:.2f}%')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

# def show_mask_on_image(img, mask):
#     img = np.float32(img) / 255
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)
def show_mask_on_image(img, mask):
    mask = 1 - mask
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_mask_on_image_fan(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_mask_on_sar(sar, mask):
    mask = 1 - mask
    img = np.float32(sar) / 255
    # 将单通道sar图像转换为三通道BGR图像
    img_bgr = cv2.cvtColor(np.uint8(255 * img), cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img_bgr)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
from PIL import Image
import cv2
import torch.nn.functional as F
@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    dir_name = 'vis_dct_outputs_heatmap_shuffle_norm'

    for idx, (images, target) in enumerate(data_loader):
        print(idx)
        if not os.path.exists('/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}'.format(dir_name, idx)):
            os.mkdir('/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}'.format(dir_name, idx))
            os.mkdir('/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/pt'.format(dir_name, idx))

        images_ori = images.squeeze().permute(1, 2, 0).contiguous()
        images_ori = images_ori * torch.tensor(IMAGENET_DEFAULT_STD) + torch.tensor(IMAGENET_DEFAULT_MEAN)
        # print(images_ori.max(), images_ori.min())
        images_ori = images_ori.detach().cpu().numpy()
        images_ori1 = Image.fromarray((images_ori * 255).astype(np.uint8))
        images_ori1.save('/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/ori_{}.png'.format(dir_name, idx))

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output, x_after_cor, x_before_dct_list, x_dct_list, x_after_idct_list, x_after_idct2_list, time_list = model(images)
            for idx1, m in enumerate(x_before_dct_list):  # (B, H, W, C)
                for idx2, n in enumerate(m):
                    # print(n.shape)
                    # torch.save(n, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/pt/beforedct_{}_{}.pt'.format(
                    #     dir_name, idx, idx1, idx2))
                    n = n.mean(dim=-1)

                    n1 = x_after_idct_list[idx1][idx2]
                    # torch.save(n1, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/pt/afteridct_{}_{}.pt'.format(
                    #     dir_name, idx, idx1, idx2))
                    n1 = n1.mean(dim=-1)

                    n2 = x_after_idct2_list[idx1][idx2]
                    # torch.save(n2,
                    #            '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/pt/afteridctmean_{}_{}.pt'.format(
                    #                dir_name, idx, idx1, idx2))
                    n2 = n2.mean(dim=-1)

                    n3 = x_after_cor[idx1][idx2]
                    n3 = n3.mean(dim=-1)

                    # min_total = torch.min(torch.min(n.min(), n1.min()), n2.min())
                    # max_total = torch.max(torch.max(n.max(), n1.max()), n2.max())

                    # n = (n - min_total) / (max_total - min_total)
                    # n1 = (n1 - min_total) / (max_total - min_total)
                    # n2 = (n2 - min_total) / (max_total - min_total)

                    n = (n - n.min()) / (n.max() - n.min())
                    n1 = (n1 - n1.min()) / (n1.max() - n1.min())
                    n2 = (n2 - n2.min()) / (n2.max() - n2.min())
                    n3 = (n3 - n3.min()) / (n3.max() - n3.min())

                    tmp = n.squeeze().cpu().numpy().astype(float)
                    tmp1 = n1.squeeze().cpu().numpy().astype(float)
                    tmp2 = n2.squeeze().cpu().numpy().astype(float)
                    tmp3 = n3.squeeze().cpu().numpy().astype(float)

                    mask = cv2.resize(tmp, (images_ori.shape[1], images_ori.shape[0]))
                    mask1 = cv2.resize(tmp1, (images_ori.shape[1], images_ori.shape[0]))
                    mask2 = cv2.resize(tmp2, (images_ori.shape[1], images_ori.shape[0]))
                    mask3 = cv2.resize(tmp3, (images_ori.shape[1], images_ori.shape[0]))

                    mask = show_mask_on_image(images_ori1, mask)
                    mask1 = show_mask_on_image(images_ori1, mask1)
                    mask2 = show_mask_on_image_fan(images_ori1, mask2)
                    mask3 = show_mask_on_image_fan(images_ori1, mask3)

                    name = '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/beforedct_{}_{}.png'.format(dir_name,
                                                                                                           idx, idx1,
                                                                                                           idx2)
                    name1 = '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/afteridct_{}_{}.png'.format(dir_name,
                                                                                                            idx, idx1,
                                                                                                            idx2)
                    name2 = '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/afteridctmean_{}_{}.png'.format(
                        dir_name, idx, idx1, idx2)
                    name3 = '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/after_cor_{}_{}.png'.format(
                        dir_name, idx, idx1, idx2)

                    cv2.imwrite(name, mask)
                    cv2.imwrite(name1, mask1)
                    cv2.imwrite(name2, mask2)
                    cv2.imwrite(name3, mask3)
            for idx1, m in enumerate(x_dct_list):  # (B, H, W, C)
                for idx2, n in enumerate(m):
                    # print(n.shape)
                    # torch.save(n, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/pt/dct_{}_{}.pt'.format(dir_name,
                    #                                                                                           idx, idx1,
                    #                                                                                           idx2))
                    n = n.mean(dim=-1)
                    n = (n - n.min()) / (n.max() - n.min())
                    tmp = F.interpolate(n.unsqueeze(0), size=(224, 224))
                    tmp = tmp.squeeze().cpu().numpy()
                    img = Image.fromarray((tmp * 255).astype(np.uint8))
                    img.save(
                        '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/dct_{}_{}.png'.format(dir_name, idx, idx1,
                                                                                                  idx2))

            for idx1, m in enumerate(time_list):  # (H, W, C)
                for idx2, n in enumerate(m):
                    # print(n.shape)
                    # torch.save(n, '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/pt/time_{}_{}.pt'.format(dir_name,
                    #                                                                                            idx,
                    #                                                                                            idx1,
                    #                                                                                            idx2))
                    n = n.mean(dim=-1).unsqueeze(0)
                    n = n / n.max()
                    # n = (n - n.min()) / (n.max() - n.min())
                    tmp = F.interpolate(n.unsqueeze(0), size=(224, 224))
                    tmp = tmp.squeeze().cpu().numpy()
                    img = Image.fromarray((tmp * 255).astype(np.uint8))
                    img.save(
                        '/mnt/AIService/FoundationModel/Mamba/downstream/huhuiyang/vheat/classification/analysis/{}/test_{}/time_{}_{}.png'.format(dir_name, idx, idx1,
                                                                                                   idx2))
        if idx == 1000:
            assert 1 == 2

    # assert 1==2

    # measure accuracy and record loss
    loss = criterion(output, target)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    acc1 = reduce_tensor(acc1)
    acc5 = reduce_tensor(acc5)
    loss = reduce_tensor(loss)

    loss_meter.update(loss.item(), target.size(0))
    acc1_meter.update(acc1.item(), target.size(0))
    acc5_meter.update(acc5.item(), target.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if idx % config.PRINT_FREQ == 0:
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        logger.info(
            f'Test: [{idx}/{len(data_loader)}]\t'
            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
            f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
            f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        logger.info(f'Mem {memory_used:.0f}MB')
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
