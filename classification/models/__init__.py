import os
import torch
import torch.nn.functional as F

from functools import partial
import torch
import torch.nn as nn
from .vHeat.vHeat import vHeat
from .vHeat.RS_vHeat import RS_vHeat, StemLayer, LayerNorm2d


class Vheat_Classification(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()

        self.encoder = encoder
        self.classifier = nn.Sequential(
            LayerNorm2d(self.encoder.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.encoder.num_features, self.encoder.num_classes),
        )

    def forward(self, x_opt):
        # outs_opt = self.encoder(x_opt)
        # x = self.classifier(outs_opt[-1])
        x_opt = self.encoder(x_opt)
        x = self.classifier(x_opt)
        return x

def build_vHeat_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["vHeat"]:
        encoder = vHeat(
            in_chans=config.MODEL.VHEAT.IN_CHANS, 
            patch_size=config.MODEL.VHEAT.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VHEAT.DEPTHS, 
            dims=config.MODEL.VHEAT.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_ratio=config.MODEL.VHEAT.MLP_RATIO,
            post_norm=config.MODEL.VHEAT.POST_NORM,
            layer_scale=config.MODEL.VHEAT.LAYER_SCALE,
            img_size=config.DATA.IMG_SIZE,
            infer_mode=config.EVAL_MODE or config.THROUGHPUT_MODE,
        )
        if config.THROUGHPUT_MODE:
            encoder.infer_init()
        return encoder
    elif model_type in ["RS_vHeat"]:
        encoder = RS_vHeat(
            in_chans=config.MODEL.VHEAT.IN_CHANS,
            patch_size=config.MODEL.VHEAT.PATCH_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VHEAT.DEPTHS,
            dims=config.MODEL.VHEAT.EMBED_DIM,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_ratio=config.MODEL.VHEAT.MLP_RATIO,
            post_norm=config.MODEL.VHEAT.POST_NORM,
            layer_scale=config.MODEL.VHEAT.LAYER_SCALE,
            img_size=config.DATA.IMG_SIZE,
            infer_mode=config.EVAL_MODE or config.THROUGHPUT_MODE,
        )
        if config.THROUGHPUT_MODE:
            encoder.infer_init()
        return encoder

    
def build_model(config, is_pretrain=False):
    model = build_vHeat_model(config, is_pretrain)
    return model
