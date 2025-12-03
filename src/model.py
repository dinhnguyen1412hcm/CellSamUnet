import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate

# --- BUILDING BLOCKS ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class FusionBlock(nn.Module):
    def __init__(self, unet_ch, sam_ch, out_ch):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(unet_ch + sam_ch, out_ch, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True))
    def forward(self, unet_feat, sam_feat):
        return self.fusion(torch.cat([unet_feat, sam_feat], dim=1))

# --- SAM2 ENCODER WRAPPER ---
class HieraSam2ImageEncoder(nn.Module):
    def __init__(self, checkpoint_path, config_path, fine_tune=True, unfreeze_last_blocks=2):
        super().__init__()
        print(f"Loading SAM2 from {checkpoint_path}...")
        
        # Load config
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        cfg = OmegaConf.create(yaml_config)
        
        # Patch for Hydra/OmegaConf compatibility
        OmegaConf.set_struct(cfg, False)
        try:
            cfg.model.memory_encoder.fuser.layer.layer_scale_init_value = 1.0e-6
            for stage in cfg.model.image_encoder.hiera.stages:
                for block in stage.blocks:
                    if 'layer_scale_init_value' in block:
                        block.layer_scale_init_value = 1.0e-6
        except Exception: pass
        OmegaConf.set_struct(cfg, True)
        
        full_sam2_model = instantiate(cfg.model)
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
        full_sam2_model.load_state_dict(state_dict)
        self.encoder = full_sam2_model.image_encoder
        
        if fine_tune:
            for param in self.encoder.parameters(): param.requires_grad = False
            if unfreeze_last_blocks > 0:
                num_blocks = len(self.encoder.trunk.blocks)
                for i in range(num_blocks - unfreeze_last_blocks, num_blocks):
                    for param in self.encoder.trunk.blocks[i].parameters():
                        param.requires_grad = True
        else:
            for param in self.encoder.parameters(): param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)

# --- UNET DECODER ---
class UNetWithHieraFeatures(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        unet_ch = [64, 128, 256, 512, 1024]
        hiera_ch_map = {'fuse1': 256, 'fuse2': 256, 'fuse3': 256, 'fuse4': 256, 'fuse5': 256}
        
        self.inc = DoubleConv(n_channels, unet_ch[0])
        self.fuse1 = FusionBlock(unet_ch[0], hiera_ch_map['fuse1'], unet_ch[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(unet_ch[0], unet_ch[1]))
        self.fuse2 = FusionBlock(unet_ch[1], hiera_ch_map['fuse2'], unet_ch[1])
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(unet_ch[1], unet_ch[2]))
        self.fuse3 = FusionBlock(unet_ch[2], hiera_ch_map['fuse3'], unet_ch[2])
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(unet_ch[2], unet_ch[3]))
        self.fuse4 = FusionBlock(unet_ch[3], hiera_ch_map['fuse4'], unet_ch[3])
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(unet_ch[3], unet_ch[4]))
        self.fuse5 = FusionBlock(unet_ch[4], hiera_ch_map['fuse5'], unet_ch[4])
        self.up1 = nn.ConvTranspose2d(unet_ch[4], unet_ch[3], 2, 2)
        self.conv1 = DoubleConv(unet_ch[4], unet_ch[3])
        self.up2 = nn.ConvTranspose2d(unet_ch[3], unet_ch[2], 2, 2)
        self.conv2 = DoubleConv(unet_ch[3], unet_ch[2])
        self.up3 = nn.ConvTranspose2d(unet_ch[2], unet_ch[1], 2, 2)
        self.conv3 = DoubleConv(unet_ch[2], unet_ch[1])
        self.up4 = nn.ConvTranspose2d(unet_ch[1], unet_ch[0], 2, 2)
        self.conv4 = DoubleConv(unet_ch[1], unet_ch[0])
        self.outc = nn.Conv2d(unet_ch[0], n_classes, 1)

    def forward(self, x, hiera_features):
        u1_out = self.inc(x)
        u2_out = self.down1(u1_out)
        u3_out = self.down2(u2_out)
        u4_out = self.down3(u3_out)
        u5_out_bottleneck = self.down4(u4_out)

        hiera_f1, hiera_f2, hiera_f3 = hiera_features
        hiera_for_fuse1 = F.interpolate(hiera_f1, size=u1_out.shape[2:], mode='bilinear', align_corners=False)
        hiera_for_fuse2 = F.interpolate(hiera_f1, size=u2_out.shape[2:], mode='bilinear', align_corners=False)
        hiera_for_fuse3 = F.interpolate(hiera_f1, size=u3_out.shape[2:], mode='bilinear', align_corners=False)
        hiera_for_fuse4 = F.interpolate(hiera_f2, size=u4_out.shape[2:], mode='bilinear', align_corners=False)
        hiera_for_fuse5 = F.interpolate(hiera_f3, size=u5_out_bottleneck.shape[2:], mode='bilinear', align_corners=False)

        f1 = self.fuse1(u1_out, hiera_for_fuse1)
        f2 = self.fuse2(u2_out, hiera_for_fuse2)
        f3 = self.fuse3(u3_out, hiera_for_fuse3)
        f4 = self.fuse4(u4_out, hiera_for_fuse4)
        f5_bottleneck = self.fuse5(u5_out_bottleneck, hiera_for_fuse5)

        x = self.conv1(torch.cat([f4, self.up1(f5_bottleneck)], dim=1))
        x = self.conv2(torch.cat([f3, self.up2(x)], dim=1))
        x = self.conv3(torch.cat([f2, self.up3(x)], dim=1))
        x = self.conv4(torch.cat([f1, self.up4(x)], dim=1))
        
        return self.outc(x)

# --- HYBRID MODEL ---
class SAMUnetHybrid(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, sam2_checkpoint=None, sam2_config=None):
        super().__init__()
        # Nếu không truyền path thì người dùng tự xử lý hoặc báo lỗi
        if sam2_checkpoint is None or sam2_config is None:
             raise ValueError("Must provide sam2_checkpoint and sam2_config paths")
             
        self.sam2_encoder = HieraSam2ImageEncoder(
            checkpoint_path=sam2_checkpoint, 
            config_path=sam2_config,
            fine_tune=True, 
            unfreeze_last_blocks=2
        )
        self.unet = UNetWithHieraFeatures(n_channels=n_channels, n_classes=n_classes)

    def forward(self, x):
        out_dict = self.sam2_encoder(x)
        fpn_features = out_dict['backbone_fpn']
        # high_res, mid_res, low_res
        unet_output = self.unet(x, [fpn_features[0], fpn_features[1], fpn_features[2]])
        return unet_output