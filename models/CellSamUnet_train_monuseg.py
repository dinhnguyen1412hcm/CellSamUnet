# =============================================================================
# SECTION 0: TẢI DỮ LIỆU VÀ CÀI ĐẶT (ĐÃ CẬP NHẬT CHO MONUSEG)
# =============================================================================
# Tải dữ liệu MoNuSeg (Train)
!gdown --id 1VAEQRCPDhJ2V4iN1qJpV4ppAAk6D4_dx -O monuseg_train.zip

# Giải nén vào thư mục cụ thể
!unzip -q monuseg_train.zip -d monuseg_data/

# Cài đặt các thư viện cần thiết (giữ nguyên)
!pip install -q monai albumentations pyyaml scikit-learn

# Tải trọng số và mã nguồn SAM2 (giữ nguyên)
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
!git clone https://github.com/facebookresearch/sam2.git
%pip install -e sam2/



# =============================================================================
# SECTION 1: IMPORTS (ĐÃ SỬA LỖI)
# =============================================================================
import sys
sys.path.append('/kaggle/working/sam2')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler # <-- ĐÃ SỬA LỖI
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import warnings
from typing import Dict, Tuple, Optional, Any
import glob 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.metrics import compute_hausdorff_distance
from torch.cuda.amp import GradScaler, autocast
import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split 

warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 2: DATA LOADING AND PREPARATION (CẬP NHẬT CHO MONUSEG - ĐÃ SỬA LỖI)
# =============================================================================

# Thay thế BBBC038Dataset bằng MoNuSegDataset
class MoNuSegDataset(Dataset):
    """
    Dataset tùy chỉnh cho MoNuSeg, đọc từ 2 thư mục images/ và masks/
    """
    def __init__(self, image_paths: list, mask_paths: list, transform: Optional[A.Compose] = None):
        """
        Khởi tạo dataset.
        :param image_paths: Danh sách đường dẫn đến các file ảnh.
        :param mask_paths: Danh sách đường dẫn đến các file mask tương ứng.
        :param transform: Các phép biến đổi Albumentations.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        # Đảm bảo số lượng ảnh và mask khớp nhau
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Số lượng ảnh ({len(self.image_paths)}) và mask ({len(self.mask_paths)}) không khớp."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # 1. Tải ảnh
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển sang RGB
        
        # 2. Tải mask
        mask_path = self.mask_paths[idx]
        # Đọc mask dưới dạng grayscale (1 kênh)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
        
        if mask is None:
            # Xử lý nếu file mask bị lỗi
            print(f"Warning: Không thể đọc mask tại {mask_path}. Tạo mask rỗng.")
            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype=np.float32)

        # === SỬA LỖI VALUEERROR BẮT ĐẦU TỪ ĐÂY ===
        # Lấy kích thước H, W từ ảnh
        img_h, img_w, _ = image.shape
        
        # Kiểm tra và resize mask NẾU nó không khớp kích thước với ảnh
        if mask.shape[0] != img_h or mask.shape[1] != img_w:
            # Resize mask về đúng kích thước của ảnh
            # (img_w, img_h) là (width, height) theo yêu cầu của cv2.resize
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        # === KẾT THÚC SỬA LỖI ===

        # 3. Chuẩn hóa
        image = (image / 255.0).astype(np.float32)
        # Binarize mask: coi tất cả các loại tế bào > 0 là 1 (foreground)
        mask = (mask > 0).astype(np.float32) 
        
        # 4. Áp dụng Augmentations
        # Bây giờ image và mask đã được đảm bảo có cùng kích thước
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image_tensor, mask_tensor = augmented['image'], augmented['mask']
        else:
            # Fallback nếu không có transform
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask)

        # 5. Đảm bảo mask có 1 kênh (channel dimension)
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        return {'image': image_tensor.float(), 'mask': mask_tensor.float()}


# Các hàm Augmentation (Giữ nguyên)
def get_train_augmentations(target_size: Tuple[int, int]):
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]), A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.7), A.OneOf([A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5)], p=0.7),
        A.GaussNoise(p=0.2), A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=1.0), ToTensorV2(),
    ])

def get_val_augmentations(target_size: Tuple[int, int]):
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=1.0), ToTensorV2()
    ])

# =============================================================================
# SECTION 3: MODEL ARCHITECTURE (GIỮ NGUYÊN - ĐÃ SỬA LỖI)
# =============================================================================

class HieraSam2ImageEncoder(nn.Module):
    def __init__(self, checkpoint_path="sam2_hiera_tiny.pt", fine_tune=True, unfreeze_last_blocks=2):
        super().__init__()
        print("Building SAM2 model with Fine-Tuning enabled...")
        config_file_path = "/kaggle/working/sam2/sam2/configs/sam2/sam2_hiera_t.yaml"
        checkpoint_full_path = f"/kaggle/working/{checkpoint_path}"
        with open(config_file_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        cfg = OmegaConf.create(yaml_config)
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
        state_dict = torch.load(checkpoint_full_path, map_location="cpu")["model"]
        full_sam2_model.load_state_dict(state_dict)
        self.encoder = full_sam2_model.image_encoder
        print("Hiera Image Encoder extracted successfully.")
        if fine_tune:
            print(f"Fine-tuning mode: Freezing all layers first...")
            for param in self.encoder.parameters():
                param.requires_grad = False
            if unfreeze_last_blocks > 0:
                print(f"Unfreezing the last {unfreeze_last_blocks} block(s) of the encoder's trunk...")
                num_blocks = len(self.encoder.trunk.blocks)
                for i in range(num_blocks - unfreeze_last_blocks, num_blocks):
                    for param in self.encoder.trunk.blocks[i].parameters():
                        param.requires_grad = True
        else:
            print("Encoder weights are completely frozen.")
            for param in self.encoder.parameters():
                param.requires_grad = False
    def forward(self, x):
        return self.encoder(x)

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
        # === PHẦN 1: TÍNH TOÁN TOÀN BỘ NHÁNH U-NET ENCODER TRƯỚC ===
        u1_out = self.inc(x)
        u2_out = self.down1(u1_out)
        u3_out = self.down2(u2_out)
        u4_out = self.down3(u3_out)
        u5_out_bottleneck = self.down4(u4_out)

        # === PHẦN 2: CHUẨN BỊ CÁC ĐẶC TRƯNG TỪ SAM2 (như cũ) ===
        hiera_f1, hiera_f2, hiera_f3 = hiera_features
        hiera_for_fuse1 = F.interpolate(hiera_f1, size=u1_out.shape[2:], mode='bilinear', align_corners=False)
        hiera_for_fuse2 = F.interpolate(hiera_f1, size=u2_out.shape[2:], mode='bilinear', align_corners=False)
        hiera_for_fuse3 = F.interpolate(hiera_f1, size=u3_out.shape[2:], mode='bilinear', align_corners=False)
        hiera_for_fuse4 = F.interpolate(hiera_f2, size=u4_out.shape[2:], mode='bilinear', align_corners=False)
        hiera_for_fuse5 = F.interpolate(hiera_f3, size=u5_out_bottleneck.shape[2:], mode='bilinear', align_corners=False)

        # === PHẦN 3: HỢP NHẤT (FUSION) SONG SONG TẠI MỖI CẤP ===
        f1 = self.fuse1(u1_out, hiera_for_fuse1)
        f2 = self.fuse2(u2_out, hiera_for_fuse2)
        f3 = self.fuse3(u3_out, hiera_for_fuse3)
        f4 = self.fuse4(u4_out, hiera_for_fuse4)
        f5_bottleneck = self.fuse5(u5_out_bottleneck, hiera_for_fuse5)

        # === PHẦN 4: DECODER SỬ DỤNG CÁC ĐẶC TRƯNG ĐÃ ĐƯỢC HỢP NHẤT ===
        x = self.conv1(torch.cat([f4, self.up1(f5_bottleneck)], dim=1))
        x = self.conv2(torch.cat([f3, self.up2(x)], dim=1))
        x = self.conv3(torch.cat([f2, self.up3(x)], dim=1))
        x = self.conv4(torch.cat([f1, self.up4(x)], dim=1))
        
        return self.outc(x)

class SAMUnetHybrid(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, sam2_checkpoint="sam2_hiera_tiny.pt"):
        super().__init__()
        self.sam2_encoder = HieraSam2ImageEncoder(checkpoint_path=sam2_checkpoint, fine_tune=True, unfreeze_last_blocks=2)
        self.unet = UNetWithHieraFeatures(n_channels=n_channels, n_classes=n_classes)

    
    def forward(self, x):
        out_dict = self.sam2_encoder(x)
        
        # Lấy các feature maps đa tỷ lệ từ Feature Pyramid Network (FPN)
        fpn_features = out_dict['backbone_fpn']

        # Gán lại cho đúng với kích thước
        high_res_features = fpn_features[0]  # Kích thước 64x64
        mid_res_features  = fpn_features[1]  # Kích thước 32x32
        low_res_features  = fpn_features[2]  # Kích thước 16x16
        
        unet_output = self.unet(x, [high_res_features, mid_res_features, low_res_features])
        
        return unet_output
# =================================================================
# =================================================================

# =============================================================================
# SECTION 4 & 5: LOSS, METRICS, TRAINING LOOP (GIỮ NGUYÊN - ĐÃ SỬA LỖI)
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): 
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_score

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss() # Dùng phiên bản an toàn
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    def forward(self, pred_logits, target):
        bce = self.bce_loss(pred_logits, target)
        pred_probs = torch.sigmoid(pred_logits) # Cần sigmoid cho Dice
        dice = self.dice_loss(pred_probs, target)
        return self.bce_weight * bce + self.dice_weight * dice

def calculate_dice_score(pred, target, smooth=1.0):
    pred_flat, target_flat = pred.view(-1), target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)).item()

def calculate_accuracy(pred, target):
    return ((pred > 0.5).float() == target).float().mean().item()

def train_model(model, train_loader, val_loader, device, config):
    model.to(device)
    print("Setting up optimizer with differential learning rates...")
    optimizer = optim.AdamW([
        {'params': model.sam2_encoder.parameters(), 'lr': config['encoder_lr']},
        {'params': model.unet.parameters(), 'lr': config['decoder_lr']}
    ], weight_decay=1e-5)
    print(f" -> Encoder LR: {config['encoder_lr']}, Decoder LR: {config['decoder_lr']}")
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, verbose=True)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    scaler = GradScaler()
    accumulation_steps = 4
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_accuracy': [], 'val_hd95': []}
    epochs_no_improve = 0
    print(f"Starting training with Batch Size: {config['batch_size']}, Accumulation Steps: {accumulation_steps}")
    print(f"Effective Batch Size: {config['batch_size'] * accumulation_steps}")
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            with autocast():
                pred_logits = model(images)
                loss = criterion(pred_logits, masks)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            running_train_loss += loss.item() * accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader): 
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        model.eval()
        running_val_loss, running_val_dice, running_val_acc, running_val_hd95, batches = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch['image'].to(device), batch['mask'].to(device)
                pred_logits = model(images)
                loss = criterion(pred_logits, masks)
                running_val_loss += loss.item()
                pred_probs = torch.sigmoid(pred_logits)
                pred_binary = (pred_probs > 0.5)
                running_val_dice += calculate_dice_score(pred_binary, masks)
                running_val_acc += calculate_accuracy(pred_binary, masks)
                try:
                    if images.shape[0] > 0:
                        hd_val = compute_hausdorff_distance(pred_binary.cpu().numpy(), masks.cpu().numpy(), include_background=True, percentile=95)
                        if not np.isnan(hd_val.mean()):
                            running_val_hd95 += hd_val.mean().item()
                            batches += 1
                except: pass

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_dice = running_val_dice / len(val_loader)
        epoch_val_acc = running_val_acc / len(val_loader)
        epoch_val_hd95 = running_val_hd95 / batches if batches > 0 else 0.0
        history['train_loss'].append(epoch_train_loss); history['val_loss'].append(epoch_val_loss)
        history['val_dice'].append(epoch_val_dice); history['val_accuracy'].append(epoch_val_acc); history['val_hd95'].append(epoch_val_hd95)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Dice: {epoch_val_dice:.4f} | Val Acc: {epoch_val_acc:.4f} | Val HD95: {epoch_val_hd95:.4f}")
        
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = epoch_val_loss, 0
            torch.save(model.state_dict(), config['save_checkpoint_path'])
            print(f"Checkpoint saved at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config['early_stopping_patience']: 
            print(f"Early stopping at epoch {epoch + 1}."); break
            
    model.load_state_dict(torch.load(config['save_checkpoint_path']))
    return model, history

# =============================================================================
# SECTION 6: VISUALIZATION & MAIN EXECUTION (ĐÃ SỬA LỖI ĐƯỜNG DẪN)
# =============================================================================

# Hàm plot_history giữ nguyên
def plot_history(history, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(history['train_loss'], label='Train Loss'); axs[0, 0].plot(history['val_loss'], label='Val Loss'); axs[0, 0].legend()
    axs[0, 1].plot(history['val_dice'], label='Val Dice'); axs[0, 1].legend()
    axs[1, 0].plot(history['val_accuracy'], label='Val Accuracy'); axs[1, 0].legend()
    axs[1, 1].plot(history['val_hd95'], label='Val HD95'); axs[1, 1].legend()
    plt.tight_layout(); plt.savefig(save_path); plt.show()

# Hàm visualize_predictions (ĐÃ SỬA LỖI INDEXERROR)
def visualize_predictions(model, loader, device, num_samples=5, save_path='predictions.png'):
    model.eval()
    try:
        sample_batch = next(iter(loader))
    except StopIteration:
        print("Loader is empty. Cannot visualize predictions.")
        return

    with torch.no_grad():
        images, masks = sample_batch['image'][:num_samples].to(device), sample_batch['mask'][:num_samples]
        preds_logits = model(images)
        preds_probs = torch.sigmoid(preds_logits).cpu()
        
    # Lấy số lượng ảnh thực tế trong batch (để sửa lỗi IndexError)
    actual_num_samples = images.shape[0] 
    
    fig, axes = plt.subplots(actual_num_samples, 3, figsize=(9, 3 * actual_num_samples))
    
    if actual_num_samples == 1: 
        axes = [axes]
        
    for i in range(actual_num_samples): 
        img_np = images[i].cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        axes[i, 0].imshow(img_np); axes[i, 0].set_title("Input")
        axes[i, 1].imshow(masks[i].squeeze().numpy(), cmap='gray'); axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow((preds_probs[i].squeeze().numpy() > 0.5), cmap='gray'); axes[i, 2].set_title("Prediction")
        for ax in axes[i]: ax.axis('off')
        
    plt.tight_layout(); plt.savefig(save_path); plt.show()

def main():
    config = {
        # === SỬA LỖI ĐƯỜNG DẪN TẠI ĐÂY ===
        # Trỏ đến thư mục 'monuseg_training' bên trong 'monuseg_data'
        'data_dir': '/kaggle/working/monuseg_data/kmms_training/', 
        
        'sam_checkpoint': 'sam2_hiera_tiny.pt',
        'target_size': (256, 256),
        'batch_size': 4, 
        'num_epochs': 200,
        'decoder_lr': 1e-4,
        'encoder_lr': 1e-5,
        'early_stopping_patience': 15,
        
        'save_checkpoint_path': 'sam_unet_monuseg_checkpoint.pth',
        'history_plot_path': 'training_history_monuseg.png',
        'predictions_plot_path': 'predictions_monuseg.png'
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Logic quét và chia thư mục đã CẬP NHẬT
    data_dir = config['data_dir']
    
    # Tìm tất cả các file ảnh và mask
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')

    # === SỬA LỖI GLOB ===
    # Tìm chính xác file .tif và .png
    all_image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    all_mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

    if not all_image_paths or not all_mask_paths:
        print(f"Error: Không tìm thấy file ảnh/mask nào trong '{image_dir}' hoặc '{mask_dir}'.")
        print("Hãy kiểm tra lại cấu trúc thư mục đã giải nén.")
        return
    
    if len(all_image_paths) != len(all_mask_paths):
        print(f"Error: Số lượng ảnh ({len(all_image_paths)}) và mask ({len(all_mask_paths)}) không khớp.")
        return

    # Chia danh sách các đường dẫn file
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        all_image_paths, all_mask_paths, test_size=0.2, random_state=42
    )
    
    print(f"Found {len(all_image_paths)} total samples.")
    print(f"Splitting into {len(train_img_paths)} training samples and {len(val_img_paths)} validation samples.")

    train_augs = get_train_augmentations(config['target_size'])
    val_augs = get_val_augmentations(config['target_size'])
    
    # Sử dụng MoNuSegDataset mới
    train_dataset = MoNuSegDataset(train_img_paths, train_mask_paths, transform=train_augs)
    val_dataset = MoNuSegDataset(val_img_paths, val_mask_paths, transform=val_augs)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    # Khởi tạo mô hình (giữ nguyên)
    model = SAMUnetHybrid(sam2_checkpoint=config['sam_checkpoint'])
    
    trained_model, history = train_model(model, train_loader, val_loader, device, config)
    
    if history:
        plot_history(history, config['history_plot_path'])
        visualize_predictions(trained_model, val_loader, device, save_path=config['predictions_plot_path'])

if __name__ == "__main__":
    main()