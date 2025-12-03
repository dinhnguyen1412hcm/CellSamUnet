import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader

from configs.defaults import CONFIG
from src.dataset import MoNuSegDataset, get_val_augmentations
from src.model import SAMUnetHybrid
from src.loss import CombinedLoss, calculate_dice_score, calculate_accuracy, get_hd95
from src.utils import visualize_predictions

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Model
    model = SAMUnetHybrid(
        sam2_checkpoint=CONFIG['sam_checkpoint'],
        sam2_config=CONFIG['sam_config_file']
    )
    
    checkpoint_path = CONFIG['save_checkpoint_path']
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Load Test Data
    image_dir = os.path.join(CONFIG['data_path_test'], 'images')
    mask_dir = os.path.join(CONFIG['data_path_test'], 'masks')
    
    # 
    all_image_paths = sorted(glob.glob(os.path.join(image_dir, '*.*')))
    all_mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    
    test_ds = MoNuSegDataset(all_image_paths, all_mask_paths, transform=get_val_augmentations(CONFIG['target_size']))
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    # 3. Evaluation Loop
    criterion = CombinedLoss()
    metrics = {'loss': [], 'dice': [], 'acc': [], 'hd95': []}
    
    print("Running Test Evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            pred_logits = model(images)
            
            # Loss
            loss = criterion(pred_logits, masks)
            metrics['loss'].append(loss.item())
            
            # Metrics
            pred_binary = (torch.sigmoid(pred_logits) > 0.5)
            metrics['dice'].append(calculate_dice_score(pred_binary, masks))
            metrics['acc'].append(calculate_accuracy(pred_binary, masks))
            
            hd = get_hd95(pred_binary.cpu().numpy(), masks.cpu().numpy())
            if hd > 0: metrics['hd95'].append(hd)
            
    print("="*30)
    print(f"TEST RESULTS (Samples: {len(test_ds)})")
    print(f"Loss: {np.mean(metrics['loss']):.4f}")
    print(f"Dice: {np.mean(metrics['dice']):.4f}")
    print(f"Acc : {np.mean(metrics['acc']):.4f}")
    print(f"HD95: {np.mean(metrics['hd95']):.4f}")
    print("="*30)
    
    visualize_predictions(model, test_loader, device, save_path=os.path.join(os.path.dirname(CONFIG['predictions_plot_path']), 'test_preds.png'))

if __name__ == "__main__":
    run_test()