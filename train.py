import os
import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import project modules
from configs.defaults import CONFIG
from src.dataset import MoNuSegDataset, get_train_augmentations, get_val_augmentations
from src.model import SAMUnetHybrid
from src.trainer import train_engine
from src.utils import plot_history, visualize_predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Data Paths
    image_dir = os.path.join(CONFIG['data_path_train'], 'images')
    mask_dir = os.path.join(CONFIG['data_path_train'], 'masks')
    
    all_image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    all_mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    
    train_img, val_img, train_mask, val_mask = train_test_split(
        all_image_paths, all_mask_paths, test_size=0.2, random_state=42
    )
    
    # 2. Datasets & Loaders
    train_ds = MoNuSegDataset(train_img, train_mask, transform=get_train_augmentations(CONFIG['target_size']))
    val_ds = MoNuSegDataset(val_img, val_mask, transform=get_val_augmentations(CONFIG['target_size']))
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # 3. Model
    model = SAMUnetHybrid(
        sam2_checkpoint=CONFIG['sam_checkpoint'],
        sam2_config=CONFIG['sam_config_file']
    )
    
    # 4. Train
    model, history = train_engine(model, train_loader, val_loader, device, CONFIG)
    
    # 5. Visualize
    plot_history(history, CONFIG['history_plot_path'])
    visualize_predictions(model, val_loader, device, save_path=CONFIG['predictions_plot_path'])

if __name__ == "__main__":
    main()