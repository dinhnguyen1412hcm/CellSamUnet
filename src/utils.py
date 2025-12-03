import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_history(history, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(history['train_loss'], label='Train Loss'); axs[0, 0].plot(history['val_loss'], label='Val Loss'); axs[0, 0].legend()
    axs[0, 1].plot(history['val_dice'], label='Val Dice'); axs[0, 1].legend()
    axs[1, 0].plot(history['val_accuracy'], label='Val Accuracy'); axs[1, 0].legend()
    axs[1, 1].plot(history['val_hd95'], label='Val HD95'); axs[1, 1].legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def visualize_predictions(model, loader, device, num_samples=5, save_path='predictions.png'):
    model.eval()
    try:
        sample_batch = next(iter(loader))
    except StopIteration:
        return

    with torch.no_grad():
        images, masks = sample_batch['image'][:num_samples].to(device), sample_batch['mask'][:num_samples]
        preds_logits = model(images)
        preds_probs = torch.sigmoid(preds_logits).cpu()
        
    actual_num_samples = images.shape[0]
    if actual_num_samples == 0: return

    fig, axes = plt.subplots(actual_num_samples, 3, figsize=(9, 3 * actual_num_samples))
    if actual_num_samples == 1: axes = np.array([axes])
        
    for i in range(actual_num_samples): 
        img_np = images[i].cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # Access safe indexing
        ax_row = axes[i] if len(axes.shape) > 1 else axes
        
        ax_row[0].imshow(img_np); ax_row[0].set_title("Input")
        ax_row[1].imshow(masks[i].squeeze().numpy(), cmap='gray'); ax_row[1].set_title("GT")
        ax_row[2].imshow((preds_probs[i].squeeze().numpy() > 0.5), cmap='gray'); ax_row[2].set_title("Pred")
        for ax in ax_row: ax.axis('off')
        
    plt.tight_layout(); plt.savefig(save_path); plt.close()