import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from .loss import CombinedLoss, calculate_dice_score, calculate_accuracy, get_hd95

def train_engine(model, train_loader, val_loader, device, config):
    model.to(device)
    
    optimizer = optim.AdamW([
        {'params': model.sam2_encoder.parameters(), 'lr': config['encoder_lr']},
        {'params': model.unet.parameters(), 'lr': config['decoder_lr']}
    ], weight_decay=1e-5)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, verbose=True)
    criterion = CombinedLoss()
    scaler = GradScaler()
    
    accumulation_steps = config['accumulation_steps']
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_accuracy': [], 'val_hd95': []}
    epochs_no_improve = 0
    
    print(f"Start training: Epochs={config['num_epochs']}, Batch={config['batch_size']}")

    for epoch in range(config['num_epochs']):
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
        
        # Validation Phase
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
                
                hd = get_hd95(pred_binary.cpu().numpy(), masks.cpu().numpy())
                if hd > 0:
                    running_val_hd95 += hd
                    batches += 1

        # Aggregation
        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_dice = running_val_dice / len(val_loader)
        epoch_val_acc = running_val_acc / len(val_loader)
        epoch_val_hd95 = running_val_hd95 / batches if batches > 0 else 0.0
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_dice'].append(epoch_val_dice)
        history['val_accuracy'].append(epoch_val_acc)
        history['val_hd95'].append(epoch_val_hd95)
        
        print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Dice: {epoch_val_dice:.4f} | HD95: {epoch_val_hd95:.4f}")
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config['save_checkpoint_path'])
            print(f"--> Saved best model at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config['early_stopping_patience']: 
            print("Early stopping triggered."); break
            
    return model, history