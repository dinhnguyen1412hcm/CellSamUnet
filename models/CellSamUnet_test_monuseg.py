
# =============================================================================
# CELL 2: HÀM ĐÁNH GIÁ ĐẦY ĐỦ TRÊN TẬP TEST (MONUSEG - ĐÃ SỬA LỖI)
# =============================================================================

def run_evaluation_on_test_set(model_checkpoint_path, test_data_dir, device, config):
    """
    Tải mô hình SAM2-UNet đã train và chạy đánh giá đầy đủ (tính metrics) 
    trên tập test MoNuSeg.
    
    LƯU Ý: Hàm này giả định các hàm/lớp sau ĐÃ TỒN TẠI TỪ TRƯỚC:
    - SAMUnetHybrid()
    - MoNuSegDataset() (đã có code fix lỗi ValueError)
    - get_val_augmentations()
    - CombinedLoss()
    - calculate_dice_score(), calculate_accuracy()
    - compute_hausdorff_distance()
    - visualize_predictions()
    """
    
    # 1. Khởi tạo kiến trúc mô hình và tải trọng số đã lưu
    model = SAMUnetHybrid() 
    
    # Kiểm tra xem file checkpoint có tồn tại không
    if not os.path.exists(model_checkpoint_path):
        print(f"LỖI: Không tìm thấy file checkpoint tại: {model_checkpoint_path}")
        print("Vui lòng kiểm tra lại tên file hoặc đảm bảo quá trình training đã lưu file.")
        return

    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval() # Chuyển sang chế độ dự đoán
    print(f"Model loaded from {model_checkpoint_path} and set to eval mode.")

    # 2. Chuẩn bị Test DataLoader
    image_dir = os.path.join(test_data_dir, 'images')
    mask_dir = os.path.join(test_data_dir, 'masks')

    # === SỬA LỖI GLOB TẠI ĐÂY ===
    # Tìm TẤT CẢ các file (*.*) trong images, không chỉ .tif
    all_image_paths = sorted(glob.glob(os.path.join(image_dir, '*.*')))
    all_mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    
    if not all_image_paths or not all_mask_paths:
        print(f"Error: Không tìm thấy file ảnh/mask nào trong '{image_dir}' hoặc '{mask_dir}'.")
        return

    if len(all_image_paths) != len(all_mask_paths):
        print(f"Error: Số lượng ảnh test ({len(all_image_paths)}) và mask test ({len(all_mask_paths)}) không khớp.")
        print("Kiểm tra lại 2 thư mục. Đảm bảo chúng có cùng số lượng file.")
        return

    print(f"Found {len(all_image_paths)} test samples.")
    
    # Dùng get_val_augmentations (chỉ resize, normalize)
    test_augs = get_val_augmentations(config['target_size'])
    # Sử dụng MoNuSegDataset (đã có sẵn fix lỗi ValueError)
    test_dataset = MoNuSegDataset(all_image_paths, all_mask_paths, transform=test_augs)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    # 3. Chạy vòng lặp đánh giá
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    running_test_loss, running_test_dice, running_test_acc, running_test_hd95, batches = 0.0, 0.0, 0.0, 0.0, 0
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            
            pred_logits = model(images)
            loss = criterion(pred_logits, masks)
            running_test_loss += loss.item()
            
            pred_probs = torch.sigmoid(pred_logits)
            pred_binary = (pred_probs > 0.5)
            
            running_test_dice += calculate_dice_score(pred_binary, masks)
            running_test_acc += calculate_accuracy(pred_binary, masks)
            
            try:
                if images.shape[0] > 0:
                    hd_val = compute_hausdorff_distance(pred_binary.cpu().numpy(), masks.cpu().numpy(), include_background=True, percentile=95)
                    if not np.isnan(hd_val.mean()):
                        running_test_hd95 += hd_val.mean().item()
                        batches += 1
            except:
                pass # Bỏ qua nếu có lỗi tính HD95

    # 4. Tính toán và In kết quả trung bình
    final_loss = running_test_loss / len(test_loader)
    final_dice = running_test_dice / len(test_loader)
    final_acc = running_test_acc / len(test_loader)
    final_hd95 = running_test_hd95 / batches if batches > 0 else 0.0

    print("\n" + "="*30)
    print("      FINAL TEST SET EVALUATION (MONUSEG)     ")
    print("="*30)
    print(f"Test Loss (Combined): {final_loss:.4f}")
    print(f"Test Dice Score:      {final_dice:.4f}")
    print(f"Test Accuracy:        {final_acc:.4f}")
    print(f"Test HD95:            {final_hd95:.4f}")
    print("="*30 + "\n")

    # 5. Visualize một vài dự đoán từ tập test
    print("Visualizing predictions from the test set...")
    # Tái sử dụng hàm visualize_predictions (đã sửa lỗi) từ code train
    visualize_predictions(
        model, 
        test_loader, 
        device, 
        num_samples=5, 
        save_path=config['predictions_plot_path']
    )

# =============================================================================
# CELL 3: HÀM MAIN ĐỂ CHẠY ĐÁNH GIÁ TEST
# =============================================================================

def main_test():
    # Configs cần thiết cho test
    TEST_CONFIG = {
        # Thư mục dữ liệu test đã giải nén
        # (Giả sử file zip tạo ra thư mục 'monuseg_test' bên trong 'monuseg_test_data')
        'test_data_dir': '/kaggle/working/monuseg_test_data/kmms_test/', 
        
        # File checkpoint đã lưu từ quá trình train MoNuSeg
        'model_checkpoint_path': '/kaggle/working/sam_unet_monuseg_checkpoint.pth', 
        
        'target_size': (256, 256), # Phải giống hệt lúc train
        'batch_size': 8, # Có thể tăng batch size khi test/eval
        'predictions_plot_path': 'final_test_predictions_monuseg.png'
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for testing.")

    # Chạy hàm đánh giá
    run_evaluation_on_test_set(
        TEST_CONFIG['model_checkpoint_path'],
        TEST_CONFIG['test_data_dir'],
        device,
        TEST_CONFIG
    )

# =================================
# CHẠY HÀM TEST
# =================================
if __name__ == "__main__":
    main_test()