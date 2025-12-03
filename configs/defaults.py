import os


BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'monuseg_data') 
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG = {
    # Data Configs
    'data_path_train': os.path.join(DATA_DIR, 'kmms_training'),
    'data_path_test': os.path.join(DATA_DIR, 'kmms_test'),
    'target_size': (256, 256),
    
    # Model Configs
    'sam_checkpoint': 'sam2_hiera_tiny.pt', 
    'sam_config_file': 'sam2_hiera_t.yaml', 
    
    # Training Configs
    'batch_size': 4,
    'num_epochs': 200,
    'encoder_lr': 1e-5,
    'decoder_lr': 1e-4,
    'accumulation_steps': 4,
    'early_stopping_patience': 15,
    'num_workers': 2,
    
    # Output Configs
    'save_checkpoint_path': os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
    'history_plot_path': os.path.join(RESULTS_DIR, 'training_history.png'),
    'predictions_plot_path': os.path.join(RESULTS_DIR, 'predictions.png')
}