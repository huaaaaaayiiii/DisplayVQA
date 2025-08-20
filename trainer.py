import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from dataset import DisplayVideoDataset
from utils import compute_performance_metrics, plcc_loss, plcc_rank_loss
from torchvision import transforms
import time
from model import DisplayVQAModel


def train_display_vqa(config):
    
    test_results = {
        'base': {'SRCC': [], 'KRCC': [], 'PLCC': [], 'RMSE': []},
        'spatial': {'SRCC': [], 'KRCC': [], 'PLCC': [], 'RMSE': []},
        'temporal': {'SRCC': [], 'KRCC': [], 'PLCC': [], 'RMSE': []},
        'combined': {'SRCC': [], 'KRCC': [], 'PLCC': [], 'RMSE': []}
    }
    
    log_directory = 'training_logs'
    os.makedirs(log_directory, exist_ok=True)

    for round_idx in range(config.num_rounds):
        config.current_round = round_idx
        print(f'Round {round_idx + 1}/{config.num_rounds}')
        
        seed = round_idx * 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _initialize_model(config, device)
        
        optimizer = optim.Adam(model.parameters(), 
                             lr=config.learning_rate, 
                             weight_decay=1e-7)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=config.lr_decay_step, 
                                            gamma=config.lr_decay_rate)
        
        criterion = _setup_criterion(config, device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Parameters: {total_params / 1e6:.2f}M')

        train_transform, test_transform = _setup_transforms(config)
        
        train_dataset, test_dataset = _load_datasets(config, train_transform, test_transform, seed)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True, 
            num_workers=config.num_workers, 
            drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1,
            shuffle=False, 
            num_workers=config.num_workers
        )

        best_performance = -1.0
        best_results = {}
        
        log_file = os.path.join(log_directory, 
                               f'{config.model_name}_{config.database}_round{round_idx}.csv')

        print('Starting training...')
        
        for epoch in range(config.num_epochs):
            model.train()
            epoch_losses = []
            
            start_time = time.time()
            
            for batch_idx, (frames, motion_feat, scores, lp_feat, _) in enumerate(train_loader):
                frames = frames.to(device)
                motion_feat = motion_feat.to(device)
                lp_feat = lp_feat.to(device)
                target_scores = scores.to(device).float()
                
                base_pred, spatial_pred, temporal_pred, combined_pred = model(
                    frames, motion_feat, lp_feat
                )
                
                optimizer.zero_grad()
                loss = criterion(target_scores, combined_pred)
                epoch_losses.append(loss.item())
                
                loss.backward()
                optimizer.step()

                if (batch_idx + 1) % (config.print_freq // config.batch_size) == 0:
                    elapsed_time = time.time() - start_time
                    avg_loss = sum(epoch_losses[-config.print_freq//config.batch_size:]) / \
                              (config.print_freq // config.batch_size)
                    print(f'Epoch {epoch+1}/{config.num_epochs} | '
                          f'Batch {batch_idx+1}/{len(train_loader)} | '
                          f'Loss: {avg_loss:.4f} | '
                          f'Time: {elapsed_time:.2f}s')
                    start_time = time.time()

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            avg_epoch_loss = np.mean(epoch_losses)
            print(f'Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f} | LR: {current_lr:.2e}')

            with torch.no_grad():
                model.eval()
                
                predictions = {'base': [], 'spatial': [], 'temporal': [], 'combined': []}
                ground_truth = []
                
                for frames, motion_feat, scores, lp_feat, _ in test_loader:
                    frames = frames.to(device)
                    motion_feat = motion_feat.to(device)
                    lp_feat = lp_feat.to(device)
                    
                    base_pred, spatial_pred, temporal_pred, combined_pred = model(
                        frames, motion_feat, lp_feat
                    )
                    
                    predictions['base'].append(base_pred.item())
                    predictions['spatial'].append(spatial_pred.item())
                    predictions['temporal'].append(temporal_pred.item())
                    predictions['combined'].append(combined_pred.item())
                    ground_truth.append(scores.item())

                results = {}
                for pred_type in predictions:
                    plcc, srcc, krcc, rmse = compute_performance_metrics(
                        ground_truth, predictions[pred_type]
                    )
                    results[pred_type] = {'PLCC': plcc, 'SRCC': srcc, 'KRCC': krcc, 'RMSE': rmse}
                    
                    print(f'{pred_type.capitalize()} - '
                          f'SRCC: {srcc:.4f}, KRCC: {krcc:.4f}, '
                          f'PLCC: {plcc:.4f}, RMSE: {rmse:.4f}')

                if results['combined']['SRCC'] > best_performance:
                    best_performance = results['combined']['SRCC']
                    best_results = results.copy()
                    
                    model_path = f"checkpoints/{config.database}_{best_performance:.4f}_epoch{epoch+1}.pth"
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(model.state_dict(), model_path)
                    print(f'Best model saved: {model_path}')

                _log_results(log_file, epoch, results)

        print('Training completed')
        
        for pred_type in best_results:
            for metric in best_results[pred_type]:
                test_results[pred_type][metric].append(best_results[pred_type][metric])

        _print_round_summary(round_idx, best_results)

    _print_final_summary(test_results)


def _initialize_model(config, device):
    if config.model_name == 'DisplayVQA_Modular':
        if config.database == 'LiveVQC_indep':
            video_length = 10
        elif config.database in ['8kpro_indep']:
            video_length = 5
        else:
            video_length = 8
            
        model = DisplayVQAModel(
            sequence_length=video_length,
            enable_spatial=True,
            enable_temporal=True,
            spatial_dropout=0.2,
            temporal_dropout=0.2
        )
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    model = model.to(device)
    return model


def _setup_criterion(config, device):
    if config.loss_type == 'plcc':
        return plcc_loss
    elif config.loss_type == 'plcc_rank':
        return plcc_rank_loss
    elif config.loss_type == 'mse':
        return nn.MSELoss().to(device)
    elif config.loss_type == 'mae':
        return nn.L1Loss().to(device)
    elif config.loss_type == 'huber':
        return nn.HuberLoss().to(device)
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")


def _setup_transforms(config):
    train_transform = transforms.Compose([
        transforms.Resize(config.resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(config.resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def _load_datasets(config, train_transform, test_transform, seed):
    if config.database == '8kpro_indep':
        data_root = 'data/8kpro_frames'
        annotation_file = 'data/8k_quality_annotations.csv'
        lp_root = 'data/8kpro_laplacian_features'
        motion_root = 'data/8kpro_motion_features'
    elif config.database == 'KoNViD-1k':
        data_root = 'data/konvid_frames'
        annotation_file = 'data/konvid_annotations.mat'
        lp_root = 'data/konvid_laplacian_features'
        motion_root = 'data/konvid_motion_features'
    elif config.database == 'LiveVQC':
        data_root = 'data/livevqc_frames'
        annotation_file = 'data/livevqc_annotations.mat'
        lp_root = 'data/livevqc_laplacian_features'
        motion_root = 'data/livevqc_motion_features'
    else:
        raise ValueError(f"Dataset configuration not found for: {config.database}")
    
    train_dataset = DisplayVideoDataset(
        data_root=data_root,
        motion_feature_root=motion_root,
        laplacian_root=lp_root,
        annotation_file=annotation_file,
        transform=train_transform,
        database_type=config.database + '_train',
        input_size=config.input_size,
        feature_mode='Fast',
        random_seed=seed
    )
    
    test_dataset = DisplayVideoDataset(
        data_root=data_root,
        motion_feature_root=motion_root,
        laplacian_root=lp_root,
        annotation_file=annotation_file,
        transform=test_transform,
        database_type=config.database + '_test',
        input_size=config.input_size,
        feature_mode='Fast',
        random_seed=seed
    )
    
    return train_dataset, test_dataset


def _log_results(log_file, epoch, results):
    with open(log_file, 'a+', newline='') as f:
        f.write(f"Epoch {epoch+1}:\n")
        for pred_type in results:
            metrics = results[pred_type]
            f.write(f"{pred_type}: SRCC={metrics['SRCC']:.4f}, "
                   f"KRCC={metrics['KRCC']:.4f}, "
                   f"PLCC={metrics['PLCC']:.4f}, "
                   f"RMSE={metrics['RMSE']:.4f}\n")
        f.write("-" * 50 + "\n")


def _print_round_summary(round_idx, results):
    print(f"\nRound {round_idx+1} Best Results:")
    print("=" * 60)
    for pred_type in results:
        metrics = results[pred_type]
        print(f"{pred_type.capitalize():>10}: "
              f"SRCC={metrics['SRCC']:.4f}, "
              f"KRCC={metrics['KRCC']:.4f}, "
              f"PLCC={metrics['PLCC']:.4f}, "
              f"RMSE={metrics['RMSE']:.4f}")


def _print_final_summary(results):
    print(f"\nFINAL RESULTS (Median across all rounds):")
    print("=" * 70)
    
    for pred_type in results:
        metrics = results[pred_type]
        median_results = {
            metric: np.median(values) for metric, values in metrics.items()
        }
        print(f"{pred_type.capitalize():>10}: "
              f"SRCC={median_results['SRCC']:.4f}, "
              f"KRCC={median_results['KRCC']:.4f}, "
              f"PLCC={median_results['PLCC']:.4f}, "
              f"RMSE={median_results['RMSE']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DisplayVQA Training')

    parser.add_argument('--database', type=str, required=True,
                       help='Database name (8kpro_indep, KoNViD-1k, LiveVQC)')
    parser.add_argument('--model_name', type=str, default='DisplayVQA_Modular',
                       help='Model architecture name')

    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95,
                       help='Learning rate decay factor')
    parser.add_argument('--lr_decay_step', type=int, default=2,
                       help='Learning rate decay step size')
    parser.add_argument('--num_rounds', type=int, default=10,
                       help='Number of training rounds')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--print_freq', type=int, default=100,
                       help='Print frequency during training')

    parser.add_argument('--resize_size', type=int, default=256,
                       help='Image resize size')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Model input size')
    parser.add_argument('--loss_type', type=str, default='plcc',
                       choices=['plcc', 'plcc_rank', 'mse', 'mae', 'huber'],
                       help='Loss function type')

    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='Use multiple GPUs for training')
    parser.add_argument('--gpu_ids', type=list, default=None,
                       help='GPU device IDs to use')

    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pretrained model checkpoint')

    config = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    print("DisplayVQA Training")
    print(f"Database: {config.database}")
    print(f"Model: {config.model_name}")
    print(f"Training rounds: {config.num_rounds}")
    print(f"Epochs per round: {config.num_epochs}")
    
    train_display_vqa(config)