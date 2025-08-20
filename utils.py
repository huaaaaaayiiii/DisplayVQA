import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.optimize import curve_fit


def compute_performance_metrics(ground_truth, predictions):
    
    y_true = np.array(ground_truth)
    y_pred = np.array(predictions)
    
    try:
        def logistic_func(x, beta1, beta2, beta3, beta4):
            return beta1 * (0.5 - 1.0/(1 + np.exp(beta2 * (x - beta3)))) + beta4
        
        beta_init = [np.max(y_true) - np.min(y_true), 0.5, np.mean(y_pred), np.min(y_true)]
        
        popt, _ = curve_fit(logistic_func, y_pred, y_true, 
                           p0=beta_init, maxfev=10000)
        
        y_pred_fitted = logistic_func(y_pred, *popt)
        
    except (RuntimeError, TypeError):
        coeffs = np.polyfit(y_pred, y_true, 1)
        y_pred_fitted = np.polyval(coeffs, y_pred)
    
    plcc = stats.pearsonr(y_true, y_pred_fitted)[0]
    srcc = stats.spearmanr(y_true, y_pred)[0]
    krcc = stats.kendalltau(y_true, y_pred)[0]
    rmse = np.sqrt(np.mean((y_true - y_pred_fitted) ** 2))
    
    plcc = 0.0 if np.isnan(plcc) else plcc
    srcc = 0.0 if np.isnan(srcc) else srcc
    krcc = 0.0 if np.isnan(krcc) else krcc
    rmse = float('inf') if np.isnan(rmse) else rmse
    
    return plcc, srcc, krcc, rmse


def plcc_loss(y_true, y_pred):
    
    y_true_centered = y_true - torch.mean(y_true)
    y_pred_centered = y_pred - torch.mean(y_pred)
    
    numerator = torch.sum(y_true_centered * y_pred_centered)
    denominator = torch.sqrt(torch.sum(y_true_centered ** 2) * torch.sum(y_pred_centered ** 2))
    
    correlation = numerator / (denominator + 1e-8)
    
    return 1.0 - correlation


def plcc_rank_loss(y_true, y_pred, alpha=0.5):
    
    plcc_component = plcc_loss(y_true, y_pred)
    
    batch_size = y_true.size(0)
    ranking_loss = 0.0
    
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                if y_true[i] > y_true[j]:
                    ranking_loss += torch.relu(y_pred[j] - y_pred[i] + 0.1)
                elif y_true[i] < y_true[j]:
                    ranking_loss += torch.relu(y_pred[i] - y_pred[j] + 0.1)
    
    ranking_loss = ranking_loss / (batch_size * (batch_size - 1))
    
    total_loss = alpha * plcc_component + (1 - alpha) * ranking_loss
    
    return total_loss


def spearman_correlation(y_true, y_pred):
    
    def rank_tensor(tensor):
        sorted_tensor, indices = torch.sort(tensor)
        ranks = torch.empty_like(indices, dtype=torch.float)
        ranks[indices] = torch.arange(1, len(tensor) + 1, dtype=torch.float, device=tensor.device)
        return ranks
    
    ranks_true = rank_tensor(y_true)
    ranks_pred = rank_tensor(y_pred)
    
    ranks_true_centered = ranks_true - torch.mean(ranks_true)
    ranks_pred_centered = ranks_pred - torch.mean(ranks_pred)
    
    numerator = torch.sum(ranks_true_centered * ranks_pred_centered)
    denominator = torch.sqrt(torch.sum(ranks_true_centered ** 2) * torch.sum(ranks_pred_centered ** 2))
    
    correlation = numerator / (denominator + 1e-8)
    
    return correlation


def srcc_loss(y_true, y_pred):
    correlation = spearman_correlation(y_true, y_pred)
    return 1.0 - correlation


class AdaptiveLoss(nn.Module):
    
    def __init__(self, loss_types=['mse', 'plcc'], weights=None):
        super(AdaptiveLoss, self).__init__()
        
        self.loss_types = loss_types
        self.weights = weights if weights else [1.0] * len(loss_types)
        
        self.loss_functions = {}
        for loss_type in loss_types:
            if loss_type == 'mse':
                self.loss_functions[loss_type] = nn.MSELoss()
            elif loss_type == 'mae':
                self.loss_functions[loss_type] = nn.L1Loss()
            elif loss_type == 'huber':
                self.loss_functions[loss_type] = nn.HuberLoss()
            elif loss_type == 'plcc':
                self.loss_functions[loss_type] = plcc_loss
            elif loss_type == 'srcc':
                self.loss_functions[loss_type] = srcc_loss
            elif loss_type == 'plcc_rank':
                self.loss_functions[loss_type] = plcc_rank_loss
    
    def forward(self, y_true, y_pred):
        
        total_loss = 0.0
        
        for i, loss_type in enumerate(self.loss_types):
            loss_value = self.loss_functions[loss_type](y_true, y_pred)
            total_loss += self.weights[i] * loss_value
        
        return total_loss


def compute_dataset_statistics(dataset_loader):
    
    all_scores = []
    video_count = 0
    
    print("Computing dataset statistics...")
    
    for batch_idx, (_, _, scores, _, video_names) in enumerate(dataset_loader):
        all_scores.extend(scores.numpy().tolist())
        video_count += len(video_names)
        
        if batch_idx % 100 == 0:
            print(f"Processed {batch_idx + 1} batches...")
    
    all_scores = np.array(all_scores)
    
    statistics = {
        'total_videos': video_count,
        'score_mean': np.mean(all_scores),
        'score_std': np.std(all_scores),
        'score_min': np.min(all_scores),
        'score_max': np.max(all_scores),
        'score_median': np.median(all_scores),
        'score_25th_percentile': np.percentile(all_scores, 25),
        'score_75th_percentile': np.percentile(all_scores, 75),
    }
    
    return statistics


def print_dataset_info(statistics):
    
    print("\nDataset Statistics:")
    print("=" * 50)
    print(f"Total videos: {statistics['total_videos']}")
    print(f"Score range: [{statistics['score_min']:.3f}, {statistics['score_max']:.3f}]")
    print(f"Score mean: {statistics['score_mean']:.3f} ± {statistics['score_std']:.3f}")
    print(f"Score median: {statistics['score_median']:.3f}")
    print(f"25th percentile: {statistics['score_25th_percentile']:.3f}")
    print(f"75th percentile: {statistics['score_75th_percentile']:.3f}")
    print("=" * 50)


def save_predictions(ground_truth, predictions, video_names, output_file):
    
    import csv
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['video_name', 'ground_truth'] + list(predictions.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, video_name in enumerate(video_names):
            row = {
                'video_name': video_name,
                'ground_truth': ground_truth[i]
            }
            for pred_type in predictions:
                row[pred_type] = predictions[pred_type][i]
            writer.writerow(row)
    
    print(f"Predictions saved to: {output_file}")


def load_model_checkpoint(model, checkpoint_path, device):
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    print("Checkpoint loaded successfully")
    
    return model


def create_results_summary(results_dict, output_file):
    
    import json
    
    summary = {}
    
    for pred_type in results_dict:
        summary[pred_type] = {}
        for metric in results_dict[pred_type]:
            values = results_dict[pred_type][metric]
            summary[pred_type][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to: {output_file}")
    
    return summary