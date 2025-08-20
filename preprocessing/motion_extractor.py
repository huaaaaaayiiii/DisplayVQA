import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
import scipy.io as scio
from pathlib import Path


def prepare_slowfast_input(frames, device):
    
    fast_pathway = frames
    
    slow_indices = torch.linspace(0, frames.shape[2] - 1, frames.shape[2] // 4).long()
    slow_pathway = torch.index_select(frames, 2, slow_indices)
    
    pathway_list = [slow_pathway.to(device), fast_pathway.to(device)]
    
    return pathway_list


class SlowFastNetwork(nn.Module):
    
    def __init__(self, pretrained=True):
        super(SlowFastNetwork, self).__init__()
        
        try:
            from pytorchvideo.models.hub import slowfast_r50
            slowfast_model = slowfast_r50(pretrained=pretrained)
            
            slowfast_features = nn.Sequential(*list(slowfast_model.children())[0])
            
            self.feature_extractor = nn.Sequential()
            self.slow_pooling = nn.Sequential()
            self.fast_pooling = nn.Sequential()
            self.adaptive_pooling = nn.Sequential()
            
            for layer_idx in range(5):
                self.feature_extractor.add_module(str(layer_idx), slowfast_features[layer_idx])
            
            self.slow_pooling.add_module('slow_pool', slowfast_features[5].pool[0])
            self.fast_pooling.add_module('fast_pool', slowfast_features[5].pool[1])
            self.adaptive_pooling.add_module('adaptive_pool', slowfast_features[6].output_pool)
            
        except ImportError:
            print("PyTorchVideo not available, using simplified architecture")
            self._create_simplified_architecture()
    
    def _create_simplified_architecture(self):
        
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.slow_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fast_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.adaptive_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, pathway_inputs):
        
        with torch.no_grad():
            if hasattr(self, 'feature_extractor') and len(pathway_inputs) == 2:
                pathway_features = self.feature_extractor(pathway_inputs)
                
                slow_features = self.slow_pooling(pathway_features[0])
                fast_features = self.fast_pooling(pathway_features[1])
                
                slow_features = self.adaptive_pooling(slow_features)
                fast_features = self.adaptive_pooling(fast_features)
                
            else:
                slow_input = pathway_inputs[0] if len(pathway_inputs) > 0 else pathway_inputs
                fast_input = pathway_inputs[1] if len(pathway_inputs) > 1 else pathway_inputs
                
                slow_features = self.feature_extractor(slow_input.unsqueeze(0))
                fast_features = self.feature_extractor(fast_input.unsqueeze(0))
                
                slow_features = slow_features.squeeze()
                fast_features = fast_features.squeeze()
        
        return slow_features, fast_features


class VideoMotionDataset(Dataset):
    
    def __init__(self, database_name, video_directory, annotation_file, 
                 transform, target_size, num_frames):
        super(VideoMotionDataset, self).__init__()
        
        self.database_name = database_name
        self.video_dir = Path(video_directory)
        self.transform = transform
        self.target_size = target_size
        self.num_frames = num_frames
        
        self._load_annotations(annotation_file)
        
        print(f"Loaded {len(self.video_names)} videos for motion extraction")
    
    def _load_annotations(self, annotation_file):
        
        if self.database_name == '8kpro':
            data_info = pd.read_csv(annotation_file)
            self.video_names = data_info['video_name'].tolist()
            self.scores = data_info['mos'].tolist()
            
        elif self.database_name == 'KoNViD-1k':
            mat_data = scio.loadmat(annotation_file)
            video_list = []
            score_list = []
            
            index_array = mat_data['index'][0]
            for idx in index_array:
                video_name = mat_data['video_names'][idx][0][0].split('_')[0] + '.mp4'
                video_list.append(video_name)
                score_list.append(mat_data['scores'][idx][0])
            
            self.video_names = video_list
            self.scores = score_list
            
        elif self.database_name == 'LiveVQC':
            mat_data = scio.loadmat(annotation_file)
            dataInfo = pd.DataFrame(mat_data['video_list'])
            dataInfo['MOS'] = mat_data['mos']
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
            
            self.video_names = dataInfo['file_names'].tolist()
            self.scores = dataInfo['MOS'].tolist()
            
        else:
            raise ValueError(f"Unsupported database: {self.database_name}")
    
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        
        video_name = self.video_names[idx]
        video_score = torch.FloatTensor([float(self.scores[idx])])
        
        video_path = self.video_dir / video_name
        
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            return torch.zeros(1, 3, 224, 224), 0, 'missing'
        
        video_capture = cv2.VideoCapture(str(video_path))
        
        if not video_capture.isOpened():
            print(f"Cannot open video: {video_path}")
            return torch.zeros(1, 3, 224, 224), 0, 'missing'
        
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(round(video_capture.get(cv2.CAP_PROP_FPS)))
        
        if fps == 0:
            fps = 30
        
        print(f"Video: {video_name} - Frames: {total_frames}, FPS: {fps}")
        
        video_duration = int(total_frames / fps)
        
        if self.database_name in ['KoNViD-1k']:
            min_clips = 8
        elif self.database_name in ['8kpro']:
            min_clips = 5
        elif self.database_name in ['LiveVQC']:
            min_clips = 10
        else:
            min_clips = max(1, video_duration)
        
        all_frames = torch.zeros([total_frames, 3, self.target_size, self.target_size])
        frame_count = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = self.transform(frame_pil)
            
            if frame_count < total_frames:
                all_frames[frame_count] = frame_tensor
                frame_count += 1
        
        video_capture.release()
        
        if frame_count < total_frames:
            for i in range(frame_count, total_frames):
                all_frames[i] = all_frames[frame_count - 1]
        
        video_clips = []
        
        for clip_idx in range(video_duration):
            clip_frames = torch.zeros([self.num_frames, 3, self.target_size, self.target_size])
            
            start_frame = clip_idx * fps
            end_frame = min(start_frame + self.num_frames, total_frames)
            
            if end_frame <= total_frames:
                clip_frames[:end_frame - start_frame] = all_frames[start_frame:end_frame]
                
                if end_frame - start_frame < self.num_frames:
                    for pad_idx in range(end_frame - start_frame, self.num_frames):
                        clip_frames[pad_idx] = clip_frames[end_frame - start_frame - 1]
            
            video_clips.append(clip_frames)
        
        while len(video_clips) < min_clips:
            video_clips.append(video_clips[-1] if video_clips else 
                             torch.zeros([self.num_frames, 3, self.target_size, self.target_size]))
        
        return video_clips, video_score, video_name


def extract_motion_features(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    motion_model = SlowFastNetwork(pretrained=True)
    motion_model = motion_model.to(device)
    motion_model.eval()
    
    transform = transforms.Compose([
        transforms.Resize([config.target_size, config.target_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
    
    if config.database == '8kpro':
        video_dir = './videos/8kpro'
        annotation_file = './data/8k_quality_annotations.csv'
    elif config.database == 'KoNViD-1k':
        video_dir = './videos/konvid'
        annotation_file = './data/konvid_annotations.mat'
    elif config.database == 'LiveVQC':
        video_dir = './videos/livevqc'
        annotation_file = './data/livevqc_annotations.mat'
    else:
        raise ValueError(f"Unsupported database: {config.database}")
    
    dataset = VideoMotionDataset(
        database_name=config.database,
        video_directory=video_dir,
        annotation_file=annotation_file,
        transform=transform,
        target_size=config.target_size,
        num_frames=config.num_frames
    )
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                           num_workers=config.num_workers)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"Starting motion feature extraction...")
    
    with torch.no_grad():
        for batch_idx, (video_clips, scores, video_names) in enumerate(data_loader):
            
            video_name = video_names[0]
            
            if video_name == 'missing':
                print(f"Skipping missing video at index {batch_idx}")
                continue
            
            print(f"Processing video {batch_idx + 1}: {video_name}")
            
            video_output_dir = os.path.join(config.output_dir, video_name)
            if video_name.endswith('.mp4'):
                video_output_dir = os.path.join(config.output_dir, video_name)
            else:
                video_output_dir = os.path.join(config.output_dir, video_name + '.mp4')
            
            os.makedirs(video_output_dir, exist_ok=True)
            
            for clip_idx, clip in enumerate(video_clips[0]):
                clip = clip.permute(1, 0, 2, 3)
                pathway_inputs = prepare_slowfast_input(clip, device)
                
                slow_features, fast_features = motion_model(pathway_inputs)
                
                slow_output_file = os.path.join(video_output_dir, 
                                              f'feature_{clip_idx}_slow_feature.npy')
                fast_output_file = os.path.join(video_output_dir, 
                                              f'feature_{clip_idx}_fast_feature.npy')
                
                np.save(slow_output_file, slow_features.cpu().numpy())
                np.save(fast_output_file, fast_features.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataset)} videos")
    
    print("Motion feature extraction completed")


def main():
    
    parser = argparse.ArgumentParser(description='Motion Feature Extraction')
    parser.add_argument('--database', type=str, default='8kpro',
                       choices=['8kpro', 'KoNViD-1k', 'LiveVQC'],
                       help='Database name')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--target_size', type=int, default=224,
                       help='Target image size')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames per clip')
    parser.add_argument('--output_dir', type=str, default='./data/motion_features',
                       help='Output directory for motion features')
    
    args = parser.parse_args()
    
    print("Motion Feature Extraction")
    print(f"Database: {args.database}")
    print(f"Target size: {args.target_size}")
    print(f"Frames per clip: {args.num_frames}")
    print(f"Output directory: {args.output_dir}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    extract_motion_features(args)


if __name__ == '__main__':
    main()