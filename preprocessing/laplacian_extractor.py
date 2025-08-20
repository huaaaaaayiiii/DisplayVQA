import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import random
import cv2
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import scipy.io as scio


class LaplacianExtractor:
    
    def __init__(self, num_pyramid_levels=6, target_size=224):
        self.num_levels = num_pyramid_levels
        self.target_size = target_size
        
    def create_enhanced_pyramids(self, image):
        
        original_height, original_width = image.shape[:2]
        
        if original_width > (self.target_size + self.num_levels) and \
           original_height > (self.target_size + self.num_levels):
            
            if original_width > original_height:
                target_height = self.target_size
                target_width = int((original_width * target_height) / original_height)
            elif original_height > original_width:
                target_width = self.target_size
                target_height = int((original_height * target_width) / original_width)
            else:
                target_width = target_height = self.target_size
            
            height_step = int((original_height - target_height) / (self.num_levels - 1)) * (-1)
            width_step = int((original_width - target_width) / (self.num_levels - 1)) * (-1)
            
            height_list = list(range(original_height, target_height - 1, height_step))
            width_list = list(range(original_width, target_width - 1, width_step))
            
        elif original_width == self.target_size or original_height == self.target_size:
            height_list = [original_height] * self.num_levels
            width_list = [original_width] * self.num_levels
            
        else:
            if original_width > original_height:
                target_height = self.target_size
                target_width = int((original_width * target_height) / original_height)
            elif original_height > original_width:
                target_width = self.target_size
                target_height = int((original_height * target_width) / original_width)
            else:
                target_width = target_height = self.target_size
                
            image = cv2.resize(image, (target_width, target_height), 
                             interpolation=cv2.INTER_CUBIC)
            height_list = [target_height] * self.num_levels
            width_list = [target_width] * self.num_levels
        
        current_layer = image.copy()
        gaussian_pyramid = [current_layer]
        laplacian_pyramid = []
        edge_pyramid = []
        
        for i in range(self.num_levels - 1):
            blurred = cv2.GaussianBlur(gaussian_pyramid[i], (5, 5), 5)
            
            next_layer = cv2.resize(blurred, 
                                  (width_list[i + 1], height_list[i + 1]),
                                  interpolation=cv2.INTER_CUBIC)
            gaussian_pyramid.append(next_layer)
            
            upsampled = cv2.resize(blurred, 
                                 (width_list[i], height_list[i]),
                                 interpolation=cv2.INTER_CUBIC)
            laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
            laplacian_pyramid.append(laplacian)
            
            gray_layer = cv2.cvtColor(gaussian_pyramid[i], cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_layer, 100, 200)
            edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edge_pyramid.append(edges_3ch)
        
        gaussian_pyramid.pop(-1)
        
        return gaussian_pyramid, laplacian_pyramid, edge_pyramid
    
    def resize_pyramids(self, gaussian_pyramid, laplacian_pyramid, edge_pyramid, 
                       target_width, target_height):
        
        gaussian_resized = []
        laplacian_resized = []
        edge_resized = []
        
        for i in range(self.num_levels - 1):
            gaussian_level = cv2.resize(gaussian_pyramid[i], 
                                      (target_width, target_height),
                                      interpolation=cv2.INTER_CUBIC)
            gaussian_resized.append(gaussian_level)
            
            laplacian_level = cv2.resize(laplacian_pyramid[i],
                                       (target_width, target_height),
                                       interpolation=cv2.INTER_CUBIC)
            laplacian_resized.append(laplacian_level)
            
            edge_level = cv2.resize(edge_pyramid[i],
                                  (target_width, target_height),
                                  interpolation=cv2.INTER_CUBIC)
            edge_resized.append(edge_level)
        
        return gaussian_resized, laplacian_resized, edge_resized


class VideoLaplacianDataset(Dataset):
    
    def __init__(self, database_name, frames_directory, video_names, num_pyramid_levels=6):
        super(VideoLaplacianDataset, self).__init__()
        
        self.database_name = database_name
        self.frames_dir = Path(frames_directory)
        self.video_names = video_names
        self.num_levels = num_pyramid_levels
        
        if database_name == '8kpro':
            self.video_length = 5
        elif database_name == 'KoNViD-1k':
            self.video_length = 8
        elif database_name == 'LiveVQC':
            self.video_length = 10
        else:
            self.video_length = 8
    
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        
        if self.database_name == '8kpro':
            video_name = self.video_names[idx]
            video_id = video_name[:-4] if video_name.endswith('.mp4') else video_name
        else:
            video_id = self.video_names[idx]
        
        print(f"Processing video: {video_id}")
        
        frame_path = self.frames_dir / video_id
        
        if not frame_path.exists():
            print(f"Warning: Frame directory not found: {frame_path}")
            return torch.zeros(1), video_id
        
        first_frame_path = frame_path / "000.png"
        if not first_frame_path.exists():
            print(f"Warning: First frame not found: {first_frame_path}")
            return torch.zeros(1), video_id
        
        first_frame = Image.open(first_frame_path)
        frame_width, frame_height = first_frame.size
        
        output_features = torch.zeros([
            self.video_length * (self.num_levels - 1) * 2, 
            3, frame_height, frame_width
        ])
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        extractor = LaplacianExtractor(
            num_pyramid_levels=self.num_levels,
            target_size=224
        )
        
        random.seed(np.random.randint(20240101, 20241231))
        
        for frame_idx in range(self.video_length):
            frame_file = frame_path / f"{frame_idx:03d}.png"
            
            if not frame_file.exists():
                print(f"Warning: Frame {frame_idx} not found for video {video_id}")
                continue
            
            frame_bgr = cv2.imread(str(frame_file))
            if frame_bgr is None:
                continue
            
            gaussian_pyramid, laplacian_pyramid, edge_pyramid = \
                extractor.create_enhanced_pyramids(frame_bgr)
            
            _, laplacian_resized, edge_resized = extractor.resize_pyramids(
                gaussian_pyramid, laplacian_pyramid, edge_pyramid,
                frame_width, frame_height
            )
            
            for level_idx in range(len(edge_resized)):
                edge_features = edge_resized[level_idx]
                edge_rgb = cv2.cvtColor(edge_features, cv2.COLOR_BGR2RGB)
                edge_tensor = transform(Image.fromarray(edge_rgb))
                
                output_idx = frame_idx * (self.num_levels - 1) + level_idx
                output_features[output_idx] = edge_tensor
        
        return output_features, video_id


def attention(x):
    batch_size, channels, height, width = x.size()
    
    query = x.view(batch_size, channels, -1)
    key = query
    query = query.permute(0, 2, 1)
    
    similarity_map = torch.matmul(query, key)
    
    query_l2 = torch.norm(query, dim=2, keepdim=True)
    key_l2 = torch.norm(key, dim=1, keepdim=True)
    similarity_map = torch.div(similarity_map, 
                             torch.matmul(query_l2, key_l2).clamp(min=1e-8))
    
    return similarity_map


class ResNet18LP(nn.Module):
    
    def __init__(self, extraction_layer=2):
        super(ResNet18LP, self).__init__()
        
        resnet18 = models.resnet18(pretrained=True)
        
        if extraction_layer == 1:
            self.features = nn.Sequential(*list(resnet18.children())[:-5])
        elif extraction_layer == 2:
            self.features = nn.Sequential(*list(resnet18.children())[:-4])
        elif extraction_layer == 3:
            self.features = nn.Sequential(*list(resnet18.children())[:-3])
        else:
            self.features = nn.Sequential(*list(resnet18.children())[:-2])
        
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
        features_std = global_std_pool2d(x)
        return features_mean, features_std


class RGBExtractor(nn.Module):
    
    def __init__(self):
        super(RGBExtractor, self).__init__()
        
        self.head_rgb = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(20736, 3000),
            nn.Dropout(p=0.75),
            nn.Linear(3000, 128),
            nn.Softmax(dim=1)
        )
        self.avg_RGB = nn.AdaptiveAvgPool2d((12, 12))
        
    def forward(self, x):
        x = self.avg_RGB(x)
        x = attention(x)
        x = x.view(x.size(0), -1)
        x = self.head_rgb(x)
        x = x.view(-1, 128, 1, 1)
        return x


def global_std_pool2d(x):
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, layer=2, frame_batch_size=10, device='cuda'):
    
    extractor = ResNet18LP(layer=layer).to(device)
    rgb = RGBExtractor().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    output3 = torch.Tensor().to(device)
    extractor.eval()
    
    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            features_mean, features_std = extractor(batch)
            rgb_output = rgb(batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            output3 = torch.cat((output3, rgb_output), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        features_mean, features_std = extractor(last_batch)
        rgb_output = rgb(last_batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        output3 = torch.cat((output3, rgb_output), 0)
        output = torch.cat((output1, output2, output3), 1).squeeze()
        print(output.shape)

    return output


def main():
    
    parser = ArgumentParser(description='Laplacian Pyramid Feature Extraction')
    parser.add_argument("--seed", type=int, default=20240529)
    parser.add_argument('--database', default='8kpro', type=str,
                       help='Database name')
    parser.add_argument('--frame_batch_size', type=int, default=8,
                       help='Frame batch size for feature extraction')
    parser.add_argument('--extraction_layer', type=int, default=2,
                       help='ResNet18 layer for feature extraction')
    parser.add_argument('--num_pyramid_levels', type=int, default=6,
                       help='Number of Gaussian pyramid levels')
    parser.add_argument('--disable_gpu', action='store_true',
                       help='Disable GPU usage')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.database == '8kpro':
        frames_dir = 'data/8kpro_frames'
        annotation_file = 'data/8k_quality_annotations.csv'
        output_dir = 'data/8kpro_laplacian_features'
        
        data_info = pd.read_csv(annotation_file)
        video_names = data_info['video_name'].tolist()
        
    elif args.database == 'KoNViD-1k':
        frames_dir = 'data/konvid_frames'
        annotation_file = 'data/konvid_annotations.mat'
        output_dir = 'data/konvid_laplacian_features'
        
        data_info = scio.loadmat(annotation_file)
        video_names = []
        for i in range(len(data_info['video_names'])):
            video_names.append(data_info['video_names'][i][0][0].split('_')[0])
            
    elif args.database == 'LiveVQC':
        frames_dir = 'data/livevqc_frames'
        annotation_file = 'data/livevqc_annotations.mat'
        output_dir = 'data/livevqc_laplacian_features'
        
        mat_data = scio.loadmat(annotation_file)
        dataInfo = pd.DataFrame(mat_data['video_list'])
        dataInfo['MOS'] = mat_data['mos']
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
        video_names = dataInfo['file_names'].tolist()
    
    else:
        raise ValueError(f"Unsupported database: {args.database}")
    
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = VideoLaplacianDataset(
        database_name=args.database,
        frames_directory=frames_dir,
        video_names=video_names,
        num_pyramid_levels=args.num_pyramid_levels
    )
    
    print(f"Processing {len(dataset)} videos...")
    
    for i in range(len(dataset)):
        print(f"Processing video {i+1}/{len(dataset)}")
        
        video_features, video_id = dataset[i]
        
        if video_features.numel() == 1:
            continue
        
        enhanced_features = get_features(
            video_features, 
            args.extraction_layer, 
            args.frame_batch_size, 
            device
        )
        
        video_output_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        if args.database == '8kpro':
            video_length = 5
        elif args.database == 'LiveVQC':
            video_length = 10
        else:
            video_length = 8
        
        for frame_idx in range(video_length):
            frame_features = enhanced_features[
                frame_idx * (args.num_pyramid_levels - 1):
                (frame_idx + 1) * (args.num_pyramid_levels - 1)
            ]
            
            output_file = os.path.join(video_output_dir, f'{frame_idx:03d}.npy')
            np.save(output_file, frame_features.to('cpu').numpy())
    
    print("Laplacian feature extraction completed")


if __name__ == "__main__":
    main()