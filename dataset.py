import os
import csv
import random
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
from random import shuffle


def parse_float_with_comma(value):
    return float(value.replace(",", "."))


class DisplayVideoDataset(data.Dataset):
    
    def __init__(self, data_root, motion_feature_root, laplacian_root, 
                 annotation_file, transform, database_type, input_size, 
                 feature_mode, random_seed=0):
        super(DisplayVideoDataset, self).__init__()

        self.database_type = database_type
        self.data_root = data_root
        self.motion_feature_root = motion_feature_root
        self.laplacian_root = laplacian_root
        self.transform = transform
        self.input_size = input_size
        self.feature_mode = feature_mode
        
        self._load_annotations(annotation_file, random_seed)
        print(f"Loaded {len(self.video_names)} videos from {database_type}")

    def _load_annotations(self, annotation_file, seed):
        if self.database_type.startswith('KoNViD'):
            self._load_konvid_annotations(annotation_file, seed)
        elif self.database_type.startswith('LiveVQC'):
            self._load_livevqc_annotations(annotation_file, seed)
        elif self.database_type.startswith('8kpro'):
            self._load_8kpro_annotations(annotation_file, seed)
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def _load_konvid_annotations(self, annotation_file, seed):
        mat_data = scio.loadmat(annotation_file)
        video_list = []
        score_list = []
        index_array = mat_data['index'][0]
        
        for idx in index_array:
            video_name = mat_data['video_names'][idx][0][0].split('_')[0] + '.mp4'
            video_list.append(video_name)
            score_list.append(mat_data['scores'][idx][0])

        if self.database_type == 'KoNViD-1k':
            self.video_names = video_list
            self.scores = score_list
        else:
            self._split_dataset(video_list, score_list, seed)

    def _load_livevqc_annotations(self, annotation_file, seed):
        mat_data = scio.loadmat(annotation_file)
        dataInfo = pd.DataFrame(mat_data['video_list'])
        dataInfo['MOS'] = mat_data['mos']
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
        
        video_list = dataInfo['file_names'].tolist()
        score_list = dataInfo['MOS'].tolist()
        
        if self.database_type == 'LiveVQC':
            self.video_names = video_list
            self.scores = score_list
        else:
            self._split_dataset(video_list, score_list, seed)

    def _load_8kpro_annotations(self, annotation_file, seed):
        data_info = pd.read_csv(annotation_file)
        video_list = data_info['video_name'].tolist()
        score_list = data_info['mos'].tolist()
        
        if self.database_type == '8kpro':
            self.video_names = video_list
            self.scores = score_list
        else:
            self._split_dataset(video_list, score_list, seed)

    def _split_dataset(self, video_list, score_list, seed):
        data_frame = pd.DataFrame({'video_names': video_list, 'scores': score_list})
        total_count = len(data_frame)
        
        random.seed(seed)
        np.random.seed(seed)
        random_indices = np.random.permutation(total_count)
        
        train_indices = random_indices[:int(total_count * 0.6)]
        val_indices = random_indices[int(total_count * 0.6):int(total_count * 0.8)]
        test_indices = random_indices[int(total_count * 0.8):]
        
        if self.database_type.endswith('_train'):
            selected_indices = train_indices
        elif self.database_type.endswith('_val'):
            selected_indices = val_indices
        elif self.database_type.endswith('_test'):
            selected_indices = test_indices
        else:
            selected_indices = random_indices
            
        self.video_names = data_frame.iloc[selected_indices]['video_names'].tolist()
        self.scores = data_frame.iloc[selected_indices]['scores'].tolist()

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        if self.database_type.startswith('KoNViD'):
            video_name = self.video_names[idx]
            video_id = video_name[:-4]
        elif self.database_type.startswith('LiveVQC'):
            video_name = self.video_names[idx]
            video_id = video_name
        elif self.database_type.startswith('8kpro'):
            video_name = self.video_names[idx]
            video_id = video_name[:-4] if video_name.endswith('.mp4') else video_name
        else:
            video_name = self.video_names[idx]
            video_id = video_name

        quality_score = torch.FloatTensor([float(self.scores[idx])])

        frame_path = os.path.join(self.data_root, video_id)
        laplacian_path = os.path.join(self.laplacian_root, video_id)

        if self.database_type.startswith('8kpro'):
            video_length = 5
        elif self.database_type.startswith('KoNViD'):
            video_length = 8
        elif self.database_type.startswith('LiveVQC'):
            video_length = 10
        else:
            video_length = 8

        video_frames = torch.zeros([video_length, 3, self.input_size, self.input_size])
        laplacian_features = torch.zeros([video_length, 5 * 384])

        random.seed(np.random.randint(20240101, 20241231))
        
        for frame_idx in range(video_length):
            frame_file = os.path.join(frame_path, f'{frame_idx:03d}.png')
            if os.path.exists(frame_file):
                frame = Image.open(frame_file).convert('RGB')
                frame = self.transform(frame)
                video_frames[frame_idx] = frame
            
            lp_file = os.path.join(laplacian_path, f'{frame_idx:03d}.npy')
            if os.path.exists(lp_file):
                lp_feature = np.load(lp_file)
                laplacian_features[frame_idx] = torch.from_numpy(lp_feature).view(-1)

        motion_features = self._load_motion_features(video_id, video_length)

        return video_frames, motion_features, quality_score, laplacian_features, video_id

    def _load_motion_features(self, video_id, video_length):
        if self.feature_mode == 'Slow':
            feature_dim = 2048
        elif self.feature_mode == 'Fast':
            feature_dim = 256
        elif self.feature_mode == 'SlowFast':
            feature_dim = 2048 + 256
        else:
            feature_dim = 256

        motion_features = torch.zeros([video_length, feature_dim])
        
        if self.database_type.startswith('8kpro'):
            feature_folder = os.path.join(self.motion_feature_root, video_id + '.mp4')
        else:
            feature_folder = os.path.join(self.motion_feature_root, video_id)

        if os.path.exists(feature_folder):
            for frame_idx in range(video_length):
                if self.feature_mode == 'Slow':
                    feature_file = os.path.join(feature_folder, f'feature_{frame_idx}_slow_feature.npy')
                    if os.path.exists(feature_file):
                        feature = np.load(feature_file)
                        motion_features[frame_idx] = torch.from_numpy(feature).squeeze()
                        
                elif self.feature_mode == 'Fast':
                    feature_file = os.path.join(feature_folder, f'feature_{frame_idx}_fast_feature.npy')
                    if os.path.exists(feature_file):
                        feature = np.load(feature_file)
                        motion_features[frame_idx] = torch.from_numpy(feature).squeeze()
                        
                elif self.feature_mode == 'SlowFast':
                    slow_file = os.path.join(feature_folder, f'feature_{frame_idx}_slow_feature.npy')
                    fast_file = os.path.join(feature_folder, f'feature_{frame_idx}_fast_feature.npy')
                    
                    if os.path.exists(slow_file) and os.path.exists(fast_file):
                        slow_feature = torch.from_numpy(np.load(slow_file)).squeeze()
                        fast_feature = torch.from_numpy(np.load(fast_file)).squeeze()
                        combined_feature = torch.cat([slow_feature, fast_feature])
                        motion_features[frame_idx] = combined_feature

        return motion_features