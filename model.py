import torch
import torch.nn as nn
import torchvision.models as models
import timm


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DisplaySwinTransformer(nn.Module):
    
    def __init__(self, pretrained=True, global_pooling=''):
        super(DisplaySwinTransformer, self).__init__()
        
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224_in22k', 
            pretrained=pretrained, 
            global_pool=global_pooling
        )
        
        self.backbone.head = Identity()
        self.feature_dim = 1024

    def forward(self, x):
        features = self.backbone(x)
        return features


class ColorAttentionModule(nn.Module):
    
    def __init__(self, input_channels):
        super(ColorAttentionModule, self).__init__()
        
        self.attention_pooling = nn.AdaptiveAvgPool2d((12, 12))
        self.attention_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(input_channels, 3000),
            nn.Dropout(p=0.75),
            nn.Linear(3000, 128),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        query = x.view(batch_size, channels, -1)
        key = query
        query = query.permute(0, 2, 1)
        
        similarity_map = torch.matmul(query, key)
        
        query_norm = torch.norm(query, dim=2, keepdim=True)
        key_norm = torch.norm(key, dim=1, keepdim=True)
        similarity_map = torch.div(
            similarity_map, 
            torch.matmul(query_norm, key_norm).clamp(min=1e-8)
        )
        
        x_pooled = self.attention_pooling(x)
        x_flattened = x_pooled.view(x_pooled.size(0), -1)
        
        attention_weights = self.attention_head(x_flattened)
        attention_weights = attention_weights.view(-1, 128, 1, 1)
        
        return attention_weights


class SpatialQualityModule(nn.Module):
    
    def __init__(self, laplacian_feature_dim, dropout_rate=0.2):
        super(SpatialQualityModule, self).__init__()
        
        self.rectifier = nn.Sequential(
            nn.Linear(laplacian_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_rate),
        )
        
    def forward(self, laplacian_features):
        batch_size = laplacian_features.size(0)
        lp_flattened = laplacian_features.view(batch_size, -1)
        spatial_params = self.rectifier(lp_flattened)
        return spatial_params


class TemporalQualityModule(nn.Module):
    
    def __init__(self, motion_feature_dim, dropout_rate=0.2):
        super(TemporalQualityModule, self).__init__()
        
        self.rectifier = nn.Sequential(
            nn.Linear(motion_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_rate),
        )
        
    def forward(self, motion_features):
        batch_size = motion_features.size(0)
        motion_flattened = motion_features.view(batch_size, -1)
        temporal_params = self.rectifier(motion_flattened)
        return temporal_params


class BaseQualityRegressor(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(BaseQualityRegressor, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return self.regressor(x)


class DisplayVQAModel(nn.Module):
    
    def __init__(self, sequence_length=8, enable_spatial=True, enable_temporal=True,
                 spatial_dropout=0.2, temporal_dropout=0.2):
        super(DisplayVQAModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.enable_spatial = enable_spatial
        self.enable_temporal = enable_temporal
        
        self.spatial_extractor = DisplaySwinTransformer(pretrained=True)
        
        self.base_quality_regressor = BaseQualityRegressor(
            input_dim=self.spatial_extractor.feature_dim,
            hidden_dim=128,
            output_dim=1
        )
        
        if self.enable_spatial:
            laplacian_dim = 5 * 384 * self.sequence_length
            self.spatial_module = SpatialQualityModule(
                laplacian_feature_dim=laplacian_dim,
                dropout_rate=spatial_dropout
            )
        
        if self.enable_temporal:
            motion_dim = 256 * self.sequence_length
            self.temporal_module = TemporalQualityModule(
                motion_feature_dim=motion_dim,
                dropout_rate=temporal_dropout
            )
            
        self.color_attention = ColorAttentionModule(input_channels=20736)
        
    def forward(self, video_frames, motion_features, laplacian_features):
        
        batch_size, seq_len, channels, height, width = video_frames.shape
        
        frames_reshaped = video_frames.view(-1, channels, height, width)
        
        spatial_features = self.spatial_extractor(frames_reshaped)
        
        base_predictions = self.base_quality_regressor(spatial_features)
        
        base_predictions = base_predictions.view(batch_size, seq_len, -1)
        base_quality = torch.mean(base_predictions, dim=1)
        
        spatial_quality = base_quality.clone()
        temporal_quality = base_quality.clone()
        combined_quality = base_quality.clone()
        
        if self.enable_spatial:
            spatial_params = self.spatial_module(laplacian_features)
            
            spatial_scale = torch.chunk(spatial_params, 2, dim=1)[0]
            spatial_bias = torch.chunk(spatial_params, 2, dim=1)[1]
            
            ones_like_base = torch.ones_like(base_quality)
            spatial_scale = torch.add(spatial_scale, ones_like_base)
            
            spatial_quality = torch.add(
                torch.mul(torch.abs(spatial_scale), base_quality), 
                spatial_bias
            )
        
        if self.enable_temporal:
            device = video_frames.device
            motion_features = motion_features.to(device)
            
            temporal_params = self.temporal_module(motion_features)
            
            temporal_scale = torch.chunk(temporal_params, 2, dim=1)[0]
            temporal_bias = torch.chunk(temporal_params, 2, dim=1)[1]
            
            ones_like_base = torch.ones_like(base_quality)
            temporal_scale = torch.add(temporal_scale, ones_like_base)
            
            temporal_quality = torch.add(
                torch.mul(torch.abs(temporal_scale), base_quality), 
                temporal_bias
            )
        
        if self.enable_spatial and self.enable_temporal:
            combined_scale = torch.sqrt(torch.abs(torch.mul(
                torch.chunk(self.spatial_module(laplacian_features), 2, dim=1)[0] + 
                torch.ones_like(base_quality),
                torch.chunk(self.temporal_module(motion_features), 2, dim=1)[0] + 
                torch.ones_like(base_quality)
            )))
            
            combined_bias = torch.div(torch.add(
                torch.chunk(self.spatial_module(laplacian_features), 2, dim=1)[1],
                torch.chunk(self.temporal_module(motion_features), 2, dim=1)[1]
            ), 2)
            
            combined_quality = torch.add(
                torch.mul(combined_scale, base_quality), 
                combined_bias
            )
        elif self.enable_spatial:
            combined_quality = spatial_quality
        elif self.enable_temporal:
            combined_quality = temporal_quality
        
        return (
            base_quality.squeeze(1),
            spatial_quality.squeeze(1),
            temporal_quality.squeeze(1),
            combined_quality.squeeze(1)
        )
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': 'DisplayVQA Model',
            'backbone': 'Swin Transformer Base',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'sequence_length': self.sequence_length,
            'spatial_enhancement': self.enable_spatial,
            'temporal_enhancement': self.enable_temporal,
        }
        
        return info


def create_display_vqa_model(config):
    
    model = DisplayVQAModel(
        sequence_length=getattr(config, 'sequence_length', 8),
        enable_spatial=getattr(config, 'enable_spatial', True),
        enable_temporal=getattr(config, 'enable_temporal', True),
        spatial_dropout=getattr(config, 'spatial_dropout', 0.2),
        temporal_dropout=getattr(config, 'temporal_dropout', 0.2)
    )
    
    return model