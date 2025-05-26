# vild_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from vild_config import AudioViLDConfig # 타입 힌팅용

class SimpleAudioEncoder(nn.Module):
    def __init__(self, config): # config: AudioViLDConfig
        super().__init__()
        # 입력: [B, 1, n_mels, time_frames] (예: [B, 1, 64, 101])
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # [B, 32, 64, 101]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 32, 32, 50] (padding에 따라 달라질 수 있음)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [B, 64, 32, 50]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 64, 16, 25]
            
            # AdaptiveAvgPool2d는 출력 크기를 (1,1)로 고정
            nn.AdaptiveAvgPool2d((1, 1)), # [B, 64, 1, 1]
            nn.Flatten(), # [B, 64]
            nn.Linear(64, config.embedding_dim) # [B, embedding_dim]
        )

    def forward(self, x):
        return self.model(x)


class ViLDTextHead(nn.Module):
    def __init__(self, config): # config: AudioViLDConfig
        super().__init__()
        self.temperature = 0.07 # CLIP 기본값
        # 배경 임베딩은 학습 가능한 파라미터
        if config.use_background_embedding:
            self.background_embedding = nn.Parameter(torch.randn(config.embedding_dim))
        else:
            self.background_embedding = None


    def forward(self, region_embeddings, class_text_embeddings_non_bg):
        # region_embeddings: [B, D] (오디오 세그먼트 임베딩)
        # class_text_embeddings_non_bg: [C_non_bg, D] (배경 제외 실제 클래스 텍스트 임베딩)
        
        region_norm = F.normalize(region_embeddings, dim=1)
        text_norm = F.normalize(class_text_embeddings_non_bg, dim=1)
        
        # 실제 클래스와의 유사도
        sim_actual_classes = torch.matmul(region_norm, text_norm.T)  # [B, C_non_bg]

        if self.background_embedding is not None:
            bg_embed_norm = F.normalize(self.background_embedding.unsqueeze(0), dim=1)
            sim_background = torch.matmul(region_norm, bg_embed_norm.T)  # [B, 1]
            # 로짓: [배경 유사도, 실제 클래스1 유사도, 실제 클래스2 유사도, ...]
            logits = torch.cat([sim_background, sim_actual_classes], dim=1) / self.temperature # [B, 1 + C_non_bg]
        else: # 배경 임베딩 사용 안할 시 (이 경우 loss 계산 시 target 조정 필요)
            logits = sim_actual_classes / self.temperature # [B, C_non_bg]
            
        return logits

# ViLDImageHead는 현재 사용되지 않고, student_train에서 ViLDHead (projection)를 직접 사용.
# 필요하다면 아래와 같이 정의할 수 있으나, ViLDHead와 기능이 거의 동일 (L1 loss는 ViLDLosses에서 계산)
# class ViLDImageHead(nn.Module):
#     def __init__(self, config): # config: AudioViLDConfig
#         super().__init__()
#         self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
# 
#     def forward(self, student_region_embeddings): # teacher_embeddings는 loss 계산 시 사용
#         student_proj = self.projection(student_region_embeddings)
#         return student_proj # L1 loss는 외부에서 계산