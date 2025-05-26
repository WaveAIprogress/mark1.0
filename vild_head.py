# vild_head.py
'''
역할: 이미지 RoI(Region of Interest, 관심영역)에서 feature 추출
변환 방향:
  1) 오디오 segment에 대한 projection layer로 재정의
  2) Projection -> L2 Norm -> region embedding 구조는 동일하게 유지
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViLDHead(nn.Module):
    """
    - Student 모델의 region embedding을 임베딩 공간에 투영(projection) + 정규화하는 공통 헤드
    - 공통 region head: projection → L2 normalization
    - Student 모델이 생성한 segment embedding을 최종 임베딩 공간으로 변환
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        - 입력: [B, input_dim] 형태의 raw audio embedding
        - 처리: Linear -> L2 Normalize
        - 출력: [B, output_dim] 크기의 normalized region embedding
          -> 이 임베딩은 ViLD-text 또한 ViLD-image에 사용됨.
        """
        x = self.projection(x)
        return F.normalize(x, dim=1)
