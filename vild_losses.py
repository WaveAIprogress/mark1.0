# vild_losses.py
'''
역할: 
  1) ViLD-text: CrossEntropy -> cosine similarity 기반의 softmax
  2) ViLD-image: Teacher vs Student embedding 간 L1 loss
  
변환 방향:
  1) 오디오 embedding 간 거리 정렬로 그대로 사용 가능
  2) background embedding은 별도로 학습하게 둬야 함(논문과 동일)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from vild_config import AudioViLDConfig

class ViLDLosses:
    def __init__(self, config: AudioViLDConfig):
        self.text_loss_weight = config.text_loss_weight
        self.image_loss_weight = config.image_loss_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_text_loss(self, logits, targets):
        """
        - ViLD-text용 cross-entropy loss
        - background class가 포함된 [B, C+1] 로짓과 레이블 비교
        """
        return self.text_loss_weight * self.ce_loss(logits, targets)

    def compute_image_loss(self, student_proj, teacher_embeddings):
        """
        - ViLD-image용 L1 loss
        - Student region embedding(projected) <-> Teacher embedding 비교
        """
        return self.image_loss_weight * F.l1_loss(student_proj, teacher_embeddings)

    def total_loss(self, logits, targets, student_proj, teacher_embeddings):
        """
        - 위 두 손실의 가중합(weight는 vild_config.py에서 설정함)
        """
        text_loss = self.compute_text_loss(logits, targets)
        image_loss = self.compute_image_loss(student_proj, teacher_embeddings)
        return text_loss + image_loss, text_loss, image_loss
