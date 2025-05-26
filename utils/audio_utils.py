# audio_utils.py
# loss 땜에 만든거

import torch

def prepare_teacher_embedding(embedding, device):
    embedding = embedding.to(device)

    if embedding.dim() == 4:
        embedding = embedding.squeeze(0).squeeze(0).squeeze(0)  # (1,1,1,384) -> (384,)
    elif embedding.dim() == 3:
        embedding = embedding.squeeze(0).squeeze(0)             # (1,1,384) -> (384,)
    elif embedding.dim() == 2:
        embedding = embedding.squeeze(0)                        # (1,384) -> (384,)

    # 최종적으로 (1, 384)로 복원
    embedding = embedding.unsqueeze(0)
    return embedding
