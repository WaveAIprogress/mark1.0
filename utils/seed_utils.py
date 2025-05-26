# seed_utils.py
import random
import numpy as np
import torch
import os

def set_seed(seed_value=42):
    """Reproducibility를 위해 random seed를 설정하는 함수."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) # 모든 GPU에 대해 seed 설정
    
    # 더 엄격한 재현성을 위해 (성능 저하 가능성 있음)
    # 아래 두 줄은 필요에 따라 활성화/비활성화 할 예정.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # os.environ['PYTHONHASHSEED'] = str(seed_value) # 파이썬 해시 시드 고정 (필요시)
    print(f"Global seed set to {seed_value}")