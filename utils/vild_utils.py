# vild_utils.py 
# 오류때문에 파일 새로 팠음 ㅠㅠ
# 티쳐야~ 스튜던트야~ 제발 내 마음을 알아죠잉 ㅠㅠㅠ 흑흑..

import torch

def normalize_mel_shape(mel: torch.Tensor) -> torch.Tensor:
    """
    mel 입력 텐서를 [1, 1, 64, 101] 형식으로 통일함.
    
    형식 설명:
        - 첫 번째 차원: 배치 크기 (batch size)
        - 두 번째 차원: 채널 수 (channel, 보통 1로 고정)
        - 세 번째 차원: 멜 주파수 축 (mel frequency bins, 보통 64)
        - 네 번째 차원: 시간 축 (time frames, 보통 101)

    이 함수는 다양한 형태로 들어올 수 있는 mel spectrogram 텐서를
    모델에서 기대하는 입력 형태로 변환해줌.
    """
    if mel.dim() == 2:
        # [64, 101] → [1, 1, 64, 101]로 변환 (채널/배치 차원 추가)
        return mel.unsqueeze(0).unsqueeze(0)  
    elif mel.dim() == 3:
        if mel.shape[0] == 1:
            # [1, 64, 101] → [1, 1, 64, 101]로 변환 (배치 차원 추가)
            return mel.unsqueeze(0)           
        else:
            # 예외 처리용: [64, 101, X] 등 예기치 않은 경우에도 일단 4차원으로 맞춤
            return mel.unsqueeze(0).unsqueeze(0)  # [64, 101, X] -> 오류 회피
    elif mel.dim() == 4:
        # 이미 [1, 1, 64, 101] 형식일 경우 그대로 반환
        return mel  # 이미 정상
    elif mel.dim() == 5 and mel.shape[0] == 1:
        # 예외 처리: [1, 1, 1, 64, 101] -> [1, 1, 64, 101]
        return mel.squeeze(0)  
    else:
        raise ValueError(f"Unexpected mel shape: {mel.shape}")
