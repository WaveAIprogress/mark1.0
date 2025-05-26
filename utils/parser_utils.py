# parser_utils.py
import torch
import torchaudio
import soundfile as sf
import os

# 리샘플러 캐시를 위한 전역 변수 (또는 클래스 멤버로 관리 가능)
# 여기서는 간단하게 함수 내에서 처리하거나, AudioParser 클래스에서 관리하도록 함.
# _resampler_cache = {}

# def _get_resampler(orig_sr, new_sr, device="cpu"): # device 인자 추가 가능
#     key = (orig_sr, new_sr)
#     if key not in _resampler_cache:
#         _resampler_cache[key] = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr).to(device)
#     return _resampler_cache[key]

def load_audio_file(file_path, target_sample_rate, resampler_cache=None):
    """
    오디오 파일을 로드하고, 모노 변환 및 리샘플링을 수행하여 [1, T] 형태의 텐서로 반환.
    
    Args:
        file_path (str): 오디오 파일 경로
        target_sample_rate (int): 목표 샘플링 레이트
        resampler_cache (dict, optional): 리샘플러 객체를 캐싱하기 위한 딕셔너리. 
                                          None이면 캐싱하지 않음.

    Returns:
        torch.Tensor or None: 처리된 웨이브폼 [1, T] 또는 오류 시 None
    """
    file_path_norm = os.path.normpath(file_path).replace("\\", "/")
    if not os.path.isfile(file_path_norm):
        print(f"[Util 오류] 파일을 찾을 수 없습니다: {file_path_norm}.")
        return None

    try:
        waveform_np, sr = sf.read(file_path_norm, dtype='float32')
        waveform = torch.from_numpy(waveform_np)
    except Exception as e:
        print(f"[Util 오류] {file_path_norm} 파일 로딩 실패: {e}.")
        return None

    if waveform.ndim == 0 or waveform.numel() == 0:
        print(f"[Util 경고] {file_path_norm} 파일이 비어있습니다.")
        return None

    # --- 모노 변환 ---
    if waveform.ndim > 1:
        # sf.read는 보통 [Samples, Channels] 또는 [Channels, Samples] 반환
        # 채널 수가 적은 차원을 채널 차원으로 가정
        # 예: (48000, 2) -> 2채널, (2, 48000) -> 2채널
        # torchaudio.load는 보통 [Channels, Samples]
        
        # 채널 수가 적은 쪽을 채널로 간주
        # 다만, sf.read가 [Samples, Channels]로 반환하는 경우가 많으므로,
        # shape[1]이 채널 수일 가능성을 먼저 고려 (채널 수가 보통 적으므로)
        if waveform.shape[1] > 1 and waveform.shape[1] <= 8 : # [Samples, Channels], 채널 8개 이하
            waveform = waveform.mean(dim=1) # 채널 평균
        elif waveform.shape[0] > 1 and waveform.shape[0] <= 8: # [Channels, Samples], 채널 8개 이하
            waveform = waveform.mean(dim=0) # 채널 평균
        elif waveform.shape[0] == 1: # [1, Samples]
            waveform = waveform.squeeze(0)
        elif waveform.shape[1] == 1: # [Samples, 1]
            waveform = waveform.squeeze(1)
        else: # 불확실한 다채널의 경우, 일단 첫 번째 채널 사용 시도 또는 전체 평균
            print(f"[Util 경고] {file_path_norm} 다채널 오디오 형태 불확실 ({waveform.shape}). 첫 채널 사용 시도.")
            try:
                # 더 작은 차원을 채널로 가정하고 첫 번째 채널 슬라이싱
                if waveform.shape[0] < waveform.shape[1]:
                    waveform = waveform[0, :]
                else:
                    waveform = waveform[:, 0]
            except IndexError: # 슬라이싱 실패 시 평균
                 waveform = waveform.contiguous().view(-1).mean() # 안전하게 전체 평균
                 if waveform.numel() == 0: return None


    # --- [1, T] 형태로 정규화 ---
    if waveform.dim() != 1: # 위에서 1D로 만들었어야 함
        waveform = waveform.contiguous().view(-1) # 강제로 1D로 만듦

    if waveform.numel() == 0:
        print(f"[Util 오류] {file_path_norm} 모노 변환 후 비어있습니다.")
        return None
    
    waveform = waveform.unsqueeze(0) # [1, T] 형태로 만듦

    # --- 리샘플링 ---
    if sr != target_sample_rate:
        if resampler_cache is not None:
            key = (sr, target_sample_rate)
            if key not in resampler_cache:
                resampler_cache[key] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            resampler = resampler_cache[key]
        else: # 캐시 사용 안 함
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        
        try:
            waveform = resampler(waveform)
        except Exception as e_resample:
            print(f"[Util 오류] {file_path_norm} 리샘플링 실패 (sr={sr} -> {target_sample_rate}): {e_resample}")
            return None
            
    return waveform