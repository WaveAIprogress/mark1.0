# plot_audio.py
import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import sys
import numpy as np # 밑에 주석 해제 한 것 때문에 추가
import soundfile as sf # torchaudio.load 대신 sf.read 사용 (vild_parser와 일관성)

# utils 폴더를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from seed_utils import set_seed

# torchaudio 백엔드 설정은 main 또는 사용하는 곳에서 한 번만.
# torchaudio.set_audio_backend("sox_io") # 또는 "soundfile"

def plot_waveform_and_mel(path, save_dir="C:/Users/user/Desktop/AI_model/mark1/plots", seed_value=42): # seed_value 인자 추가
    set_seed(seed_value) # Matplotlib 내부의 미세한 랜덤 요소 제어 (거의 영향 없음)
    
    # torchaudio 백엔드 설정 (plot_audio.py가 단독 실행될 경우를 대비)
    try:
        torchaudio.set_audio_backend("soundfile") # 또는 "sox_io"
    except RuntimeError as e:
        print(f"Torchaudio backend error: {e}. Ensure soundfile or sox is installed.")
        # 기본 백엔드로 계속 진행 시도

    try:
        # waveform, sr = torchaudio.load(path) # 이전 방식
        waveform_np, sr = sf.read(path, dtype='float32')
        waveform = torch.from_numpy(waveform_np)
        if waveform.ndim > 1: # 스테레오 등 다채널 오디오의 경우 첫 번째 채널만 사용 또는 평균
            waveform = waveform[:, 0] if waveform.shape[1] > 0 else waveform.mean(dim=1)
        waveform = waveform.unsqueeze(0) # [1, T] 형태로
    except Exception as e:
        print(f"Error loading audio file {path}: {e}")
        return
        
    filename = os.path.splitext(os.path.basename(path))[0]

    # [그래프1: Waveform plot]
    plt.figure(figsize=(10, 3))
    plt.plot(waveform.t().numpy()) # .t()는 2D 텐서에 사용, 현재 waveform은 [1, T]이므로 waveform[0] 사용
    plt.title("Waveform: " + filename) # 파형
    plt.xlabel("Time") # 시간
    plt.ylabel("Amplitude") # 진폭
    os.makedirs(save_dir, exist_ok=True)
    wave_path = os.path.join(save_dir, f"{filename}_waveform.png")
    plt.savefig(wave_path)
    plt.close()
    print(f"Waveform plot saved: {wave_path}")

    # [그래프2: Mel-spectrogram]
    try:
        # vild_config에서 파라미터 가져오기 (일관성 유지)
        from vild_config import AudioViLDConfig
        config = AudioViLDConfig()

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, # 로드된 파일의 sr 사용 또는 config.sample_rate로 리샘플링 후 사용
            n_fft=config.fft_size,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        # 원본 샘플링레이트와 config.sample_rate가 다를 경우 리샘플링
        if sr != config.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config.sample_rate)
            waveform_resampled = resampler(waveform)
        else:
            waveform_resampled = waveform

        mel = mel_transform(waveform_resampled) # [1, n_mels, time]
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        plt.figure(figsize=(10, 4))
        # mel_db[0] -> [n_mels, time]
        plt.imshow(mel_db.squeeze(0).numpy(), origin='lower', aspect='auto', cmap='viridis')
        plt.title("Mel Spectrogram: " + filename)
        plt.xlabel("Time")
        plt.ylabel("Mel bins")
        mel_path = os.path.join(save_dir, f"{filename}_mel.png")
        plt.savefig(mel_path)
        plt.close()
        print(f"Mel spectrogram plot saved: {mel_path}")
    except Exception as e:
        print(f"Error generating or saving Mel spectrogram for {path}: {e}")


    print(f"시각화 저장을 완료했거나 시도했습니다. -> {wave_path}, {mel_path if 'mel_path' in locals() else 'Mel plot failed'}")


if __name__ == "__main__":
    # set_seed(42) # plot_waveform_and_mel 함수 내부에서 설정하거나 여기서 한 번만.
    # 테스트할 오디오 파일 경로를 지정하세요.
    test_audio_file = "C:/Users/user/Desktop/AI_model/mark1/data_wav/apartment_adultFootstep8.wav" # 이 파일이 실제 있어야 함
    if not os.path.exists(test_audio_file):
        print(f"테스트 파일 없음: {test_audio_file}. 임시 파일을 생성하거나 실제 파일 경로로 변경하세요.")
        # 임시 파일 생성 (테스트용) 
        '''지금은 주석 해제'''
        os.makedirs("./data_wav", exist_ok=True)
        sample_rate = 16000; duration = 1
        data = np.random.uniform(-0.5, 0.5, sample_rate * duration).astype(np.float32)
        sf.write(test_audio_file, data, sample_rate)
        print(f"임시 테스트 파일 생성: {test_audio_file}")
    else:
        plot_waveform_and_mel(test_audio_file, seed_value=42)