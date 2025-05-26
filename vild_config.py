# vild_config.py

'''
역할: 학습 하이퍼파리미터, 클래스 정의, embedding 차원 설정
변환 방향:
  1) class 수: apartment_noise, daily_noise, background
  2) input type을 이미지가 아니라 오디오 segment(Mel-spectrogram)로 바꿈.
  3) embedding dim은 audio encoder의 출력에 맞춰 수정함.
'''

class AudioViLDConfig:
    def __init__(self):
        # Class configuration
        self.classes = ["apartment_noise", "daily_noise", "background_noise"]
        self.num_classes = len(self.classes)

        # Input/audio parameters
        self.sample_rate = 16000  # Hz
        self.segment_duration = 1.0  # seconds
        self.segment_samples = int(self.sample_rate * self.segment_duration)
        self.fft_size = 512
        self.hop_length = 160
        self.n_mels = 64

        # Model config
        self.embedding_dim = 384 # all-MiniLM-L6-v2 모델 출력 차원이 384라서 맞춤.
        self.use_background_embedding = True

        # Training
        self.batch_size = 16
        self.num_epochs = 100
        self.learning_rate = 1e-4

        # Loss weights
        self.text_loss_weight = 1.0
        self.image_loss_weight = 1.0

        # Others
        self.device = "cuda"  # or "cpu"

    def get_class_index(self, class_name):
        return self.classes.index(class_name)
