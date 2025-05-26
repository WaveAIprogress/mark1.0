# vild_parser.py
import torch
import torchaudio
import torchaudio.transforms as T
import os
import sys

# utils 폴더를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from parser_utils import load_audio_file # 새로 만든 유틸리티 함수 임포트

class AudioParser:
    def __init__(self, config):
        self.config = config
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.fft_size,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        self.resampler_cache = {} # 리샘플러 캐시를 AudioParser 인스턴스가 관리

        try:
            torchaudio.set_audio_backend("soundfile")
        except RuntimeError:
            pass

    def load_and_segment(self, file_path):
        # parser_utils.py의 함수를 사용하여 오디오 로드, 모노 변환, 리샘플링
        waveform = load_audio_file(file_path, self.config.sample_rate, self.resampler_cache)

        if waveform is None: # load_audio_file에서 오류 발생 시 None 반환
            print(f"[Parser] {file_path} 처리 중 load_audio_file에서 None 반환. 빈 리스트 반환.")
            return []
        
        # waveform은 이제 [1, T] 형태의 텐서여야 함
        if not (waveform.ndim == 2 and waveform.shape[0] == 1 and waveform.numel() > 0) :
            print(f"[Parser 오류] load_audio_file 반환값 형태 이상: {waveform.shape if isinstance(waveform, torch.Tensor) else type(waveform)}. 빈 리스트 반환.")
            return []


        total_samples = waveform.size(1)
        segment_samples = self.config.segment_samples
        segments_mel_tensors = []

        for start in range(0, total_samples, segment_samples):
            end = start + segment_samples
            chunk = waveform[:, start:end] # chunk is [1, segment_length]

            current_chunk_len = chunk.size(1)

            # 매우 짧은 마지막 청크에 대한 처리 (옵션)
            # 예: 최소 길이 (0.1초) 미만이면 무시, 단 첫 세그먼트이거나 전체가 짧으면 패딩해서 하나는 만듦
            min_meaningful_len = int(self.config.segment_duration * 0.1 * self.config.sample_rate)
            if current_chunk_len < min_meaningful_len:
                if len(segments_mel_tensors) > 0: # 이미 다른 세그먼트가 있다면 이 짧은 청크는 무시
                    continue
                elif start == 0 and total_samples < segment_samples : # 전체 오디오가 짧고 이것이 유일한 청크인 경우
                    pass # 아래에서 패딩 처리
                elif current_chunk_len == 0 : # 길이가 0인 청크 (이미 세그먼트가 없거나, 첫 청크가 아니면)
                    if start > 0 or len(segments_mel_tensors) > 0:
                        continue

            if current_chunk_len < segment_samples:
                if current_chunk_len == 0: # 첫 청크이고 길이가 0인 경우 (오디오가 극도로 짧거나 문제)
                    if start == 0 and not segments_mel_tensors:
                        print(f"[Parser 경고] {os.path.basename(file_path)} 첫 청크 길이가 0. 패딩된 빈 세그먼트 생성 시도.")
                        # 패딩만으로 이루어진 청크 생성
                        chunk = torch.zeros(1, segment_samples, device=waveform.device)
                    else: # 이전 세그먼트가 있거나 첫 청크가 아니면 건너뜀
                        continue
                else: # 일반적인 패딩
                    padding_size = segment_samples - current_chunk_len
                    pad = torch.zeros(1, padding_size, device=chunk.device)
                    chunk = torch.cat([chunk, pad], dim=1)
            
            try:
                mel = self.mel_transform(chunk)  # [1, n_mels, time_frames]
                processed_mel_segment = mel.unsqueeze(1) # [1, 1, n_mels, time_frames]
                segments_mel_tensors.append(processed_mel_segment)
            except Exception as e_mel:
                print(f"[Parser 오류] {os.path.basename(file_path)} Mel 변환 중 오류 (chunk shape: {chunk.shape}): {e_mel}")
                continue # 이 세그먼트 건너뛰기

        if not segments_mel_tensors and total_samples > 0: # 오디오는 있었는데 세그먼트가 안 만들어진 경우 (예: 너무 짧아서 다 무시됨)
             print(f"[Parser 경고] {os.path.basename(file_path)}에서 세그먼트가 생성되지 않았습니다 (total_samples: {total_samples}).")
             # 이 경우, 강제로 하나의 패딩된 세그먼트를 만들 수도 있음 (선택 사항)
             # chunk = torch.zeros(1, segment_samples, device=waveform.device)
             # mel = self.mel_transform(chunk)
             # processed_mel_segment = mel.unsqueeze(1)
             # segments_mel_tensors.append(processed_mel_segment)


        # [디버깅 프린트] (필요시 활성화)
        # print(f"Debug (load_and_segment for {os.path.basename(file_path)}):")
        # print(f"  Returning type: {type(segments_mel_tensors)}")
        # if isinstance(segments_mel_tensors, list):
        #     print(f"  List length: {len(segments_mel_tensors)}")
        #     if segments_mel_tensors:
        #         print(f"  First element type: {type(segments_mel_tensors[0])}")
        #         if isinstance(segments_mel_tensors[0], torch.Tensor):
        #             print(f"  First element shape: {segments_mel_tensors[0].shape}")
        # elif isinstance(segments_mel_tensors, torch.Tensor):
        #     print(f"  !!! UNEXPECTED RETURN TYPE: Tensor shape: {segments_mel_tensors.shape}")
        
        
        return segments_mel_tensors

    def parse_sample(self, file_path, label_text):
        segments_mel_tensors_list = self.load_and_segment(file_path)
        
        if not segments_mel_tensors_list:
            raise ValueError(f"No segments parsed from {file_path} (load_and_segment returned empty list).")

        try:
            mel_tensor_stacked = torch.cat(segments_mel_tensors_list, dim=0)
        except Exception as e:
            print(f"Error during torch.cat in parse_sample for {file_path}: {e}")
            for i, tensor in enumerate(segments_mel_tensors_list):
                print(f"  Shape of tensor {i} in list: {tensor.shape if isinstance(tensor, torch.Tensor) else type(tensor)}")
            raise
        
        label_index = self.config.get_class_index(label_text)
        return mel_tensor_stacked, label_index