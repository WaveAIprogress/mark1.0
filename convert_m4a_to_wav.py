# convert_m4a_to_wav.py
import os
import sys

# 현재 파일 기준으로 utils 경로 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from convert_utils import process_audio_files
from seed_utils import set_seed


def convert_all_audio_files(input_dir="C:/Users/user/Desktop/AI_model/mark1/data",
                            output_dir="C:/Users/user/Desktop/AI_model/mark1/data_wav"):
    # process_audio_files 함수가 내부적으로 필요한 모든 작업을 수행.
    process_audio_files(input_dir, output_dir)

if __name__ == "__main__":
    set_seed(42) # 스크립트 실행 시 시드 설정
    convert_all_audio_files()