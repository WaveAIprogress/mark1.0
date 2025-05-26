# generate_dataset_index.py
import os
import sys
import glob
import csv

# utils 폴더를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from seed_utils import set_seed # 추가

def generate_index(data_dir="C:/Users/user/Desktop/AI_model/mark1/data_wav", output_csv="dataset_index.csv", seed_value=42): # seed_value 인자 추가
    set_seed(seed_value) # glob 순서에는 영향 없지만, 일관성을 위해 추가
    
    label_map = {
        "apartment": "apartment_noise",
        "daily": "daily_noise",
        "background": "background_noise"
    }

    audio_paths = glob.glob(os.path.join(data_dir, "*.wav"))
    # glob 결과는 시스템에 따라 다를 수 있으므로, 재현성을 위해 정렬
    audio_paths.sort() 
    
    entries = []

    for path in audio_paths:
        basename = os.path.basename(path).lower()
        matched_label = None
        for key, label in label_map.items():
            if key in basename:
                normalized_path = os.path.normpath(path).replace("\\", "/")
                entries.append((normalized_path, label))
                matched_label = label
                break
        if matched_label is None:
            print(f"Warning: No matching label found for {basename} in label_map. Skipping this file.")


    if not entries:
        print(f"매칭되는 오디오 파일이 '{data_dir}'에 없거나, 파일명에 키워드가 없습니다.")
        return

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(entries)

    print(f"{len(entries)}개의 오디오 샘플 인덱스를 저장했습니다. ->  {output_csv}")

if __name__ == "__main__":
    # set_seed(42) # generate_index 함수 내부에서 설정하거나 여기서 한 번만.
    generate_index(seed_value=42)