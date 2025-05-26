# eval.py
# soft label 저장, 시각화, confusion matrix 등 평가 도구

import torch
from vild_config import AudioViLDConfig # AudioViLDConfig 경로 확인 필요 (vild_config.py 파일이 현재 디렉토리나 PYTHONPATH에 있어야 함)
from vild_model import SimpleAudioEncoder, ViLDTextHead # vild_model.py 경로 확인 필요
from vild_parser import AudioParser # vild_parser.py 경로 확인 필요
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 현재 파일 기준으로 utils 경로 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from seed_utils import set_seed 

def evaluate(audio_label_list, seed_value=42): # seed_value 인자 추가
    set_seed(seed_value) # 함수 시작 시 시드 설정
    config = AudioViLDConfig()
    parser = AudioParser(config)
    device = config.device if torch.cuda.is_available() else "cpu" # device 설정 추가

    # 모델 로딩
    encoder = SimpleAudioEncoder(config)
    classifier = ViLDTextHead(config)
    # 모델 경로주의~!
    encoder_path = "best_teacher_encoder.pth" # EarlyStopping으로 저장된 best 모델 사용 권장
    classifier_path = "best_teacher_classifier.pth" # EarlyStopping으로 저장된 best 모델 사용 권장
    
    # 만약 best 모델이 없다면 final 모델 사용 (또는 config에서 경로 지정)
    if not os.path.exists(encoder_path):
        encoder_path = "teacher_encoder.pth"
    if not os.path.exists(classifier_path):
        classifier_path = "teacher_classifier.pth"
        
    print(f"Loading encoder from: {encoder_path}")
    print(f"Loading classifier from: {classifier_path}")
    
    encoder.load_state_dict(torch.load(encoder_path, map_location=device)) # device 사용
    classifier.load_state_dict(torch.load(classifier_path, map_location=device)) # device 사용
    encoder.to(device) # 모델을 device로 이동
    classifier.to(device) # 모델을 device로 이동
    encoder.eval()
    classifier.eval()


    # --- 텍스트 임베딩 로딩 수정 ---
    # teacher_train.py와 동일한 방식으로 텍스트 임베딩 생성
    # config.classes에서 "background" 관련 클래스를 제외한 프롬프트 생성
    from sentence_transformers import SentenceTransformer
    text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device) # device 사용

    # non_bg_classes는 teacher_train.py에서의 non_bg_classes 생성 로직과 일치해야 함
    # ViLDTextHead는 내부적으로 background 로짓을 추가하므로, class_text_embeddings는 non-background 클래스에 대해서만.
    non_bg_classes_texts = []
    # teacher_train.py의 target 설정 로직을 보면, ViLDTextHead는 config.classes에 있는 non-background 클래스 순서대로 로짓을 생성.
    # class_text_embeddings는 이 non-background 클래스들의 임베딩이어야 함.
    
    # teacher_train.py 에서 class_text_embeddings 생성 시 사용한 prompt_texts와 동일해야 합니다.
    # prompt_texts = [f"a sound of {cls.replace('_noise', '').replace('_', ' ')}" for cls in config.classes] (teacher_train.py의 원본 코드 중 하나)
    # 위 코드는 config.classes 전체를 사용하지만, ViLDTextHead의 forward에서는 class_text_embeddings의 shape을 보고 bg를 추가함.
    # teacher_train.py의 ViLDTextHead에 전달되는 class_text_embeddings는 non-background 클래스에 대한 것임
    
    # 가장 확실한 방법은 teacher_train.py의 class_text_embeddings 생성 로직을 가져오는 것.
    # teacher_train.py에서는 다음과 같이 생성했었음:
    # prompt_texts = [
    #    f"a sound of {cls.replace('_noise', '').replace('_', ' ')}" for cls in config.classes
    # ]
    # class_text_embeddings = torch.tensor(text_model.encode(prompt_texts), dtype=torch.float).to(device)
    # 이 class_text_embeddings가 teacher_classifier에 전달됨.
    # 즉, ViLDTextHead는 config.classes 전체에 대한 임베딩을 받고,
    # 내부적으로 이 임베딩과 별개로 background 임베딩을 concat하는 것임.
    # 이 가정이 맞다면, eval.py의 원래 코드가 맞지만, 오류가 발생했으므로 다시 확인.

    # ViLDTextHead의 __init__을 보면 background_embedding을 가지고 있고,
    # forward(self, image_features, class_text_embeddings)에서
    # logits = torch.cat([background_logits, text_logits], dim=1) 와 같이 동작.
    # text_logits = image_features @ class_text_embeddings.T * self.logit_scale.exp()
    # 이 때 class_text_embeddings는 non-background 클래스에 대한 임베딩이어야함함.
    # teacher_train.py에서 class_text_embeddings를 만들 때 non-background만 사용했는지 확인 필요.

    # teacher_train.py를 다시 보니, 다음과 같이 class_text_embeddings가 생성됨:
    # prompt_texts = [
    #     f"a sound of {cls.replace('_noise', '').replace('_', ' ')}" for cls in config.classes
    # ]
    # class_text_embeddings = torch.tensor(text_model.encode(prompt_texts), dtype=torch.float).to(device)
    # 이 전체 임베딩이 ViLDTextHead로 전달됨.
    # 그렇다면 ViLDTextHead가 config.classes 전체에 대한 임베딩을 받고, 여기에 추가로 background를 더하는 것이 아니라,
    # config.classes 내에 background_noise가 있으면 그것을 background로 인식하거나,
    # 또는 첫번째 임베딩을 background로 취급하는 등의 로직이 있어야 함.
    # teacher_train.py의 target 설정 부분을 보면:
    # non_bg_classes = [c for c in config.classes if "background" not in c]
    # if "background" in current_label_text: adjusted_target_idx = 0
    # else: adjusted_target_idx = non_bg_classes.index(current_label_text) + 1
    # 이 target 설정은 ViLDTextHead의 출력이 [bg_logit, non_bg_class1_logit, non_bg_class2_logit, ...] 형태임을 가정.
    # 즉, ViLDTextHead에 전달하는 class_text_embeddings는 non-background 클래스에 대한 것이어야 함..

    # 따라서 eval.py에서도 non-background 클래스에 대한 텍스트 임베딩만 생성.
    non_bg_classes_for_prompts = [cls for cls in config.classes if "background" not in cls.lower()] # "background_noise" 등을 제외
    
    if not non_bg_classes_for_prompts:
        # 모든 클래스가 background이거나 non-background 클래스가 없는 극단적 경우 대비
        print("Warning: No non-background classes found for text embeddings in eval. This might be an issue.")
        # 이 경우 classifier가 어떻게 동작할지 불분명. 일단 빈 텐서 또는 오류 처리.
        # 여기서는 학습과 동일하게 config.classes 전체를 사용하도록 일단 두되,
        # ViLDTextHead의 동작을 명확히 이해하는 것이 중요.
        # 만약 ViLDTextHead가 class_text_embeddings를 받고 여기에 bg를 추가하는 구조라면
        # class_text_embeddings는 non_bg 여야 함.
        # 현재 오류 (3 vs 4)는 ViLDTextHead가 class_text_embeddings의 길이에 +1을 하기 때문으로 보임.

    # teacher_train.py의 로직을 따라 non-background 클래스에 대한 프롬프트만 생성
    prompt_texts_for_eval = [
        f"a sound of {cls.replace('_noise', '').replace('_', ' ')}" for cls in non_bg_classes_for_prompts
    ]
    if not prompt_texts_for_eval: # 모든 클래스가 백그라운드일 경우 (거의 없음)
        # 이 경우 class_text_embeddings는 비어있게 되며, classifier에 전달 시 오류 발생 가능.
        # 하지만 ViLDTextHead는 background_embedding을 항상 가지고 있으므로,
        # class_text_embeddings가 비어있으면 background 로짓만 출력할 수 있음 (출력 차원 1).
        # 이 경우 score_sum의 차원과 맞아야 함.
        class_text_embeddings = torch.empty((0, config.embedding_dim), dtype=torch.float).to(device)
        print("Warning: No non-background classes found. Text embeddings will be empty. Classifier might only predict background.")
    else:
        class_text_embeddings = torch.tensor(
            text_model.encode(prompt_texts_for_eval),
            dtype=torch.float
        ).to(device)
    # --- 텍스트 임베딩 로딩 수정 완료 ---

    y_true = []
    y_pred = []

    # score_sum 초기화는 classifier의 실제 출력 차원에 맞춰야 함.
    # classifier 출력: [bg_logit, non_bg_class1_logit, ..., non_bg_classM_logit]
    # 따라서 차원은 1 (bg) + len(non_bg_classes_for_prompts)
    expected_num_classifier_outputs = 1 + len(non_bg_classes_for_prompts)
    
    # 하지만 config.num_classes는 전체 클래스 수 (background 포함).
    # y_true, y_pred는 config.classes의 인덱스를 사용하므로, 이 인덱스 체계와 classifier 출력 인덱스 체계를 맞춰야 함.
    # teacher_train.py에서는 adjusted_target_idx를 사용.
    # 0: background
    # 1: non_bg_classes[0]
    # 2: non_bg_classes[1]
    # ...

    # Confusion matrix의 display_labels는 config.classes 전체를 사용.
    # score_sum도 이 전체 클래스에 대한 합계를 구하는 것이 더 직관적일 수 있으나,
    # 모델 출력과 직접 더하려면 모델 출력 차원과 같아야 함.
    # 여기서는 모델 출력 차원에 맞춰 score_sum을 만들고, 나중에 pred_idx를 config.classes 인덱스로 변환.

    for path, label_text in audio_label_list:
        segments = parser.load_and_segment(path)
        # GT 라벨 (config.classes 기준 인덱스, 0, 1, 2...)
        gt_label_config_idx = config.get_class_index(label_text) 
        
        # score_sum은 classifier의 출력 차원과 동일해야 함
        # classifier 출력 차원: 1 (bg) + len(non_bg_classes_for_prompts)
        # 이것이 probs의 차원이 됨.
        # 오류 지점 score_sum += probs 에서 probs.shape[0]은 expected_num_classifier_outputs
        # score_sum = torch.zeros(expected_num_classifier_outputs).to(device) # 이렇게 해야 함.

        # 수정: score_sum은 각 파일마다 초기화되어야 함.
        # 이전 코드: score_sum = torch.zeros(config.num_classes).to(class_text_embeddings.device) # 이 부분이 오류의 원인이었음. config.num_classes (3) vs probs (4)
        # 수정된 score_sum 초기화
        score_sum_for_file = torch.zeros(expected_num_classifier_outputs).to(device)


        if not segments:
            print(f"Warning: No segments found for {path}. Skipping.")
            # 이 경우 y_true, y_pred에 추가하지 않거나, 기본값(예: background)으로 처리 가능
            # 여기서는 일단 건너뛰고, 아래에서 y_true, y_pred에 추가할 때 문제 없는지 확인 필요
            continue


        with torch.no_grad():
            for mel_segment_tensor in segments: # 변수명 명확화
                # vild_utils.normalize_mel_shape 사용 권장
                # from vild_utils import normalize_mel_shape
                # mel = normalize_mel_shape(mel_segment_tensor.to(device))
                
                # 임시 정규화 (vild_utils.py의 normalize_mel_shape와 동일하게)
                mel = mel_segment_tensor.to(device)
                if mel.dim() == 2: mel = mel.unsqueeze(0).unsqueeze(0)
                elif mel.dim() == 3: mel = mel.unsqueeze(0)
                # mel.dim() == 4는 그대로 사용

                region_embedding = encoder(mel) # [1, embedding_dim]
                logits = classifier(region_embedding, class_text_embeddings)  # [1, expected_num_classifier_outputs]
                
                # print(f"Debug: logits.shape = {logits.shape}, expected_num_classifier_outputs = {expected_num_classifier_outputs}") # 디버깅용
                # print(f"Debug: score_sum_for_file.shape = {score_sum_for_file.shape}") # 디버깅용

                probs = torch.softmax(logits, dim=1).squeeze(0)  # [expected_num_classifier_outputs]
                
                # print(f"Debug: probs.shape = {probs.shape}") # 디버깅용
                
                score_sum_for_file += probs # 여기가 오류 발생 지점이었음. 이제 차원이 맞아야 함.

        if not segments: # 위에서 continue 했으므로 이 부분은 실행 안될 것 같지만, 방어적으로.
            avg_probs = torch.zeros(expected_num_classifier_outputs).to(device)
            # 기본 예측을 background(인덱스 0)로 설정
            pred_classifier_idx = 0
        else:
            avg_probs = score_sum_for_file / len(segments)
            pred_classifier_idx = torch.argmax(avg_probs).item() # 0:bg, 1:non_bg1, 2:non_bg2 ...

        y_true.append(gt_label_config_idx) # GT는 config.classes 인덱스 (0,1,2...)
        
        # 예측된 classifier 인덱스(pred_classifier_idx)를 config.classes 인덱스로 변환
        # pred_classifier_idx: 0은 background, 1은 non_bg_classes[0], 2는 non_bg_classes[1]...
        # config.classes: ["apartment", "daily", "background"] (예시)
        # non_bg_classes_for_prompts: ["apartment", "daily"] (예시)
        
        pred_label_config_idx = -1 # 초기화 (오류 시 확인용)
        predicted_label_text = "<Unknown Prediction Logic Error>"

        if pred_classifier_idx == 0: # 모델이 background로 예측
            try:
                # config.classes에서 "background" 문자열을 포함하는 첫 번째 클래스의 인덱스를 찾음
                bg_class_candidates = [cls_name for cls_name in config.classes if "background" in cls_name.lower()]
                if not bg_class_candidates:
                    raise ValueError("No 'background' class defined in config.classes for mapping prediction.")
                # 보통 background_noise 하나일 것임
                pred_label_config_idx = config.get_class_index(bg_class_candidates[0])
                predicted_label_text = bg_class_candidates[0]
            except Exception as e:
                print(f"Error mapping predicted background to config.classes: {e}")
                # 안전하게 config.classes의 마지막 인덱스를 background로 가정하거나, 특정 인덱스 사용
                # 또는 오류 발생 시 y_pred에 추가하지 않도록 처리
                # 여기서는 non_bg_classes_for_prompts가 비어있는 경우를 생각해서 처리해야함
                if not non_bg_classes_for_prompts: # 모든 클래스가 백그라운드인 경우
                     if config.classes: pred_label_config_idx = 0 # 첫번째 클래스를 백그라운드로 가정
                else: # non_bg 클래스가 있는데 bg 매핑 실패
                     pred_label_config_idx = len(config.classes) -1 # 마지막 클래스를 bg로 가정 (위험할 수 있음)
                predicted_label_text = config.classes[pred_label_config_idx] if pred_label_config_idx != -1 and pred_label_config_idx < len(config.classes) else "Background (Mapping Error)"

        else: # 모델이 non-background 클래스로 예측 (pred_classifier_idx >= 1)
            # pred_classifier_idx-1은 non_bg_classes_for_prompts의 인덱스
            if (pred_classifier_idx - 1) < len(non_bg_classes_for_prompts):
                predicted_non_bg_class_name = non_bg_classes_for_prompts[pred_classifier_idx-1]
                pred_label_config_idx = config.get_class_index(predicted_non_bg_class_name)
                predicted_label_text = predicted_non_bg_class_name
            else:
                # 로직 오류: pred_classifier_idx가 non_bg_classes_for_prompts 범위를 벗어남
                print(f"Error: pred_classifier_idx {pred_classifier_idx} is out of bounds for non_bg_classes_for_prompts (len: {len(non_bg_classes_for_prompts)}). Defaulting prediction.")
                # 안전하게 첫번째 non-bg 클래스 또는 background로 처리
                if non_bg_classes_for_prompts:
                     pred_label_config_idx = config.get_class_index(non_bg_classes_for_prompts[0])
                     predicted_label_text = non_bg_classes_for_prompts[0]
                else: # non_bg 클래스가 없는 경우 (위에서 처리되었어야 함)
                    bg_class_candidates = [cls_name for cls_name in config.classes if "background" in cls_name.lower()]
                    if bg_class_candidates:
                        pred_label_config_idx = config.get_class_index(bg_class_candidates[0])
                        predicted_label_text = bg_class_candidates[0]
                    elif config.classes: # 최후의 수단
                        pred_label_config_idx = 0 
                        predicted_label_text = config.classes[0]


        if pred_label_config_idx == -1 : # 매핑에 실패한 경우
            print(f"Critical Error: Could not map pred_classifier_idx {pred_classifier_idx} to a config.classes index. path: {path}")
            # y_pred에 유효하지 않은 값을 넣지 않도록 처리하거나, 기본값 설정
            # 여기서는 샘플을 건너뛰거나, 기본 예측 (예: 가장 빈번한 클래스 또는 background)을 할 수 있음.
            # 일단은 y_pred에 추가하지 않고, confusion matrix 생성 시 문제 없는지 확인 필요.
            # 또는, 가장 가능성 높은 background 클래스의 인덱스를 사용
            bg_class_candidates = [cls_name for cls_name in config.classes if "background" in cls_name.lower()]
            if bg_class_candidates:
                y_pred.append(config.get_class_index(bg_class_candidates[0]))
            elif config.classes: # 정말 어쩔 수 없는 경우
                 y_pred.append(0) # 첫번째 클래스로 예측
            # continue # 또는 이 샘플은 건너뜀 (y_true와 y_pred 길이가 달라짐)

        else:
            y_pred.append(pred_label_config_idx) # y_pred는 config.classes 인덱스 (0,1,2...)

        print(f"[{os.path.basename(path)}] GT: {label_text} (idx:{gt_label_config_idx}), Pred: {predicted_label_text} (idx:{pred_label_config_idx}) --- Classifier Raw Idx: {pred_classifier_idx}")


    # Confusion Matrix 출력
    # y_true와 y_pred는 모두 config.classes의 인덱스 체계를 따름 (0, 1, ..., len(config.classes)-1)
    
    # cm_labels는 0부터 len(config.classes)-1 까지의 정수 리스트
    cm_labels = list(range(len(config.classes))) 
    display_labels = config.classes # 실제 클래스 이름들
    
    if not y_true or not y_pred:
        print("Warning: y_true or y_pred is empty. Skipping confusion matrix generation.")
    elif len(y_true) != len(y_pred):
        print(f"Warning: Length mismatch between y_true ({len(y_true)}) and y_pred ({len(y_pred)}). Skipping confusion matrix generation.")
    else:
        # plots 폴더 생성 및 저장 경로 설정 (student_train.py와 유사하게)
        plots_dir = "plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            print(f"'{plots_dir}' directory created.")

        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        
        fig, ax = plt.subplots(figsize=(8, 6)) # figure와 axes를 명시적으로 생성
        disp.plot(cmap=plt.cm.Blues, ax=ax) # 생성된 axes에 그리기
        ax.set_title("Confusion Matrix (Teacher Model)") # 제목에 모델 정보 추가
        
        # x축 라벨 회전 (라벨이 길 경우 겹침 방지)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout() # 레이아웃 자동 조정

        cm_filename = "confusion_matrix_teacher.png" # 파일명에 모델 정보 추가
        cm_save_path = os.path.join(plots_dir, cm_filename)
        plt.savefig(cm_save_path)
        print(f"Confusion matrix 저장 완료 -> {cm_save_path}")
        plt.close(fig) # 메모리 해제를 위해 figure 닫기

if __name__ == "__main__":
    # set_seed(42) # evaluate 함수 내부에서 설정하거나 여기서 한 번만.
    
    # 테스트 리스트 생성 로직 (기존 코드 유지)
    # (2) 두번째 시도는 test_list를 클래스별로 샘플링하여 구성:
    import csv
    from collections import defaultdict

    test_samples_by_label = defaultdict(list)
    test_list = [] # test_list 초기화 추가

    try:
        with open("dataset_index.csv", newline='', encoding='utf-8') as f: # encoding 추가
            reader = csv.DictReader(f)
            for row in reader:
                # 파일 경로 유효성 검사 (옵션)
                if not os.path.exists(row["path"]):
                    print(f"Warning: Path not found in dataset_index.csv, skipping: {row['path']}")
                    continue
                label = row["label"]
                test_samples_by_label[label].append((row["path"], label))

        max_per_class = 30
        for label, samples in test_samples_by_label.items():
            # 클래스가 config.classes에 정의되어 있는지 확인 (옵션)
            if label not in AudioViLDConfig().classes: # 임시 Config 객체 생성 (config 변수가 아직 없음)
                print(f"Warning: Label '{label}' from CSV not in config.classes. Skipping samples for this label.")
                continue
            test_list.extend(samples[:max_per_class])

        if not test_list:
            print("Error: No valid test samples found after processing dataset_index.csv. Evaluation cannot proceed.")
            print("Please check dataset_index.csv and ensure paths are correct and labels match config.classes.")
            # 기본 테스트 리스트 (파일이 존재한다고 가정)
            # test_list = [
            #    ("C:/Users/user/Desktop/AI_model/mark1/data_wav/apartment_test.wav", "apartment_noise"),
            #    ("C:/Users/user/Desktop/AI_model/mark1/data_wav/daily_test.wav", "daily_noise"),
            #    ("C:/Users/user/Desktop/AI_model/mark1/data_wav/background_test.wav", "background_noise")
            # ]
            # print("Using default fallback test_list. Ensure these files exist.")
            sys.exit(1) # 샘플이 없으면 종료


        print(f"총 테스트 샘플 수: {len(test_list)} (라벨별 최대 {max_per_class}개 샘플링)")

    except FileNotFoundError:
        print("Error: dataset_index.csv not found. Evaluation cannot proceed.")
        # 기본 테스트 리스트 (파일이 존재한다고 가정)
        # test_list = [
        #    ("C:/Users/user/Desktop/AI_model/mark1/data_wav/apartment_test.wav", "apartment_noise"),
        #    ("C:/Users/user/Desktop/AI_model/mark1/data_wav/daily_test.wav", "daily_noise"),
        #    ("C:/Users/user/Desktop/AI_model/mark1/data_wav/background_test.wav", "background_noise")
        # ]
        # print("Using default fallback test_list. Ensure these files exist and update paths if necessary.")
        sys.exit(1) # 파일이 없으면 종료
    except Exception as e:
        print(f"Error processing dataset_index.csv: {e}")
        sys.exit(1)
    
    evaluate(test_list, seed_value=42)