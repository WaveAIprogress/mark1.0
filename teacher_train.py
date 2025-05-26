# teacher_train.py
import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# utils 폴더를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from vild_utils import normalize_mel_shape
from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder, ViLDTextHead
from vild_parser import AudioParser
from vild_losses import ViLDLosses
from seed_utils import set_seed # 추가

# AudioSegmentDataset 정의 (vild_parser.py에서 parse_sample 사용)
class LabeledAudioDataset(Dataset):
    def __init__(self, file_label_list, parser, config, text_embedder, device): # text_embedder, device 추가
        self.data = file_label_list  # [(path, label_text), ...]
        self.parser = parser
        self.config = config
        # self.text_embedder = text_embedder # 텍스트 임베딩은 학습 루프 밖에서 한 번만 생성
        # self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label_text = self.data[idx]
        # parse_sample은 (mel_tensor, label_index) 반환
        # mel_tensor는 [num_segments, 1, mel_bins, time]
        # label_index는 정수
        mel_tensor, label_idx = self.parser.parse_sample(path, label_text)
        return mel_tensor, label_idx

def train_teacher(seed_value=42): # seed_value 인자 추가
    set_seed(seed_value) # 함수 시작 시 시드 설정
    config = AudioViLDConfig()
    parser = AudioParser(config) # AudioViLDConfig 인스턴스 전달
    device = config.device if torch.cuda.is_available() else "cpu"

    file_label_list = []
    try:
        with open("dataset_index.csv", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_label_list.append((row["path"], row["label"]))
    except FileNotFoundError:
        print("Error: dataset_index.csv not found. Please run generate_dataset_index.py first.")
        return

    if not file_label_list:
        print("Error: dataset_index.csv is empty or invalid.")
        return
    
    # 텍스트 임베딩 생성 (한 번만)
    # config.classes = ["apartment_noise", "daily_noise", "background_noise"]
    # ViLDTextHead는 background_embedding을 별도로 가지므로, class_text_embeddings는 실제 클래스에 대해서만.
    prompt_texts = [
        f"a sound of {cls.replace('_noise', '').replace('_', ' ')}" for cls in config.classes
    ]
    # prompt_texts = [ # 논문/CLIP 스타일 프롬프트
    #     "a sound of apartment noise",
    #     "a sound of daily life noise",
    #     "a meaningless background noise" # 이 부분은 ViLDTextHead의 background_embedding과 중복될 수 있음
    # ]
    # ViLDTextHead가 background를 처리하므로, config.classes에 있는 non-background 클래스만 임베딩.
    
    text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    # config.classes에 정의된 순서대로 임베딩
    class_text_embeddings = torch.tensor(
        text_model.encode(prompt_texts), 
        dtype=torch.float
    ).to(device)


    full_dataset = LabeledAudioDataset(file_label_list, parser, config, text_model, device)
    
    dataset_size = len(full_dataset)
    if dataset_size < 2:
        print(f"Error: Dataset size ({dataset_size}) is too small to split.")
        return
        
    val_size = int(0.2 * dataset_size)
    if val_size == 0 and dataset_size > 1: val_size = 1
    train_size = dataset_size - val_size

    if train_size == 0:
        print(f"Error: Training dataset size is 0 after split. Original size: {dataset_size}")
        return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(seed_value))
    
    # DataLoader의 shuffle은 학습 순서에 영향을 미치므로 seed로 제어됨
    # batch_size=1, 각 아이템은 (mel_tensor, label_idx)
    # mel_tensor: [num_segments, 1, mel_bins, time]
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    teacher_encoder = SimpleAudioEncoder(config)
    teacher_classifier = ViLDTextHead(config) # 오디오 임베딩과 클래스 텍스트 임베딩 비교

    # Optimizer는 encoder와 classifier 파라미터 모두 포함
    optimizer = optim.Adam(
        list(teacher_encoder.parameters()) + list(teacher_classifier.parameters()), 
        lr=config.learning_rate
    )
    loss_fn_calculator = ViLDLosses(config)

    teacher_encoder.to(device)
    teacher_classifier.to(device)

    best_val_loss = float('inf')
    patience = 3 # 3으로 조정
    trigger_times = 0
    train_loss_history = []
    val_loss_history = []

    print(f"Teacher training started on {device} for {config.num_epochs} epochs.")

    for epoch in range(config.num_epochs):
        teacher_encoder.train()
        teacher_classifier.train()
        epoch_train_loss = 0
        
        for mel_tensor_batch, label_idx_batch in train_loader:
            # batch_size=1이므로, _batch 접미사는 단일 아이템을 의미
            mel_tensor = mel_tensor_batch[0].to(device) # [num_segments, 1, mel_bins, time]
            label_idx = label_idx_batch[0].to(device)   # 스칼라 텐서

            # 각 오디오 파일(샘플)은 여러 세그먼트로 구성됨.
            # ViLD 논문에서는 각 region proposal(여기서는 segment)에 대해 독립적으로 loss 계산 후 평균 가능성.
            # 또는 전체 세그먼트 임베딩의 평균을 내서 한 번 classifier 통과.
            # 여기서는 각 세그먼트에 대해 로짓 계산 후 평균 또는 개별 로짓에 대한 손실 평균.
            # 현재 SimpleAudioEncoder는 [B, 1, T, F]를 받으므로, 세그먼트들을 배치처럼 처리.
            
            # mel_tensor: [num_segments, 1, mel_bins, time]
            # label_idx는 이 모든 세그먼트에 대한 단일 라벨
            
            region_embeddings = teacher_encoder(mel_tensor) # [num_segments, embedding_dim]
            
            # Classifier는 [B, D]와 [C, D]를 받아 [B, C+1] 로짓 반환
            # 여기서 B는 num_segments
            logits = teacher_classifier(region_embeddings, class_text_embeddings) # [num_segments, num_classes+1]
            
            # 모든 세그먼트에 대해 동일한 타겟 라벨 적용
            # label_idx는 0, 1, ... (config.classes 인덱스)
            # ViLDTextHead의 logits은 0=background, 1=class1, ...
            # 따라서 target_labels는 label_idx + 1 이어야 함. (background_noise의 경우 특별 처리 필요 없음, 어차피 ViLDTextHead가 background embedding 사용)
            # 단, config.classes에 background_noise가 있다면, 해당 인덱스 + 1
            # config.get_class_index("background_noise")는 예를 들어 2. 그럼 target은 3.
            # ViLDTextHead는 0을 배경으로, 1~N을 config.classes[0]~config.classes[N-1]에 대응하는 것으로 처리
            # 따라서, target은 label_idx를 그대로 사용하면 안되고, classifier의 출력 형식에 맞춰야 함.
            # label_idx가 0,1,2 (apartment, daily, background 순서 가정)
            # logits은 0:bg, 1:apartment, 2:daily
            # 만약 label_idx가 background(2)이면, logits에서 0번 인덱스에 해당해야.
            # label_idx가 apartment(0)이면, logits에서 1번 인덱스.
            # label_idx가 daily(1)이면, logits에서 2번 인덱스.
            # => targets = label_idx + 1 (만약 background_noise가 config.classes의 마지막이 아니라면 복잡해짐)
            # 가장 간단한 방법: config.classes에서 "background_noise"를 찾아서 그 경우 target=0, 나머지는 index+1

            current_label_text = config.classes[label_idx.item()]
            if "background" in current_label_text: # "background_noise" 등
                targets = torch.tensor([0] * logits.size(0), device=device, dtype=torch.long) # 모든 세그먼트의 타겟은 0 (background)
            else:
                # config.classes에서 "background_noise"를 제외한 순서로 인덱싱
                # 예를 들어 config.classes = ["ap", "dly", "bg"]
                # "ap" -> index 0 -> target 1
                # "dly" -> index 1 -> target 2
                # 이 방식은 config.classes에 "background_noise"가 어디에 있든 유효
                
                # non_bg_classes = [c for c in config.classes if "background" not in c]
                # try:
                #     # 현재 라벨이 non_bg_classes에서 몇 번째인지 찾아서 +1
                #     adjusted_label_idx = non_bg_classes.index(current_label_text) + 1
                # except ValueError: # current_label_text가 non_bg_classes에 없는 경우 (사실상 background인데 위에서 안걸린 경우)
                #     print(f"Logic error: {current_label_text} should be background or in non_bg_classes")
                #     adjusted_label_idx = 0 # 안전하게 배경으로 처리
                
                # 더 간단하게: ViLDTextHead가 config.classes 순서대로 로짓을 생성 (background 제외하고)
                # 그럼 label_idx를 그대로 사용하고, ViLDTextHead가 background 로짓을 맨 앞에 붙인다.
                # 즉, classifier(region_embedding, class_text_embeddings)의 출력은
                # [B, 1 (bg) + C (실제 클래스)] 형태.
                # 이때 target은 (실제 클래스 인덱스 + 1)이 되어야 함.
                # background_noise 라벨의 경우 target은 0.

                # config.get_class_index(label_text)는 0, 1, 2... 를 반환.
                # ViLDTextHead의 logits은 [B, C+1] (0: background, 1~C: classes)
                # 따라서, 만약 label_text가 "background_noise"라면 target은 0.
                # 그렇지 않다면, target은 config.get_class_index(label_text) + 1.
                # label_idx가 이미 config.get_class_index()의 결과.
                
                # label_idx = config.get_class_index(실제 라벨 텍스트)
                # config.classes = ["apartment_noise", "daily_noise", "background_noise"]
                # "apartment_noise" -> label_idx 0 -> target 1
                # "daily_noise" -> label_idx 1 -> target 2
                # "background_noise" -> label_idx 2 -> target 0 (이게 문제)

                # ViLDTextHead의 class_text_embeddings는 config.classes 그대로의 순서로 가정.
                # 그러면 logits은 [B, background_logit, class0_logit, class1_logit, class2_logit]
                # target label은 이 logits의 인덱스여야 함.
                # label_idx가 background_noise를 가리키면 target=0,
                # label_idx가 config.classes[i] (non-background)를 가리키면 target=i+1

                # 현재 class_text_embeddings는 background_noise를 제외하고 생성됨.
                # prompt_texts = [f"a sound of {cls.replace('_noise', '').replace('_', ' ')}" for cls in config.classes IF NOT "background" in cls]
                # 이 경우 ViLDTextHead의 class_text_embeddings 인자는 non-background 클래스 임베딩만 받음.
                # class_text_embeddings: [C_non_bg, D]
                # logits: [B, 1 (bg) + C_non_bg]
                # target: if background, 0. Else, (index in non_bg_classes) + 1

                non_bg_classes = [c for c in config.classes if "background" not in c]
                if "background" in current_label_text:
                    adjusted_target_idx = 0
                else:
                    try:
                        adjusted_target_idx = non_bg_classes.index(current_label_text) + 1
                    except ValueError: # current_label_text가 non_bg_classes에 없는 희귀 케이스
                        print(f"Warning: Label '{current_label_text}' not found in non_background_classes. Treating as background.")
                        adjusted_target_idx = 0
                
                targets = torch.tensor([adjusted_target_idx] * logits.size(0), device=device, dtype=torch.long)


            loss = loss_fn_calculator.compute_text_loss(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_loss_history.append(avg_train_loss)

        # Validation
        teacher_encoder.eval()
        teacher_classifier.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for mel_tensor_batch, label_idx_batch in val_loader:
                mel_tensor = mel_tensor_batch[0].to(device)
                label_idx = label_idx_batch[0].to(device)

                region_embeddings = teacher_encoder(mel_tensor)
                logits = teacher_classifier(region_embeddings, class_text_embeddings)
                
                current_label_text = config.classes[label_idx.item()]
                non_bg_classes = [c for c in config.classes if "background" not in c]
                if "background" in current_label_text:
                    adjusted_target_idx = 0
                else:
                    try:
                        adjusted_target_idx = non_bg_classes.index(current_label_text) + 1
                    except ValueError:
                        adjusted_target_idx = 0 # Fallback
                targets = torch.tensor([adjusted_target_idx] * logits.size(0), device=device, dtype=torch.long)

                loss = loss_fn_calculator.compute_text_loss(logits, targets)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_loss_history.append(avg_val_loss)

        print(f"[Epoch {epoch+1}/{config.num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(teacher_encoder.state_dict(), "best_teacher_encoder.pth")
            torch.save(teacher_classifier.state_dict(), "best_teacher_classifier.pth")
            print(f"New best validation loss: {best_val_loss:.4f}. Models saved.")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"Validation loss did not improve. Trigger: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
    torch.save(teacher_encoder.state_dict(), "teacher_encoder.pth")
    torch.save(teacher_classifier.state_dict(), "teacher_classifier.pth")
    print("Final teacher models (encoder and classifier) saved.")
    
    # --- 그래프 저장 경로 수정 ---
    # plots 폴더 생성 (이미 존재하면 넘어감)
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"'{plots_dir}' directory created.")

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss') # val_loss_history가 비어있지 않은 경우 (즉, val_loader가 있었던 경우)
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Teacher Model Loss Curve (Train vs Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CrossEntropy)') 
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 저장 경로를 plots 폴더 내부로 지정
    plot_filename = "loss_curve_teacher_train_val.png"
    plot_save_path = os.path.join(plots_dir, plot_filename) # os.path.join 사용
    plt.savefig(plot_save_path)
    print(f"'{plot_save_path}' 저장 완료.")
    # --- 그래프 저장 경로 수정 완료 ---
    

if __name__ == "__main__":
    # set_seed(42) # train_teacher 함수 내부에서 설정하거나 여기서 한 번만.
    train_teacher(seed_value=42)