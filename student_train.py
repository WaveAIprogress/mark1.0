# student_train.py (collate_fn_skip_none 및 DataLoader 루프 수정)
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate # 추가
import pickle
import matplotlib.pyplot as plt

# utils 폴더를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from vild_utils import normalize_mel_shape
from audio_utils import prepare_teacher_embedding
from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder
from vild_head import ViLDHead
from vild_parser import AudioParser
from vild_losses import ViLDLosses
from seed_utils import set_seed

class SoftLabelDataset(Dataset):
    def __init__(self, sample_list, parser, config):
        self.samples = []
        if sample_list:
            for path, teacher_embedding in sample_list:
                if not os.path.exists(path):
                    # print(f"Warning (SoftLabelDataset): Path {path} from soft_labels.pkl does not exist. Skipping.")
                    continue
                self.samples.append((path, teacher_embedding))
        
        if not self.samples:
            print("Warning (SoftLabelDataset): No valid samples available after checking paths or initial list was empty.")
        self.parser = parser
        self.config = config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, teacher_embedding_tensor = self.samples[idx]
        segments = self.parser.load_and_segment(path) # list of [1,1,H,W] tensors 또는 []

        if not segments:
            # print(f"Warning (__getitem__): No segments loaded for {path}. Returning None for this sample.")
            return None
        
        # 반환값을 딕셔너리로 변경 (collate_fn에서 명확한 참조를 위해)
        return {"segments_list": segments, "teacher_embedding": teacher_embedding_tensor}

# 수정된 collate_fn (student_train용)
def student_collate_fn(batch):
    # batch는 [{'segments_list': sl1, 'teacher_embedding': te1}, None, ...] 형태의 리스트
    filtered_batch = [item for item in batch if item is not None]
    if not filtered_batch:
        return None

    # 'teacher_embedding' 부분은 default_collate로 텐서로 stack
    teacher_embeddings_batch = default_collate([item['teacher_embedding'] for item in filtered_batch])
    
    # 'segments_list' 부분은 그대로 리스트들의 리스트로 유지
    # batch_size=1 이면, [[seg1,seg2..]sample1] 형태
    segments_lists_batch = [item['segments_list'] for item in filtered_batch]
    
    return segments_lists_batch, teacher_embeddings_batch


def train_student(seed_value=42):
    set_seed(seed_value)
    config = AudioViLDConfig()
    parser = AudioParser(config)
    device = config.device if torch.cuda.is_available() else "cpu"

    raw_sample_list = None
    try:
        with open("soft_labels.pkl", "rb") as f:
            raw_sample_list = pickle.load(f)
    except FileNotFoundError:
        print("Error: soft_labels.pkl not found. Student training cannot proceed.")
        return
    except EOFError:
        print("Error: soft_labels.pkl is empty or corrupted (EOFError). Student training cannot proceed.")
        return
    except pickle.UnpicklingError:
        print("Error: Could not unpickle soft_labels.pkl. File might be corrupted.")
        return

    if not raw_sample_list:
        print("Error: soft_labels.pkl is empty or contains no data. Student training cannot proceed.")
        return

    full_dataset = SoftLabelDataset(raw_sample_list, parser, config)

    if len(full_dataset) == 0:
        print("Error: No valid samples in SoftLabelDataset after filtering. Student training cannot proceed.")
        return
    
    dataset_size = len(full_dataset)
    train_dataset, val_dataset = None, None

    if dataset_size < 2 :
        # print(f"Dataset size ({dataset_size}) is less than 2. Adjusting training/validation split.")
        if dataset_size == 1:
            # print("Only 1 sample available. Using it for training, no validation.")
            train_dataset = full_dataset
    else:
        val_size = int(0.2 * dataset_size)
        if val_size == 0: val_size = 1
        if val_size >= dataset_size : val_size = dataset_size - 1
        train_size = dataset_size - val_size
        
        if train_size <= 0:
            # print(f"Error: Training dataset size not positive ({train_size}). Using all for training.")
            train_dataset = full_dataset
        else:
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(seed_value))

    # 수정된 collate_fn 사용
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=student_collate_fn)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=student_collate_fn)

    student_encoder = SimpleAudioEncoder(config)
    student_head = ViLDHead(config.embedding_dim, config.embedding_dim)
    optimizer = optim.Adam(list(student_encoder.parameters()) + list(student_head.parameters()), lr=config.learning_rate)
    loss_fn_calculator = ViLDLosses(config)
    student_encoder.to(device)
    student_head.to(device)
    best_val_loss = float('inf')
    patience = 3 # 3으로 조정
    trigger_times = 0
    train_loss_history, val_loss_history = [], []

    print(f"Student training started on {device} for {config.num_epochs} epochs.")
    print(f"Training with {len(train_dataset)} samples, Validating with {len(val_dataset) if val_dataset else 0} samples.")

    for epoch in range(config.num_epochs):
        student_encoder.train()
        student_head.train()
        epoch_train_loss = 0
        processed_batches_train = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data is None: # collate_fn이 None 반환 (모든 샘플이 유효하지 않음)
                continue
            
            # student_collate_fn 반환: (segments_lists_batch, teacher_embedding_batch)
            # segments_lists_batch: [ list_of_tensors_for_sample1 ] (batch_size=1)
            # teacher_embedding_batch: Tensor (stacked)
            segments_lists_batch, teacher_embedding_batch = batch_data
            
            # 방어 코드: segments_lists_batch는 길이가 1인 리스트여야 함 (batch_size=1이므로)
            # 그 안의 첫번째 요소가 실제 텐서들의 리스트.
            if not isinstance(segments_lists_batch, list) or len(segments_lists_batch) != 1:
                print(f"Train batch {batch_idx}: segments_lists_batch has unexpected structure. Skipping. Content: {segments_lists_batch}")
                continue
            
            current_sample_segments_list = segments_lists_batch[0] # parser가 반환한 list of tensors
            if not isinstance(current_sample_segments_list, list) or not current_sample_segments_list:
                # print(f"Train batch {batch_idx}: current_sample_segments_list (inner list) is invalid or empty. Skipping. Content: {current_sample_segments_list}")
                continue # 이 샘플은 세그먼트가 없음 (parser가 빈 리스트 반환)
            
            if not isinstance(teacher_embedding_batch, torch.Tensor) or teacher_embedding_batch.numel() == 0:
                # print(f"Train batch {batch_idx}: teacher_embedding_batch is invalid or empty. Skipping.")
                continue
                
            processed_batches_train +=1
            
            segments_list = current_sample_segments_list 
            teacher_embedding_single = teacher_embedding_batch[0] # batch_size=1이므로 첫 번째 teacher embedding
            
            teacher_embedding = prepare_teacher_embedding(teacher_embedding_single, device)

            sample_total_loss = 0
            num_segments_in_sample = len(segments_list)
            if num_segments_in_sample == 0: # 위에서 이미 처리됨
                continue

            for seg_idx, mel_segment_tensor in enumerate(segments_list):
                if not isinstance(mel_segment_tensor, torch.Tensor) or mel_segment_tensor.numel() == 0:
                    # print(f"Train batch {batch_idx}, sample, seg {seg_idx}: Invalid/empty segment tensor. Skipping.")
                    continue
                mel = normalize_mel_shape(mel_segment_tensor.to(device))
                student_raw_embedding = student_encoder(mel)
                student_projected_embedding = student_head(student_raw_embedding)
                loss = loss_fn_calculator.compute_image_loss(student_projected_embedding, teacher_embedding)
                sample_total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_train_loss += (sample_total_loss / num_segments_in_sample if num_segments_in_sample > 0 else 0)

        avg_train_loss = epoch_train_loss / processed_batches_train if processed_batches_train > 0 else 0
        train_loss_history.append(avg_train_loss)

        avg_val_loss_current_epoch = float('inf')
        if val_loader:
            student_encoder.eval()
            student_head.eval()
            epoch_val_loss = 0
            processed_batches_val = 0
            with torch.no_grad():
                for val_batch_idx, batch_data_val in enumerate(val_loader):
                    if batch_data_val is None:
                        continue
                    segments_lists_batch_val, teacher_embedding_batch_val = batch_data_val

                    if not isinstance(segments_lists_batch_val, list) or len(segments_lists_batch_val) != 1:
                        # print(f"Val batch {val_batch_idx}: segments_lists_batch_val unexpected. Skipping.")
                        continue
                    current_sample_segments_list_val = segments_lists_batch_val[0]
                    if not isinstance(current_sample_segments_list_val, list) or not current_sample_segments_list_val:
                        # print(f"Val batch {val_batch_idx}: inner segments list invalid/empty. Skipping.")
                        continue
                    if not isinstance(teacher_embedding_batch_val, torch.Tensor) or teacher_embedding_batch_val.numel() == 0:
                        # print(f"Val batch {val_batch_idx}: teacher_embedding_batch_val invalid/empty. Skipping.")
                        continue
                        
                    processed_batches_val += 1
                    segments_list_val = current_sample_segments_list_val
                    teacher_embedding_single_val = teacher_embedding_batch_val[0]
                    teacher_embedding_val = prepare_teacher_embedding(teacher_embedding_single_val, device)

                    sample_total_loss_val = 0
                    num_segments_in_sample_val = len(segments_list_val)
                    if num_segments_in_sample_val == 0:
                        continue

                    for val_seg_idx, mel_segment_tensor_val in enumerate(segments_list_val):
                        if not isinstance(mel_segment_tensor_val, torch.Tensor) or mel_segment_tensor_val.numel() == 0:
                            continue
                        mel_val = normalize_mel_shape(mel_segment_tensor_val.to(device))
                        student_raw_embedding_val = student_encoder(mel_val)
                        student_projected_embedding_val = student_head(student_raw_embedding_val)
                        loss_val = loss_fn_calculator.compute_image_loss(student_projected_embedding_val, teacher_embedding_val)
                        sample_total_loss_val += loss_val.item()
                    
                    epoch_val_loss += (sample_total_loss_val / num_segments_in_sample_val if num_segments_in_sample_val > 0 else 0)
            
            avg_val_loss_current_epoch = epoch_val_loss / processed_batches_val if processed_batches_val > 0 else float('inf')
            val_loss_history.append(avg_val_loss_current_epoch)
            print(f"[Epoch {epoch+1}/{config.num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss_current_epoch:.4f}")

            if avg_val_loss_current_epoch < best_val_loss:
                best_val_loss = avg_val_loss_current_epoch
                torch.save(student_encoder.state_dict(), "best_student_encoder.pth")
                torch.save(student_head.state_dict(), "best_student_head.pth")
                print(f"New best validation loss: {best_val_loss:.4f}. Models saved.")
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
        else:
            print(f"[Epoch {epoch+1}/{config.num_epochs}] Train Loss: {avg_train_loss:.4f} (No validation)")
            if (epoch + 1) % 10 == 0 or (epoch + 1) == config.num_epochs :
                 torch.save(student_encoder.state_dict(), f"student_encoder_epoch{epoch+1}.pth")
                 torch.save(student_head.state_dict(), f"student_head_epoch{epoch+1}.pth")
                 print(f"Models saved at epoch {epoch+1} (no validation).")

    torch.save(student_encoder.state_dict(), "student_encoder.pth")
    torch.save(student_head.state_dict(), "student_head.pth")
    print("Final student models (encoder and head) saved.")
    
    # --- 그래프 저장 경로 수정 ---
    # plots 폴더 생성 (이미 존재하면 넘어감)
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"'{plots_dir}' directory created.")

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    if val_loss_history: # val_loss_history가 비어있지 않은 경우 (즉, val_loader가 있었던 경우)
        plt.plot(val_loss_history, label='Val Loss')
    plt.title('Student Model Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (L1)') # Student 모델의 Loss가 L1인지 확인 필요 (ViLDLosses.compute_image_loss)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 저장 경로를 plots 폴더 내부로 지정
    plot_filename = "loss_curve_student_train_val.png"
    plot_save_path = os.path.join(plots_dir, plot_filename) # os.path.join 사용
    plt.savefig(plot_save_path)
    print(f"'{plot_save_path}' 저장 완료.")
    # --- 그래프 저장 경로 수정 완료 ---

if __name__ == "__main__":
    train_student(seed_value=42)
    
    