# extract_soft_label.py
import os
import sys
import csv
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate # 기본 collate 임포트

from vild_model import SimpleAudioEncoder
from vild_parser import AudioParser
from vild_config import AudioViLDConfig

# 현재 파일 기준으로 utils 경로 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from seed_utils import set_seed

class AudioSegmentDataset(Dataset):
    def __init__(self, file_list, parser):
        self.files = file_list
        self.parser = parser

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        segments = self.parser.load_and_segment(path) # list of [1,1,H,W] tensors 또는 []
        # print(f"Debug (__getitem__ for {path}): returning type(segments)={type(segments)}, len={len(segments) if isinstance(segments,list) else 'N/A'}")
        return {"path": path, "segments_list": segments} # 딕셔너리 형태로 반환

# 사용자 정의 collate_fn
def custom_collate_for_extract(batch):
    # batch는 [{'path':p1, 'segments_list':s1}, {'path':p2, 'segments_list':s2}, ...] 형태의 리스트
    
    paths = []
    segments_lists = [] # 각 샘플의 'list of tensors'를 담을 리스트

    for item in batch:
        paths.append(item['path'])
        segments_lists.append(item['segments_list']) # parser가 반환한 list of tensors를 그대로 추가

    # paths는 문자열 리스트이므로 default_collate가 튜플로 잘 묶어줌
    # segments_lists는 'list of tensors'들의 리스트.
    # default_collate가 이를 어떻게 처리할지 모르므로, 직접 리스트로 반환.
    # 즉, 최종 반환은 ( ('path1', 'path2', ...), [ [seg1,seg2..]sample1, [segA,segB..]sample2, ... ] )
    
    # batch_size=1인 경우, paths는 ('path1',), segments_lists는 [[seg1,seg2..]sample1]
    return default_collate([item['path'] for item in batch]), segments_lists


def extract_soft_labels(sample_paths, seed_value=42):
    set_seed(seed_value)
    config = AudioViLDConfig()
    parser = AudioParser(config)
    device = config.device if torch.cuda.is_available() else "cpu"

    encoder = SimpleAudioEncoder(config)
    try:
        encoder.load_state_dict(torch.load("teacher_encoder.pth", map_location="cpu"))
    except FileNotFoundError:
        print("Error: teacher_encoder.pth not found. Please train the teacher model first.")
        return
    encoder.to(device)
    encoder.eval()

    dataset = AudioSegmentDataset(sample_paths, parser)
    # 사용자 정의 collate_fn 사용
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_for_extract)

    results = []
    print("\nStarting soft label extraction...")
    with torch.no_grad():
        for i, data_batch in enumerate(dataloader):
            print(f"\n--- DataLoader Batch {i} ---")
            if data_batch is None: # custom_collate_for_extract가 None을 반환할 수 있음 (모든 샘플이 None일때-현재는 그런 로직 없음)
                print("  data_batch is None. Skipping.")
                continue

            # custom_collate_for_extract의 반환값: (path_tuple_batch, segments_lists_batch)
            # path_tuple_batch: ('path1_str',)
            # segments_lists_batch: [ list_of_tensors_for_sample1 ] (parser가 반환한 리스트를 담은 리스트)
            
            path_tuple_batch, segments_lists_batch = data_batch
            
            # [디버깅 출력]
            # print(f"  type(path_tuple_batch): {type(path_tuple_batch)}, content: {path_tuple_batch}")
            # print(f"  type(segments_lists_batch) from DataLoader: {type(segments_lists_batch)}") # Expected: list
            
            if not isinstance(segments_lists_batch, list) or not segments_lists_batch:
                print(f"  Error: segments_lists_batch is not a valid list or is empty. Content: {segments_lists_batch}")
                continue

            path = path_tuple_batch[0] # 실제 경로 문자열
            segments_for_processing = segments_lists_batch[0] # 이것이 parser가 반환한 list of tensors

            # [디버깅 출력]
            # print(f"    type(segments_for_processing) (expected list): {type(segments_for_processing)}")
            if isinstance(segments_for_processing, list):
                # [디버깅 출력]
                # print(f"      len(segments_for_processing): {len(segments_for_processing)}")
                if segments_for_processing:
                    # [디버깅 출력]
                    # print(f"        type(segments_for_processing[0]): {type(segments_for_processing[0])}")
                    if isinstance(segments_for_processing[0], torch.Tensor):
                        # [디버깅 출력]
                        print(f"        segments_for_processing[0].shape: {segments_for_processing[0].shape}")
            elif isinstance(segments_for_processing, torch.Tensor):
                 print(f"      !!! segments_for_processing IS A TENSOR. Shape: {segments_for_processing.shape}")


            if not isinstance(segments_for_processing, list):
                print(f"Error (SoftLabel Batch {i}): Segments for {path} is NOT a list (actual type: {type(segments_for_processing)}). Skipping.")
                continue
            
            if not segments_for_processing:
                print(f"Info (SoftLabel Batch {i}): No segments extracted for {path} by parser. Skipping.")
                continue
            
            # [디버깅 출력]
            # print(f"  Processing {path} with {len(segments_for_processing)} segments.")

            segment_embeddings = []
            for seg_idx, mel_segment_tensor in enumerate(segments_for_processing):
                if not isinstance(mel_segment_tensor, torch.Tensor) or mel_segment_tensor.numel() == 0:
                    print(f"    Warning (SoftLabel Batch {i}, Seg {seg_idx}): Invalid or empty mel_segment_tensor for {path}. Skipping segment.")
                    continue
                if mel_segment_tensor.dim() != 4 or mel_segment_tensor.shape[0] != 1 or mel_segment_tensor.shape[1] != 1:
                    print(f"    Warning (SoftLabel Batch {i}, Seg {seg_idx}): mel_segment_tensor for {path} has unexpected shape {mel_segment_tensor.shape}. Expected [1,1,H,W]. Skipping.")
                    continue
                
                mel = mel_segment_tensor.to(device).to(torch.float32)
                try:
                    embedding = encoder(mel)
                    segment_embeddings.append(embedding)
                except Exception as e_enc:
                    print(f"    Error (SoftLabel Batch {i}, Seg {seg_idx}): during encoder(mel) for {path}, shape {mel.shape}: {e_enc}")
                    continue
            
            if not segment_embeddings:
                print(f"  Warning (SoftLabel Batch {i}): No valid embeddings extracted for {path} after processing segments. Skipping.")
                continue
            
            try:
                stacked_embeddings = torch.stack(segment_embeddings, dim=0)
                mean_embedding = torch.mean(stacked_embeddings, dim=0)
                results.append((path, mean_embedding.cpu()))
            except Exception as e_stack:
                print(f"  Error (SoftLabel Batch {i}): stacking/meaning embeddings for {path}: {e_stack}")
                continue
    
    print("\nSoft label extraction finished.")
    if not results:
        print("No soft labels were successfully extracted.")
        if os.path.exists("soft_labels.pkl"):
            print("Deleting previous (potentially empty or outdated) soft_labels.pkl.")
            try:
                os.remove("soft_labels.pkl")
            except OSError as e_del:
                print(f"Error deleting soft_labels.pkl: {e_del}")
        return

    try:
        with open("soft_labels.pkl", "wb") as f:
            pickle.dump(results, f)
        print(f"\n{len(results)} soft labels saved to soft_labels.pkl.")
    except Exception as e_pkl:
        print(f"Error saving soft_labels.pkl: {e_pkl}")

if __name__ == "__main__":
    sample_paths = []
    csv_file = "dataset_index.csv"
    try:
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "path" not in reader.fieldnames:
                print(f"Error: '{csv_file}' does not contain 'path' column or is malformed.")
                exit()
            for row_idx, row in enumerate(reader):
                if "path" in row and row["path"]:
                    sample_paths.append(row["path"])
                else:
                    print(f"Warning: Row {row_idx+1} in '{csv_file}' has no 'path' or empty path. Skipping.")
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run generate_dataset_index.py first.")
        exit()
    except Exception as e_csv:
        print(f"Error reading {csv_file}: {e_csv}")
        exit()
    
    if not sample_paths:
        print(f"No valid paths found in {csv_file}. Cannot extract soft labels.")
        exit()
    
    print(f"Found {len(sample_paths)} paths in {csv_file} to process for soft labels.")
    extract_soft_labels(sample_paths, seed_value=42)