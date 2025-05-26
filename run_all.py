# run_all.py

# Step 4에서 OpenMP 오류때문에 일단 하게 된건데
# PyTorch와 NumPy가 함께 사용될 때, 특히 intel MKL과 관련된 OpenMP 충돌 방지를 위해... 추가.
# student_train.py 보다는 run_all.py에 먼저 추가해봄.
# 새로 설치했다가 깔았다가 하는것도 지쳐서 새로운 방법 시도..

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1' # 또는 '0'으로 시도
# 다음 라인은 OMP 에러 메시지에서 제안하는 임시방편이지만, 먼저 위의 두 라인을 시도. 
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import subprocess
# utils 폴더를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from seed_utils import set_seed 

def run_step(description, command):
    print(f"\n[실행] {description} ...")
    # shell=True는 보안 위험이 있을 수 있으므로, 가능하면 리스트 형태로 명령어를 전달하는 것이 좋습니다.
    # 예: command = ["python", "convert_m4a_to_wav.py"]
    # 여기서는 기존 방식을 유지하되, 참고로 남깁니다.
    result = subprocess.run(command, shell=True, check=False) # check=False로 변경하여 직접 오류 처리
    if result.returncode != 0:
        print(f"[오류] {description} 실패 (반환 코드: {result.returncode}). 파이프라인 중단.")
        exit(1) # 오류 발생 시 스크립트 종료
    print(f"[완료] {description}")

if __name__ == "__main__":
    set_seed(42) # 전체 파이프라인 시작 시 시드 설정
    print("=== 소음 분류 전체 학습 파이프라인을 시작합니다. ===\n")

    # 각 스크립트가 내부적으로 시드를 설정하므로, run_all.py에서의 시드 설정은
    # subprocess로 실행되는 파이썬 스크립트의 시드에 직접 영향을 주지 않음음.
    # 각 스크립트가 자체적으로 set_seed를 호출하도록 하는 것이 중요.
    # run_all.py의 set_seed는 run_all.py 자체의 랜덤 요소(만약 있다면)를 제어함함.

    run_step("Step 0: m4a/wav 파일 처리 (변환 및 복사)", "python convert_m4a_to_wav.py")
    run_step("Step 1: 데이터셋 인덱스 생성", "python generate_dataset_index.py")
    run_step("Step 2: Teacher 모델 학습 (validation/early stopping 포함)", "python train.py --mode teacher")
    run_step("Step 3: Soft Label 추출 (teacher_encoder.pth 기반)", "python extract_soft_label.py")
    run_step("Step 4: Student 모델 학습 (validation/early stopping 포함)", "python train.py --mode student")
    run_step("Step 5: 모델 평가 (선택 사항)", "python eval.py") # 평가 단계 추가 (필요시)
    run_step("Step 6: plot(파형, 멜스펙트로그램) 생성(선택 사항)", "python plot_audio.py")


    print("\n 전체 학습 파이프라인이 성공적으로 완료되었습니다.")