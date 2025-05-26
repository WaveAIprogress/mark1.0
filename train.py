# train.py
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from seed_utils import set_seed # 추가

def main():
    # 여기서 시드를 설정하면, import되는 모듈의 top-level 랜덤 연산에는 영향 X
    # 각 학습 함수(train_teacher, train_student) 내부에서 시드 설정 권장
    # 또는, 각 함수에 seed 값을 전달하여 설정하도록 수정
    
    parser = argparse.ArgumentParser(description="Train Teacher or Student model.")
    parser.add_argument('--mode', type=str, choices=['teacher', 'student'], required=True, help="학습 모드 선택 (teacher 또는 student)")
    parser.add_argument('--seed', type=int, default=42, help="전역 랜덤 시드 값") # 시드 인자 추가
    args = parser.parse_args()

    set_seed(args.seed) # 명령줄 인자로 받은 시드 설정 또는 기본값 사용

    if args.mode == "teacher":
        from teacher_train import train_teacher
        print(f"ViLD-text Teacher 모델 학습을 시작합니다. (Seed: {args.seed})")
        train_teacher(seed_value=args.seed) # 함수에 시드 전달
    elif args.mode == "student":
        from student_train import train_student
        print(f"ViLD-image Student 모델 학습을 시작합니다. (Seed: {args.seed})")
        train_student(seed_value=args.seed) # 함수에 시드 전달

if __name__ == "__main__":
    main()