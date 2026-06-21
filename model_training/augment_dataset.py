import os
import random
import cv2
import albumentations as A
from pathlib import Path

# =========================================================
# [필수 설정] 아래 경로와 클래스를 본인 환경에 맞게 수정하세요.
# =========================================================

DATASET_DIR = r"본인의_데이터셋_최상위_경로를_여기에_입력하세요"
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "images", "train")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "labels", "train")

TARGET_COUNT = #소수 클래스들을 끌어올릴 목표 개수 입력

transform = A.Compose([
    A.HorizontalFlip(p=0.5),                                      
    A.VerticalFlip(p=0.5),                                        
    A.RandomRotate90(p=0.5),                                       
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT), # 미세 구도 변형
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

#사용자 클래스에 맞춰 변형
class_to_files = {0: [], 1: [], 2: [], 3: [], 4: []}
file_to_bboxes = {}

print(" 3단계: 가공 완료된 라벨 스캔 중...")
for lbl_file in os.listdir(TRAIN_LBL_DIR):
    if lbl_file.lower().endswith('.txt'):
        stem = Path(lbl_file).stem
        img_extensions = ['.jpg', '.jpeg', '.png']
        img_file = None
        
        for ext in img_extensions:
            if os.path.exists(os.path.join(TRAIN_IMG_DIR, stem + ext)):
                img_file = stem + ext
                break
                
        if not img_file:
            continue
            
        lbl_path = os.path.join(TRAIN_LBL_DIR, lbl_file)
        bboxes = []
        classes_in_file = set()
        
        with open(lbl_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    bboxes.append([x, y, w, h])
                    classes_in_file.add(cls_id)
                    
        if bboxes:
            file_to_bboxes[stem] = {"img_file": img_file, "bboxes": bboxes, "classes": list(classes_in_file)}
            for cls_id in classes_in_file:
                if cls_id in class_to_files:
                    class_to_files[cls_id].append(stem)

print("\n소수 클래스 균형 증강 엔진 스타트!")
#사용자 클래스에 맞춰 변형
for cls_id in range(1, 5):
    current_count = len(class_to_files[cls_id])
    needed_count = TARGET_COUNT - current_count
    
    print(f" {cls_id}번 클래스 현재 {current_count}개 -> 목표치 달성을 위해 {needed_count}개 증강 필요")
    
    if needed_count <= 0:
        continue
        
    available_stems = class_to_files[cls_id]
    aug_done_count = 0
    
    while aug_done_count < needed_count:
        src_stem = random.choice(available_stems)
        info = file_to_bboxes[src_stem]
        
        img_path = os.path.join(TRAIN_IMG_DIR, info["img_file"])
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels_dummy = [cls_id] * len(info["bboxes"])
        
        try:
            transformed = transform(image=image, bboxes=info["bboxes"], class_labels=labels_dummy)
            aug_img = transformed['image']
            aug_boxes = transformed['bboxes']
            
            if not aug_boxes:
                continue 
                
            new_stem = f"aug_{cls_id}_{aug_done_count}_{src_stem}"
            new_img_name = new_stem + Path(info["img_file"]).suffix
            new_lbl_name = new_stem + ".txt"
            
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(TRAIN_IMG_DIR, new_img_name), aug_img_bgr)
            
            with open(os.path.join(TRAIN_LBL_DIR, new_lbl_name), 'w', encoding='utf-8') as f_out:
                for box in aug_boxes:
                    f_out.write(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                    
            aug_done_count += 1
            
        except Exception:
            continue

print("\n모든 소수 클래스가 증강 완료되었습니다!")