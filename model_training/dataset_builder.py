import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image
import yaml

# =========================================================
# [필수 설정] 아래 경로와 클래스를 본인 환경에 맞게 수정하세요.
# =========================================================

DATA_ROOT = r"본인의_데이터셋_최상위_경로를_여기에_입력하세요"

IMG_DIRS = {
    "정상": rf"본인의_이미지_데이터셋_경로를_여기에_입력하세요",
    "생리장해": rf"본인의_이미지_데이터셋_경로를_여기에_입력하세요",
    "병해": rf"본인의_이미지_데이터셋_경로를_여기에_입력하세요",
}

LBL_DIRS = {
    "정상": rf"본인의_라벨링_데이터셋_경로를_여기에_입력하세요",
    "생리장해": rf"본인의_라벨링_데이터셋_경로를_여기에_입력하세요",
    "병해": rf"본인의_라벨링_경로를_여기에_입력하세요",
}

DATASET_DIR = r"본인이_저장할_경로를_여기에_입력하세요"

# [클래스 이름] CLASS_MAP 순서에 맞게 입력하세요.
# 예시
CLASS_MAP = {
    "00": 0,    # 정상
    "a11": 1,   # 병해 1
    "a12": 2,   # 병해 2
    "b4": 3,    # 생리장해 1
    "b5": 4     # 생리장해 2
}

def json_to_yolo(json_path, img_w, img_h):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    ann = data.get("annotations", {})

    disease_code = ann.get("disease", data.get("description", {}).get("diseaseCode", "00"))
    
    if disease_code not in CLASS_MAP:
        return []
        
    cls_id = CLASS_MAP[disease_code]
    lines = []
    
    # bbox 또는 points 기반 좌표 처리
    for bbox in ann.get("bbox", []):
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["w"]
        h = bbox["h"]
        
        x_center = max(0.0, min(1.0, (x + w / 2) / img_w))
        y_center = max(0.0, min(1.0, (y + h / 2) / img_h))
        w_norm = max(0.0, min(1.0, w / img_w))
        h_norm = max(0.0, min(1.0, h / img_h))
        
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    return lines

all_samples = []
for cls_name, img_dir in IMG_DIRS.items():
    lbl_dir = LBL_DIRS[cls_name]
    
    if not os.path.exists(img_dir):
        print(f"⚠️ 폴더 없음 건너뜀: {img_dir}")
        continue
        
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in imgs:
        stem = Path(img_file).stem
        
        json_path = os.path.join(lbl_dir, stem + ".json")
        if not os.path.exists(json_path):

            for root, dirs, files in os.walk(lbl_dir):
                if (stem + ".json") in files:
                    json_path = os.path.join(root, stem + ".json")
                    break
        
        if os.path.exists(json_path):
            all_samples.append((img_dir, img_file, json_path))

print(f"\n매칭 완료된 총 샘플 수: {len(all_samples)}개")

random.seed(42)
random.shuffle(all_samples)
n = len(all_samples)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

splits = {
    "train": all_samples[:n_train],
    "val": all_samples[n_train:n_train+n_val],
    "test": all_samples[n_train+n_val:]
}


if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

print("파일 복사 및 YOLO 텍스트 라벨링 변환 시작...")
for split, samples in splits.items():
    os.makedirs(f"{DATASET_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/labels/{split}", exist_ok=True)
    
    for img_dir, img_file, json_path in samples:
        shutil.copy2(os.path.join(img_dir, img_file), f"{DATASET_DIR}/images/{split}/{img_file}")
        
        img = Image.open(os.path.join(img_dir, img_file))
        w, h = img.size
        lines = json_to_yolo(json_path, w, h)
        
        stem = Path(img_file).stem
        with open(f"{DATASET_DIR}/labels/{split}/{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
    print(f"{split} 완료: {len(samples)}개")

yaml_data = {
    "path": DATASET_DIR.replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 5,
    "names": ["정상(00)", "병해(a11)", "병해(a12)", "생리장해(b4)", "생리장해(b5)"]
}

with open(f"{DATASET_DIR}/data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(yaml_data, f, allow_unicode=True)

print("data.yaml 생성 완료! 준비 끝! 🍇")