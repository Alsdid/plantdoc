import os
import hashlib

# =========================================================
# [필수 설정] 아래 경로를 본인 환경에 맞게 수정하세요.
# =========================================================

BASE_PATH = r"본인의_데이터셋_최상위_경로를_여기에_입력하세요"

IMG_DIRS = [
    os.path.join('본인의_데이터셋_경로를_여기에_입력하세요')
]

LBL_DIRS = [
    os.path.join('본인의_데이터셋_경로를_여기에_입력하세요')
]

def get_md5(path):
    """파일의 MD5 해시값을 계산하여 반환합니다."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

print("1단계: 지정된 모든 포도 원본 폴더 중복 스캔 시작...")
seen_hashes = set()
img_scan = 0
del_count = 0

for img_dir in IMG_DIRS:
    if not os.path.exists(img_dir):
        print(f"경로를 찾을 수 없어 건너뜁니다: {img_dir}")
        continue
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_scan += 1
                img_path = os.path.join(root, f)
                stem = os.path.splitext(f)[0]
                
                try:
                    h = get_md5(img_path)

                    if h in seen_hashes:
                        os.remove(img_path)
                        del_count += 1
                        
                        json_name = stem + ".json"
                        for lbl_dir in LBL_DIRS:
                            if os.path.exists(lbl_dir):
                                for l_root, l_dirs, l_files in os.walk(lbl_dir):
                                    if json_name in l_files:
                                        os.remove(os.path.join(l_root, json_name))
                                        break
                    else:
                        seen_hashes.add(h)
                except Exception:
                    continue

print(f"스캔: {img_scan}장 | 삭제된 중복 세트: {del_count}개")