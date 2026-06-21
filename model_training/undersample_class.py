import os, random

# =========================================================
# [필수 설정] 아래 항목을 본인 환경에 맞게 수정하세요.
# =========================================================
label_dir    = r"본인의_라벨_경로를_여기에_입력하세요"
image_dir    = r"본인의_이미지_경로를_여기에_입력하세요"
TARGET_CLASS = 0     # 줄일 클래스 번호
TARGET_COUNT = 1500  # 남길 개수


class_files = []
for f in os.listdir(label_dir):
    if not f.endswith('.txt'):
        continue
    with open(os.path.join(label_dir, f)) as fp:
        lines = fp.readlines()
    if lines and int(lines[0].split()[0]) == TARGET_CLASS:
        class_files.append(f)

print(f"클래스 {TARGET_CLASS} 파일 수: {len(class_files)}")

if len(class_files) <= TARGET_COUNT:
    print("이미 목표 개수 이하입니다.")
else:
    random.seed(42)
    to_delete = random.sample(class_files, len(class_files) - TARGET_COUNT)
    for f in to_delete:
        os.remove(os.path.join(label_dir, f))
        stem = f.replace('.txt', '')
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(image_dir, stem + ext)
            if os.path.exists(img_path):
                os.remove(img_path)
    print(f"완료! {len(to_delete)}개 삭제됨")