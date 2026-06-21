from ultralytics import YOLO

# =========================================================
# [필수 설정] 아래 경로를 본인 환경에 맞게 수정하세요.
# =========================================================

DATASET_DIR = r"본인의_데이터셋_최상위_경로를_여기에_입력하세요"

model = YOLO("yolov8s.pt") #yolov8n 또는 yolov8s등 원하는 모델 입력

results = model.train(
    data=f"{DATASET_DIR}/data.yaml",   
    epochs=30,                         
    imgsz=960,                          
    batch=8,                            
    device="cpu",                    
    patience=10,                       
    save=True,
    project=r"저장할 주소",         
    name="저장할 이름",
)

print("\n학습이 완료되었습니다")