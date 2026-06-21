import os
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import urllib.request

st.set_page_config(page_title="PlantDoc", layout="wide")

st.markdown("""
<style>
.stApp h1 {
    text-align: center;
    color: #2E7D32;
    font-weight: bold;
}
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    font-weight: bold;
    transition: 0.3s;
}
div.stButton > button:first-child:hover {
    background-color: #2E7D32;
    color: white;
}
.report-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #f8fff4;
    border: 1px solid #c8e6c9;
    margin-bottom: 20px;
}
.chat-box {
    background-color: #f1f8e9;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 5px;
    border-left: 3px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MODEL_URLS = {
    "토마토": "https://docs.google.com/uc?export=download&id=1QTq4OFHxeGBVV9YJQCiNIekjriH9heo_",
    "딸기": "https://docs.google.com/uc?export=download&id=1DSQz_DL7msi0AJZzj-cuGwnWeMDqoGxk",
    "고추": "https://docs.google.com/uc?export=download&id=1awPCiqsXJtqwmfk0b5U20LKVw1PjfSFU",
    "파프리카": "https://docs.google.com/uc?export=download&id=1YchXs1WL0fF1IQFCQ1cSgyp10tXUTl-H",
    "포도": "https://docs.google.com/uc?export=download&id=1-o1eIa34W1crGl7Cx7ZI88nypMkEgNE7",
    "오이": "https://docs.google.com/uc?export=download&id=1ASUiAMf1BEiwSo9IFHGVp6M5ZR5bwZt6"
}

MODEL_PATHS = {
    "딸기": os.path.join(WEIGHTS_DIR, "best_s.pt"),
    "토마토": os.path.join(WEIGHTS_DIR, "best_t.pt"),
    "고추": os.path.join(WEIGHTS_DIR, "best_pe.pt"),
    "파프리카": os.path.join(WEIGHTS_DIR, "best_pa.pt"),
    "오이": os.path.join(WEIGHTS_DIR, "best_c.pt"),
    "포도": os.path.join(WEIGHTS_DIR, "best_g.pt")
}

IMG_SIZE = {
    "딸기": 960,
    "토마토": 960,
    "고추": 640,
    "파프리카": 640,
    "오이": 960,
    "포도": 640,
}

DISEASE_GUIDE_BY_PLANT = {
    "고추": {
        "0": "정상", "1": "병해 - 탄저병", "2": "병해 - 흰가루병",
        "3": "생리장해 - 칼슘(Ca) 결핍", "4": "생리장해 - 질소(N) 결핍",
        "5": "생리장해 - 인산(P) 결핍", "6": "생리장해 - 칼륨(K) 결핍"
    },
    "파프리카": {
        "0": "정상", "1": "병해 - 흰가루병", "2": "병해 - 모잘록병",
        "3": "생리장해 - 칼슘(Ca) 결핍", "4": "생리장해 - 질소(N) 결핍",
        "5": "생리장해 - 인산(P) 결핍", "6": "생리장해 - 칼륨(K) 결핍"
    },
    "포도": {
        "0": "정상", "1": "병해 - 탄저병", "2": "병해 - 노균병",
        "3": "생리장해 - 축과병", "4": "생리장해 - 일소",
    },
    "딸기": {
        "00": "정상",
        "a1": "잿빛곰팡이병",
        "a2": "흰가루병",
        "b1": "냉해피해",
        "b6": "질소(N) 결핍",
        "b7": "인산(P) 결핍",
        "b8": "칼륨(K) 결핍",
    },
    "토마토": {
        "00": "정상",
        "a5": "잿빛곰팡이병",
        "a6": "흰가루병",
        "b2": "칼슘결핍",
        "b3": "열과",
        "b6": "질소(N) 결핍",
        "b7": "인산(P) 결핍",
        "b8": "칼륨(K) 결핍",
    },
    "오이": {
        "00": "정상",
        "a4": "흰가루병",
        "a3": "노균병",
        "b1": "냉해피해",
        "b6": "질소(N) 결핍",
        "b7": "인산(P) 결핍",
        "b8": "칼륨(K) 결핍",
    },
}

DISEASE_DETAIL = {
    "딸기": {
        "정상": "현재 이미지에서는 뚜렷한 병해나 생리장애 증상이 확인되지 않습니다. 지속적인 관찰을 유지하세요.",
        "병해 - 잿빛곰팡이병": """**진단 결과: 딸기 잿빛곰팡이병이 의심됩니다.**\n\n**주요 증상**\n- 열매가 물러지고 갈색으로 부패\n- 감염 부위에 회색 곰팡이 발생\n\n**관리 방법**\n- 하우스 내부 습도를 낮추고 환기를 강화합니다.\n- 병든 열매와 잎은 즉시 제거합니다.""",
        "병해 - 흰가루병": """**진단 결과: 딸기 흰가루병이 의심됩니다.**\n\n**주요 증상**\n- 잎과 줄기에 흰 가루 형태의 병반 발생\n- 잎이 위로 말리거나 변형\n\n**관리 방법**\n- 밀식 재배를 피하고 통풍을 좋게 합니다.\n- 병든 잎은 제거하여 전염원을 줄입니다.""",
        "생리장해 - 냉해피해": """**진단 결과: 딸기 냉해피해가 의심됩니다.**\n\n**주요 증상**\n- 잎 가장자리 갈변 또는 마름\n- 꽃이 갈색으로 변하거나 낙화\n\n**관리 방법**\n- 저온 예보 시 보온 시설을 가동합니다.\n- 냉기 유입을 막고 야간 보온을 강화합니다.""",
        "생리장해 - 다량 원소 결핍": """**진단 결과: 딸기 다량원소 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎 색이 연녹색 또는 황색으로 변함\n- 생육 저하 및 과실 크기 감소\n\n**관리 방법**\n- 생육 단계에 맞는 균형 시비를 실시합니다.""",
    },
    "토마토": {
        "정상": "현재 이미지에서는 뚜렷한 병해나 생리장애 증상이 확인되지 않습니다. 지속적인 관찰을 유지하세요.",
        "병해 - 잿빛곰팡이병": """**진단 결과: 토마토 잿빛곰팡이병이 의심됩니다.**\n\n**주요 증상**\n- 잎과 줄기에 갈색 병반 발생\n- 과실이 물러지고 부패\n\n**관리 방법**\n- 환기를 강화하여 습도를 낮춥니다.\n- 병든 잎, 줄기, 과실은 제거합니다.""",
        "병해 - 흰가루병": """**진단 결과: 토마토 흰가루병이 의심됩니다.**\n\n**주요 증상**\n- 잎 표면에 흰 가루 모양 병반 발생\n- 잎이 누렇게 변하거나 마름\n\n**관리 방법**\n- 하우스 내부 통풍을 개선합니다.\n- 병든 잎은 조기에 제거합니다.""",
        "생리장해 - 칼슘(Ca) 결핍": """**진단 결과: 토마토 칼슘결핍이 의심됩니다.**\n\n**주요 증상**\n- 과실 아랫부분이 갈색 또는 검게 변함\n\n**관리 방법**\n- 토양 수분을 일정하게 유지합니다.\n- 필요 시 칼슘제를 보조적으로 공급합니다.""",
        "생리장해 - 열과": """**진단 결과: 토마토 열과가 의심됩니다.**\n\n**주요 증상**\n- 과실 표면에 세로 또는 원형 균열 발생\n\n**관리 방법**\n- 관수량을 일정하게 유지합니다.""",
        "생리장해 - 질소(N) 결핍": """**진단 결과: 토마토 질소 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎이 노랗게 변하거나 생육이 약해짐\n\n**관리 방법**\n- 결핍 원소에 맞는 비료를 보충합니다.""",
        "생리장해 - 인산(P) 결핍": """**진단 결과: 토마토 인산 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎 뒷면이 자주색 또는 적자색으로 변함\n\n**관리 방법**\n- 토양 pH를 적정 범위로 유지합니다.""",
        "생리장해 - 칼륨(K) 결핍": """**진단 결과: 토마토 칼륨 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎 가장자리부터 갈변 및 마름\n\n**관리 방법**\n- 칼륨 비료를 적절히 공급합니다.""",
    },
    "오이": {
        "정상": "현재 이미지에서는 뚜렷한 병해나 생리장애 증상이 확인되지 않습니다. 지속적인 관찰을 유지하세요.",
        "병해 - 노균병": """**진단 결과: 오이 노균병이 의심됩니다.**\n\n**주요 증상**\n- 잎에 노란색 또는 갈색의 각진 병반 발생\n\n**관리 방법**\n- 환기와 배습을 강화합니다.\n- 병든 잎은 제거하여 전염원을 줄입니다.""",
        "병해 - 흰가루병": """**진단 결과: 오이 흰가루병이 의심됩니다.**\n\n**주요 증상**\n- 잎 표면에 흰 가루 형태 병반 발생\n\n**관리 방법**\n- 통풍을 개선하고 과습을 피합니다.""",
        "생리장해 - 냉해피해": """**진단 결과: 오이 냉해피해가 의심됩니다.**\n\n**주요 증상**\n- 잎이 축 처지거나 시듦\n\n**관리 방법**\n- 저온기에는 보온 관리가 필요합니다.""",
        "생리장해 - 질소(N) 결핍": """**진단 결과: 오이 질소 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎이 연녹색 또는 노란색으로 변함\n\n**관리 방법**\n- 생육 단계에 맞는 균형 시비를 실시합니다.""",
        "생리장해 - 인산(P) 결핍": """**진단 결과: 오이 인산 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎 뒷면이 자주색으로 변함\n\n**관리 방법**\n- 토양 pH를 적정 범위로 유지합니다.""",
        "생리장해 - 칼륨(K) 결핍": """**진단 결과: 오이 칼륨 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎 가장자리부터 갈변 및 마름\n\n**관리 방법**\n- 칼륨 비료를 적절히 공급합니다.""",
    },
    "고추": {
        "정상": "현재 이미지에서는 뚜렷한 병해나 생리장애 증상이 확인되지 않습니다. 지속적인 관찰을 유지하세요.",
        "병해 - 탄저병": """**진단 결과: 고추 탄저병이 의심됩니다.**\n\n**주요 증상**\n- 과실에 갈색 또는 검은색 함몰 병반 발생\n- 병반 위에 주황색 포자 형성\n\n**관리 방법**\n- 병든 과실은 즉시 제거합니다.\n- 통풍을 강화하고 과습을 피합니다.""",
        "병해 - 흰가루병": """**진단 결과: 고추 흰가루병이 의심됩니다.**\n\n**주요 증상**\n- 잎과 줄기에 흰 가루 형태 병반 발생\n\n**관리 방법**\n- 통풍을 개선하고 과습을 피합니다.""",
        "생리장해 - 칼슘(Ca) 결핍": """**진단 결과: 고추 칼슘 결핍이 의심됩니다.**\n\n**주요 증상**\n- 과실 끝부분이 갈색 또는 검게 변함\n\n**관리 방법**\n- 토양 수분을 일정하게 유지합니다.""",
        "생리장해 - 질소(N) 결핍": """**진단 결과: 고추 질소 결핍이 의심됩니다.**\n\n**주요 증상**\n- 아랫잎부터 황화 시작\n\n**관리 방법**\n- 질소 비료를 적절히 공급합니다.""",
        "생리장해 - 인산(P) 결핍": """**진단 결과: 고추 인산 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎 뒷면이 자주색 또는 적자색으로 변함\n\n**관리 방법**\n- 토양 pH를 적정 범위로 유지합니다.""",
        "생리장해 - 칼륨(K) 결핍": """**진단 결과: 고추 칼륨 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎 가장자리부터 갈변 및 마름\n\n**관리 방법**\n- 칼륨 비료를 적절히 공급합니다.""",
    },
    "파프리카": {
        "정상": "현재 이미지에서는 뚜렷한 병해나 생리장애 증상이 확인되지 않습니다. 지속적인 관찰을 유지하세요.",
        "병해 - 흰가루병": """**진단 결과: 파프리카 흰가루병이 의심됩니다.**\n\n**주요 증상**\n- 잎과 줄기에 흰 가루 형태 병반 발생\n\n**관리 방법**\n- 통풍을 개선하고 과습을 피합니다.""",
        "병해 - 모잘록병": """**진단 결과: 파프리카 모잘록병이 의심됩니다.**\n\n**주요 증상**\n- 줄기 기부가 갈색으로 변하고 잘록해짐\n\n**관리 방법**\n- 배수가 잘 되는 토양을 사용합니다.""",
        "생리장해 - 칼슘(Ca) 결핍": """**진단 결과: 파프리카 칼슘 결핍이 의심됩니다.**\n\n**주요 증상**\n- 과실 끝부분이 갈색 또는 검게 변함\n\n**관리 방법**\n- 토양 수분을 일정하게 유지합니다.""",
        "생리장해 - 질소(N) 결핍": """**진단 결과: 파프리카 질소 결핍이 의심됩니다.**\n\n**주요 증상**\n- 아랫잎부터 황화 시작\n\n**관리 방법**\n- 질소 비료를 적절히 공급합니다.""",
        "생리장해 - 인산(P) 결핍": """**진단 결과: 파프리카 인산 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎 뒷면이 자주색으로 변함\n\n**관리 방법**\n- 토양 pH를 적정 범위로 유지합니다.""",
        "생리장해 - 칼륨(K) 결핍": """**진단 결과: 파프리카 칼륨 결핍이 의심됩니다.**\n\n**주요 증상**\n- 잎 가장자리부터 갈변 및 마름\n\n**관리 방법**\n- 칼륨 비료를 적절히 공급합니다.""",
    },
    "포도": {
        "정상": "현재 이미지에서는 뚜렷한 병해나 생리장애 증상이 확인되지 않습니다. 지속적인 관찰을 유지하세요.",
        "병해 - 탄저병": """**진단 결과: 포도 탄저병이 의심됩니다.**\n\n**주요 증상**\n- 과실에 갈색 함몰 병반 발생\n\n**관리 방법**\n- 병든 과실은 즉시 제거합니다.""",
        "병해 - 노균병": """**진단 결과: 포도 노균병이 의심됩니다.**\n\n**주요 증상**\n- 잎에 노란색 또는 갈색의 각진 병반 발생\n\n**관리 방법**\n- 환기와 배습을 강화합니다.""",
        "생리장해 - 축과병": """**진단 결과: 포도 축과병이 의심됩니다.**\n\n**주요 증상**\n- 과실이 정상보다 작고 쪼그라듦\n\n**관리 방법**\n- 수분 관리를 균일하게 합니다.""",
        "생리장해 - 일소": """**진단 결과: 포도 일소 피해가 의심됩니다.**\n\n**주요 증상**\n- 과실 표면이 하얗게 변하거나 함몰\n\n**관리 방법**\n- 차광망을 설치하여 직사광선을 줄입니다.""",
    },
}

@st.cache_resource
def load_model(plant):
    path = MODEL_PATHS.get(plant)
    url = MODEL_URLS.get(plant)
    if path is None:
        return None, "모델 경로가 등록되지 않았습니다."
        
    if not os.path.exists(path):
        with st.spinner(f"{plant} AI 모델 가중치를 외부 클라우드에서 자동 다운로드 중입니다... (최초 1회 소요)"):
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                return None, f"다운로드 실패. README를 확인해 수동으로 넣어주세요. 에러: {e}"
    try:
        return YOLO(path, task="detect"), None
    except Exception as e:
        return None, str(e)

def get_guide(plant_type, code):
    if code == "탐지 실패":
        return "탐지된 질병이 없습니다.", ""
    name = DISEASE_GUIDE_BY_PLANT.get(plant_type, {}).get(str(code), "알 수 없는 클래스")
    detail = DISEASE_DETAIL.get(plant_type, {}).get(name, "")
    return name, detail

def draw_all_detections(image, result):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return draw_img, detections
    for box in result.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        raw_code = str(cls_id)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        detections.append({"code": raw_code, "confidence": conf, "box": (x1, y1, x2, y2)})
        label = f"Class {raw_code} {conf * 100:.1f}%"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=4)
        draw.text((x1 + 6, max(0, y1 - 25)), label, fill=(255, 255, 255))
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return draw_img, detections

def crop_best_region(image, detections):
    if not detections:
        return image
    x1, y1, x2, y2 = detections[0]["box"]
    return image.crop((x1, y1, x2, y2))

def save_and_show_report(plant_name, plant_type, disease_code, confidence_score, annotated_img, crop_img, detections):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_report = {
        "plant_name": plant_name,
        "plant_type": plant_type,
        "disease_code": disease_code,
        "confidence": confidence_score,
        "annotated_img": annotated_img,
        "crop_img": crop_img,
        "detections": detections,
        "datetime": now,
    }
    st.session_state["history"].append(new_report)
    st.session_state["active_report"] = new_report
    st.rerun()

if "history" not in st.session_state:
    st.session_state["history"] = []
if "active_report" not in st.session_state:
    st.session_state["active_report"] = None
if "diagnosis_result" not in st.session_state:
    st.session_state["diagnosis_result"] = None
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "show_chat" not in st.session_state:
    st.session_state["show_chat"] = False
if "nickname" not in st.session_state:
    st.session_state["nickname"] = ""

with st.sidebar:
    st.header("📂 리포트 보관함")
    if st.session_state["history"]:
        plant_names = {}
        for idx, item in enumerate(st.session_state["history"]):
            name = item["plant_name"]
            if name not in plant_names:
                plant_names[name] = []
            plant_names[name].append((idx, item))
        for name, records in plant_names.items():
            with st.expander(f"🪴 {name}"):
                for idx, item in records:
                    guide_name, _ = get_guide(item["plant_type"], item["disease_code"])
                    date_str = item.get("datetime", "")
                    btn_label = f"{date_str} - {guide_name}"
                    if st.button(btn_label, key=f"history_{idx}", use_container_width=True):
                        st.session_state["active_report"] = item
                        st.session_state["show_chat"] = False
                        st.rerun()
    else:
        st.info("저장된 진단 기록이 없습니다.")

    st.markdown("---")
    if st.button("💬 커뮤니티 채팅", use_container_width=True):
        st.session_state["show_chat"] = not st.session_state["show_chat"]
        st.session_state["active_report"] = None
        st.rerun()

st.title("🌿 PlantDoc")

if st.button("새 식물 진단", type="primary", use_container_width=True):
    st.session_state["active_report"] = None
    st.session_state["diagnosis_result"] = None
    st.session_state["show_chat"] = False
    st.rerun()

st.markdown("---")

if st.session_state["show_chat"]:
    st.subheader("💬 커뮤니티 채팅")
    st.caption("현재 접속한 사용자들과 식물에 대해 이야기하세요!")

    if not st.session_state["nickname"]:
        st.info("채팅을 시작하려면 닉네임을 설정하세요.")
        nick_col1, nick_col2 = st.columns([3, 1])
        with nick_col1:
            new_nick = st.text_input("닉네임", placeholder="닉네임을 입력하세요", label_visibility="collapsed")
        with nick_col2:
            if st.button("설정", use_container_width=True):
                if new_nick.strip():
                    st.session_state["nickname"] = new_nick.strip()
                    st.rerun()
                else:
                    st.warning("닉네임을 입력하세요.")
    else:
        st.success(f"닉네임: **{st.session_state['nickname']}**")
        if st.button("닉네임 변경"):
            st.session_state["nickname"] = ""
            st.rerun()

        st.markdown("---")

        st.markdown("#### ✏️ 새 글 작성")
        chat_msg = st.text_input("메시지", placeholder="메시지를 입력하세요", label_visibility="collapsed")
        chat_image = st.file_uploader("사진 첨부 (선택)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if st.button("전송 💬", use_container_width=True):
            if chat_msg.strip():
                img_data = None
                if chat_image is not None:
                    img_data = Image.open(chat_image).convert("RGB")
                st.session_state["chat_messages"].append({
                    "name": st.session_state["nickname"],
                    "message": chat_msg.strip(),
                    "time": datetime.now().strftime("%H:%M"),
                    "image": img_data,
                    "comments": [],
                })
                st.rerun()
            else:
                st.warning("메시지를 입력하세요.")

        st.markdown("---")

        if st.session_state["chat_messages"]:
            for i, msg in enumerate(reversed(st.session_state["chat_messages"])):
                real_idx = len(st.session_state["chat_messages"]) - 1 - i

                st.markdown(f"""
                <div class="chat-box">
                <strong>🌱 {msg['name']}</strong> <small style="color:gray">{msg['time']}</small><br>
                {msg['message']}
                </div>
                """, unsafe_allow_html=True)

                if msg.get("image") is not None:
                    st.image(msg["image"], width=300)

                if msg.get("comments"):
                    for comment in msg["comments"]:
                        st.markdown(f"""
                        <div style="margin-left:20px; background-color:#e8f5e9; border-radius:8px; padding:8px; margin-bottom:3px; border-left:2px solid #81c784;">
                        <strong>↩ {comment['name']}</strong> <small style="color:gray">{comment['time']}</small><br>
                        {comment['message']}
                        </div>
                        """, unsafe_allow_html=True)

                with st.expander("💬 댓글 달기"):
                    comment_text = st.text_input(
                        "댓글",
                        placeholder="댓글을 입력하세요",
                        key=f"comment_input_{real_idx}",
                        label_visibility="collapsed"
                    )
                    if st.button("댓글 등록", key=f"comment_btn_{real_idx}", use_container_width=True):
                        if comment_text.strip():
                            st.session_state["chat_messages"][real_idx]["comments"].append({
                                "name": st.session_state["nickname"],
                                "message": comment_text.strip(),
                                "time": datetime.now().strftime("%H:%M"),
                            })
                            st.rerun()
                        else:
                            st.warning("댓글 내용을 입력하세요.")

                st.markdown("---")
        else:
            st.info("아직 메시지가 없습니다. 첫 메시지를 남겨보세요!")

elif st.session_state.get("active_report"):
    active_report = st.session_state["active_report"]
    st.subheader(f"📋 [{active_report['plant_name']}] 진단 리포트")
    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.image(active_report["annotated_img"], caption="탐지 결과", use_container_width=True)
        if active_report["crop_img"] is not None:
            st.image(active_report["crop_img"], caption="주요 병변 영역", use_container_width=True)

    with col2:
        plant_type = active_report["plant_type"]
        disease_code = active_report["disease_code"]
        confidence = active_report["confidence"]
        guide_name, guide_detail = get_guide(plant_type, disease_code)

        if disease_code == "탐지 실패":
            st.warning("⚠️ 병변이 탐지되지 않았습니다.")
        elif disease_code in ["00", "0"]:
            st.success("✅ 정상")
        else:
            st.error(f"🚨 {guide_name}")

        st.progress(min(int(confidence), 100), text=f"탐지 신뢰도: {confidence:.2f}%")

        st.markdown(f"""
        <div class="report-box">
        **작물 종류:** {plant_type}  
        **식물 이름:** {active_report['plant_name']}  
        **신뢰도:** {confidence:.2f}%  
        **진단 일시:** {active_report.get('datetime', '')}
        </div>
        """, unsafe_allow_html=True)

        if guide_detail:
            st.markdown("### 💡 진단 정보")
            st.markdown(guide_detail)

else:
    st.subheader("📸 식물 진단")
    col1, col2 = st.columns([1.1, 1])

    with col1:
        plant_type = st.selectbox("1. 작물 선택", ["딸기", "토마토", "고추", "파프리카", "오이", "포도"])
        plant_name_input = st.text_input("2. 식물 이름 입력", placeholder="예: 방울이")
        uploaded_file = st.file_uploader("3. 사진 업로드", type=["jpg", "jpeg", "png"])
        conf_threshold_percent = st.slider(
            "4. 신뢰도 기준 (%)",
            min_value=0, max_value=100, value=10, step=5,
            help="이 값보다 낮으면 결과를 제공하지 않습니다. 버튼으로 확인 가능합니다."
        )

        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file).convert("RGB")
            st.image(raw_image, caption="업로드 원본 이미지", use_container_width=True)

            if st.button("🔍 진단 시작", type="primary", use_container_width=True):
                if not plant_name_input.strip():
                    st.warning("식물 이름을 입력하세요.")
                else:
                    model, error = load_model(plant_type)
                    if model is None:
                        st.error(error)
                    else:
                        with st.spinner("분석 중..."):
                            img_np = np.array(raw_image)
                            results = model.predict(
                                source=img_np,
                                imgsz=IMG_SIZE.get(plant_type, 640),
                                conf=0.01,
                                iou=0.5,
                                max_det=10,
                                verbose=False
                            )
                            result = results[0]
                            annotated_img, detections = draw_all_detections(raw_image, result)

                        if len(detections) == 0:
                            st.warning("⚠️ 검출된 질병 영역이 없습니다.")
                            st.session_state["diagnosis_result"] = None
                        else:
                            abnormal_detections = [d for d in detections if d["code"] != "0"]
                            best = abnormal_detections[0] if abnormal_detections else detections[0]
                            st.session_state["diagnosis_result"] = {
                                "plant_name": plant_name_input.strip(),
                                "plant_type": plant_type,
                                "disease_code": best["code"],
                                "confidence": best["confidence"] * 100,
                                "annotated_img": annotated_img,
                                "crop_img": crop_best_region(raw_image, detections),
                                "detections": detections,
                            }

        result_data = st.session_state.get("diagnosis_result")
        if result_data:
            confidence_score = result_data["confidence"]
            if confidence_score < conf_threshold_percent:
                st.warning(f"⚠️ 신뢰도 기준 미달 ({confidence_score:.1f}%)")
                st.info("탐지된 결과가 있지만 신뢰도가 낮습니다. 그래도 결과를 확인하시겠습니까?")
                if st.button(f"📋 결과 보기 ({confidence_score:.1f}%)", key="low_conf_btn"):
                    st.session_state["diagnosis_result"] = None
                    save_and_show_report(
                        result_data["plant_name"], result_data["plant_type"],
                        result_data["disease_code"], confidence_score,
                        result_data["annotated_img"], result_data["crop_img"],
                        result_data["detections"]
                    )
            else:
                st.session_state["diagnosis_result"] = None
                save_and_show_report(
                    result_data["plant_name"], result_data["plant_type"],
                    result_data["disease_code"], confidence_score,
                    result_data["annotated_img"], result_data["crop_img"],
                    result_data["detections"]
                )

    with col2:
        st.write("### 💡 서비스 이용 가이드")
        st.markdown("1. 작물 종류 선택 ➡️ 2. 식물 이름 입력 ➡️ 3. 사진 업로드 ➡️ 4. 진단 시작")