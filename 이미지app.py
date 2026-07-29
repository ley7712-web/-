import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. 페이지 기본 설정 (귀여운 타이틀과 레이아웃)
st.set_page_config(
    page_title="😊 찰칵! 오늘 내 기분은 어떨까?",
    page_icon="🐣",
    layout="centered"
)

# Custom CSS로 귀여운 디자인 스타일 추가
st.markdown("""
    <style>
    .main-title {
        color: #FF7B9C;
        text-align: center;
        font-size: 2.3rem;
        font-weight: bold;
    }
    .sub-title {
        color: #6C5CE7;
        text-align: center;
        font-size: 1.1rem;
    }
    .stButton>button {
        background-color: #FFB7B2;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 2. 헤더 섹션
st.markdown("<p class='main-title'>🐣 찰칵! 내 마음 감정 스캐너 💖</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>카메라를 향해 예쁘게 웃거나 슬픈 표정을 지어보세요!</p>", unsafe_allow_html=True)
st.write("---")

# 3. 모델 및 라벨 로딩 함수 (캐싱 처리로 속도 향상)
@st.cache_resource
def load_teachable_machine_model():
    # 티처블 머신 Keras 모델 로드
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    # 라벨 파일 로드
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

try:
    model, class_names = load_teachable_machine_model()
except Exception as e:
    st.error("⚠️ 모델 파일을 찾을 수 없어요! `keras_model.h5`와 `labels.txt` 파일이 깃허브에 잘 올려져 있는지 확인해 주세요.")
    st.stop()

# 4. 이미지 전처리 및 예측 함수
def predict_mood(image, model, class_names):
    # 티처블 머신 입력 규격인 224x224로 리사이즈 및 중앙 크롭
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 이미지를 넘파이 배열로 변환 후 정규화 (-1 ~ 1)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # 모델 입력 형태에 맞게 배치 차원 추가 (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # 예측 수행
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

# 5. 메인 기능: 카메라 입력
st.subheader("📸 카메라로 내 얼굴 찍기")
st.write("아래 카메라 화면에서 **[사진 촬영]** 버튼을 눌러주세요!")

img_file_buffer = st.camera_input("카메라 화면")

if img_file_buffer is not None:
    # 캡처한 이미지 열기
    image = Image.open(img_file_buffer).convert("RGB")
    
    # 예측 진행
    with st.spinner("🐣 몽글몽글 감정을 분석하는 중..."):
        class_name, confidence_score = predict_mood(image, model, class_names)
    
    # 라벨 이름 정제 (예: "0 웃음" -> "웃음")
    cleaned_label = class_name.split(' ', 1)[-1] if ' ' in class_name else class_name
    confidence_pct = int(confidence_score * 100)
    
    st.write("---")
    
    # 6. 감정 분석 결과 및 귀여운 메시지 출력
    if "웃음" in cleaned_label or "smile" in cleaned_label.lower() or "happy" in cleaned_label.lower():
        st.balloons() # 폭죽 효과!
        st.success(f"### 🎉 오늘 기분: **상쾌한 웃음!** (확신도: {confidence_pct}%)")
        st.info(
            "💖 **상담가 선생님의 메시지:**\n\n"
            "우와! 방실방실 웃는 모습을 보니 제 마음까지 밝아지는 것 같아요! "
            "오늘 하루도 당신의 환한 미소처럼 반짝반짝 빛나는 기분 좋은 일들만 가득할 거예요. 이 밝은 에너지를 친구들에게도 나누어주세요! 🎈"
        )
    else:
        st.warning(f"### 🥺 오늘 기분: **토닥토닥 슬픔...** (확신도: {confidence_pct}%)")
        st.info(
            "💌 **상담가 선생님의 메시지:**\n\n"
            "마음속에 마음대로 되지 않는 속상함이나 슬픔이 조금 머물러 있군요. "
            "슬픈 감정을 느끼는 건 자연스러운 일이에요. 참으려 하지 말고 따뜻한 차 한 잔을 마시거나 "
            "좋아하는 음악을 들으며 나 자신을 꼭 안아주세요. 선생님이 언제나 당신 편에서 응원하고 있을게요! 🧸"
        )

st.write("---")
st.caption("✨ Teachable Machine 모델과 Streamlit을 활용하여 만들어진 따뜻한 감정 분석기입니다. 🎀")