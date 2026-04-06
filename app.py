import re
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st
from PIL import Image, ImageOps
import pytesseract
from ultralytics import YOLO

# -----------------------------
# API 키 (Render에서는 환경변수로 넣어)
# -----------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NAVER_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# -----------------------------
# YOLO 모델
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# 이미지 처리
# -----------------------------
def preprocess(image):
    return ImageOps.exif_transpose(image).convert("RGB")

def detect(image):
    results = model(image)
    if len(results[0].boxes) == 0:
        return image
    x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
    return image.crop((x1, y1, x2, y2))

# -----------------------------
# OCR
# -----------------------------
def get_text(image):
    text = pytesseract.image_to_string(image, lang="eng+kor")
    words = re.findall(r"[A-Za-z0-9]+", text)
    return " ".join(words)

# -----------------------------
# 네이버 쇼핑 API
# -----------------------------
def search_naver(query):
    url = "https://openapi.naver.com/v1/search/shop.json"

    headers = {
        "X-Naver-Client-Id": NAVER_ID,
        "X-Naver-Client-Secret": NAVER_SECRET
    }

    params = {"query": query, "display": 20}

    res = requests.get(url, headers=headers, params=params)
    data = res.json()

    results = []

    for item in data.get("items", []):
        price = int(item.get("lprice", 0))

        results.append({
            "title": re.sub("<.*?>", "", item["title"]),
            "price": price,
            "link": item["link"],
            "mall": item["mallName"]
        })

    return results

# -----------------------------
# SerpAPI (쿠팡 포함)
# -----------------------------
def search_serp(query):
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY
    }

    res = requests.get("https://serpapi.com/search.json", params=params)
    data = res.json()

    results = []

    for item in data.get("shopping_results", []):
        price_text = item.get("price", "")
        price = int(re.sub(r"[^0-9]", "", price_text)) if price_text else 999999

        results.append({
            "title": item["title"],
            "price": price,
            "link": item["link"],
            "mall": item.get("source", "")
        })

    return results

# -----------------------------
# 병렬 검색
# -----------------------------
def search_all(query):
    with ThreadPoolExecutor(max_workers=2) as exe:
        f1 = exe.submit(search_naver, query)
        f2 = exe.submit(search_serp, query)

        return f1.result() + f2.result()

# -----------------------------
# 최종 검색
# -----------------------------
def find_product(image):
    cropped = detect(image)
    text = get_text(cropped)

    query = text if text else "제품"

    results = search_all(query)

    # 중복 제거
    unique = {r["link"]: r for r in results}.values()

    # 최저가 정렬
    sorted_results = sorted(unique, key=lambda x: x["price"])

    return sorted_results[:15], cropped, text

# -----------------------------
# UI
# -----------------------------
st.title("📷 최저가 자동 검색 AI")

img = st.camera_input("사진 찍기")

if img:
    image = Image.open(img)
    image = preprocess(image)

    st.image(image)

    with st.spinner("검색 중..."):
        results, cropped, text = find_product(image)

    st.image(cropped, caption="탐지된 제품")
    st.write("🔍 인식된 텍스트:", text)

    st.subheader("💰 최저가 순")

    for r in results:
        st.write(f"{r['title']}")
        st.write(f"가격: {r['price']}원")
        st.write(f"쇼핑몰: {r['mall']}")
        st.link_button("구매", r["link"])
        st.divider()
