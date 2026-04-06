import re
import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageOps

SERPAPI_KEY = "7a42ad207f06e9c06ab29fe54fd67b6f14c565c65ace73ba285aa7adaf011cfa"
IMGBB_API_KEY = "475a4814c14a5d2be0f5775a85d1b450"

st.set_page_config(page_title="빠른 제품 찾기", layout="centered")
st.title("📷 빠른 같은 제품 찾기")
st.write("탐지는 유지하면서 속도를 줄인 버전입니다.")


def preprocess_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def extract_price_number(price_value):
    if price_value is None:
        return 999999999

    if isinstance(price_value, (int, float)):
        return int(float(price_value))

    text = str(price_value)
    nums = re.sub(r"[^0-9.]", "", text)
    if not nums:
        return 999999999

    try:
        return int(float(nums))
    except Exception:
        return 999999999


def upload_to_imgbb(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": IMGBB_API_KEY, "image": img_b64, "name": "photo"},
        timeout=20,
    )

    if response.status_code != 200:
        raise RuntimeError(f"ImgBB 업로드 실패: {response.status_code}")

    data = response.json()
    if not data.get("success"):
        raise RuntimeError("ImgBB 업로드 실패")

    return data["data"]["url"]


def clean_product_title(title: str) -> str:
    if not title:
        return ""

    title = title.strip()
    title = re.sub(r"\[[^\]]*\]", " ", title)
    title = re.sub(r"\([^)]{15,}\)", " ", title)
    title = re.sub(r"[|]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def score_specificity(title: str) -> int:
    if not title:
        return -999

    score = 0
    length = len(title)

    if length < 6:
        return -999

    score += min(length, 40)

    if re.search(r"\d", title):
        score += 10
    if re.search(r"\b(ml|l|oz|mm|cm|gb|tb|w|inch)\b", title.lower()):
        score += 10
    if re.search(r"[A-Z]{2,}\d*", title):
        score += 6

    return score


def normalize_words(text: str):
    text = text.lower()
    words = re.findall(r"[a-zA-Z가-힣0-9]+", text)
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "new",
        "best",
        "official",
        "상품",
        "정품",
        "국내",
        "해외",
        "무료",
        "배송",
        "판매",
        "구매",
        "쇼핑",
        "스토어",
        "브랜드",
    }
    return [w for w in words if len(w) >= 2 and w not in stopwords]


@st.cache_data(show_spinner=False, ttl=3600)
def get_lens_product_candidates(image_url: str, max_candidates=3):
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_KEY,
        "hl": "ko",
        "country": "kr",
        "no_cache": "true",
        "output": "json",
    }

    response = requests.get(
        "https://serpapi.com/search.json", params=params, timeout=40
    )

    if response.status_code != 200:
        raise RuntimeError(f"Lens 요청 실패: {response.status_code}")

    data = response.json()
    candidates = []

    for item in data.get("products", []):
        title = clean_product_title(item.get("title", ""))
        if title:
            candidates.append(title)

    for item in data.get("visual_matches", []):
        title = clean_product_title(item.get("title", ""))
        if title:
            candidates.append(title)

    unique = []
    for title in candidates:
        if title not in unique:
            unique.append(title)

    unique.sort(key=score_specificity, reverse=True)
    return unique[:max_candidates]


@st.cache_data(show_spinner=False, ttl=3600)
def search_google_shopping(query: str):
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "ko",
        "gl": "kr",
        "no_cache": "true",
    }

    response = requests.get(
        "https://serpapi.com/search.json", params=params, timeout=40
    )

    if response.status_code != 200:
        raise RuntimeError(f"Shopping 요청 실패: {response.status_code}")

    data = response.json()
    results = []

    for item in data.get("shopping_results", []):
        title = item.get("title", "").strip()
        link = item.get("product_link") or item.get("link") or ""
        source = item.get("source", "")
        thumbnail = item.get("thumbnail", "")

        price_raw = item.get("price")
        extracted_price = item.get("extracted_price")

        if extracted_price is not None:
            price_num = extract_price_number(extracted_price)
            price_text = str(price_raw) if price_raw is not None else str(price_num)
        else:
            price_num = extract_price_number(price_raw)
            price_text = str(price_raw) if price_raw is not None else ""

        if not title or not link or price_num >= 999999999:
            continue

        results.append(
            {
                "title": title,
                "link": link,
                "source": source,
                "thumbnail": thumbnail,
                "price": price_text,
                "price_num": price_num,
            }
        )

        if len(results) >= 8:
            break

    return results


def relevance_score(target_title: str, candidate_title: str) -> int:
    target_words = set(normalize_words(target_title))
    candidate_words = set(normalize_words(candidate_title))

    if not target_words or not candidate_words:
        return 0

    common = target_words & candidate_words
    score = len(common) * 10

    target_nums = set(re.findall(r"[a-zA-Z]*\d+[a-zA-Z]*", target_title.lower()))
    cand_nums = set
