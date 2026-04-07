import os
import re
import base64
import hashlib
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
from PIL import Image, ImageOps

# -----------------------------
# API 키
# -----------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="ShoppingMate", layout="centered")
st.title("ShoppingMate")

# -----------------------------
# 세션
# -----------------------------
session = requests.Session()

# -----------------------------
# 유틸
# -----------------------------
def preprocess_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_for_search(image: Image.Image, max_side=1200) -> Image.Image:
    width, height = image.size
    max_current = max(width, height)

    if max_current <= max_side:
        return image

    scale = max_side / max_current
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size)


def center_crop_image(image: Image.Image, crop_ratio=0.75) -> Image.Image:
    width, height = image.size
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))


def image_to_hash(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return hashlib.md5(buffer.getvalue()).hexdigest()


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


@st.cache_data(ttl=3600, show_spinner=False)
def upload_image_b64_to_imgbb(img_b64: str) -> str:
    response = session.post(
        "https://api.imgbb.com/1/upload",
        data={
            "key": IMGBB_API_KEY,
            "image": img_b64,
            "name": "photo"
        },
        timeout=20
    )

    if response.status_code != 200:
        raise RuntimeError(f"ImgBB 업로드 실패: {response.status_code} / {response.text}")

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(f"ImgBB 업로드 실패: {data}")

    return data["data"]["url"]


def upload_to_imgbb(image: Image.Image) -> str:
    image = resize_for_search(image, max_side=1200)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return upload_image_b64_to_imgbb(img_b64)


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

    if length < 5:
        return -999

    score += min(length, 35)

    if re.search(r"\d", title):
        score += 12

    if re.search(r"\b(ml|l|oz|mm|cm|gb|tb|w|inch)\b", title.lower()):
        score += 12

    if re.search(r"[A-Z]{2,}\d*", title):
        score += 8

    generic_words = {
        "스타벅스", "삼성", "애플", "나이키", "텀블러", "컵",
        "신발", "가방", "상품", "물병", "마우스", "키보드"
    }
    if title in generic_words:
        score -= 25

    return score


def normalize_words(text: str):
    text = text.lower()
    words = re.findall(r"[a-zA-Z가-힣0-9]+", text)
    stopwords = {
        "the", "and", "for", "with", "new", "best", "official",
        "상품", "정품", "국내", "해외", "무료", "배송", "판매", "구매",
        "쇼핑", "스토어", "브랜드", "옵션", "선택", "할인"
    }
    return [w for w in words if len(w) >= 2 and w not in stopwords]


def extract_key_tokens(title: str):
    lower = title.lower()

    known_brands = [
        "samsung", "apple", "nike", "adidas", "lg", "sony", "asus",
        "logitech", "anker", "starbucks", "삼성", "애플", "나이키",
        "아디다스", "엘지", "스타벅스", "로지텍"
    ]
    brands = set()
    for brand in known_brands:
        if brand in lower:
            brands.add(brand)

    model_tokens = set(re.findall(r"[a-zA-Z]{1,5}[-]?\d{2,}[a-zA-Z0-9-]*", title))
    size_tokens = set(re.findall(r"\d+(?:\.\d+)?\s?(?:ml|l|oz|mm|cm|gb|tb|w|inch)", lower))
    color_tokens = set(re.findall(r"(black|white|silver|blue|red|pink|green|gold|purple|gray|grey|블랙|화이트|실버|블루|레드|핑크|그린|골드|퍼플|그레이)", lower))

    return {
        "brands": brands,
        "models": model_tokens,
        "sizes": size_tokens,
        "colors": color_tokens
    }


# -----------------------------
# 1단계: 제품 탐지
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_lens_product_candidates(image_url: str, max_candidates=4):
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_KEY,
        "hl": "ko",
        "country": "kr",
        "no_cache": "true",
        "output": "json"
    }

    response = session.get(
        "https://serpapi.com/search.json",
        params=params,
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(f"Lens 요청 실패: {response.status_code} / {response.text}")

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
    seen = set()
    for title in candidates:
        key = title.lower()
        if key not in seen:
            seen.add(key)
            unique.append(title)

    unique.sort(key=score_specificity, reverse=True)
    return unique[:max_candidates]


def get_combined_candidates(original_image: Image.Image):
    original_url = upload_to_imgbb(original_image)
    original_candidates = get_lens_product_candidates(original_url, max_candidates=4)

    # 원본 후보가 괜찮으면 crop 생략
    if original_candidates and score_specificity(original_candidates[0]) >= 28:
        return original_candidates[:3], "original_only"

    cropped_image = center_crop_image(original_image, 0.75)
    cropped_url = upload_to_imgbb(cropped_image)
    cropped_candidates = get_lens_product_candidates(cropped_url, max_candidates=3)

    merged = []
    seen = set()
    for title in original_candidates + cropped_candidates:
        key = title.lower()
        if key not in seen:
            seen.add(key)
            merged.append(title)

    merged.sort(key=score_specificity, reverse=True)
    return merged[:3], "original+crop"


# -----------------------------
# 2단계: 쇼핑 검색
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def search_google_shopping(query: str):
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "ko",
        "gl": "kr",
        "no_cache": "true"
    }

    response = session.get(
        "https://serpapi.com/search.json",
        params=params,
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(f"Shopping 요청 실패: {response.status_code} / {response.text}")

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

        results.append({
            "title": title,
            "link": link,
            "source": source,
            "thumbnail": thumbnail,
            "price": price_text,
            "price_num": price_num
        })

    return results


# -----------------------------
# 3단계: 일치도 계산
# -----------------------------
def relevance_score(target_title: str, candidate_title: str) -> int:
    target_words = set(normalize_words(target_title))
    candidate_words = set(normalize_words(candidate_title))

    if not target_words or not candidate_words:
        return 0

    common_words = target_words & candidate_words
    score = len(common_words) * 8

    t = extract_key_tokens(target_title)
    c = extract_key_tokens(candidate_title)

    brand_match = t["brands"] & c["brands"]
    model_match = t["models"] & c["models"]
    size_match = t["sizes"] & c["sizes"]
    color_match = t["colors"] & c["colors"]

    score += len(brand_match) * 25
    score += len(model_match) * 40
    score += len(size_match) * 20
    score += len(color_match) * 8

    # 숫자 토큰 보조
    target_nums = set(re.findall(r"[a-zA-Z]*\d+[a-zA-Z]*", target_title.lower()))
    cand_nums = set(re.findall(r"[a-zA-Z]*\d+[a-zA-Z]*", candidate_title.lower()))
    if target_nums and cand_nums:
        score += len(target_nums & cand_nums) * 10

    # 불일치 패널티
    if t["models"] and c["models"] and not model_match:
        score -= 25

    if t["sizes"] and c["sizes"] and not size_match:
        score -= 18

    if t["brands"] and c["brands"] and not brand_match:
        score -= 20

    return score


def filter_relevant_results(target_title: str, results: list):
    scored = []

    for item in results:
        score = relevance_score(target_title, item["title"])
        new_item = item.copy()
        new_item["match_score"] = score
        scored.append(new_item)

    filtered = [x for x in scored if x["match_score"] >= 12]

    if not filtered:
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        filtered = scored[:5]

    filtered.sort(key=lambda x: (-x["match_score"], x["price_num"]))
    return filtered


def result_consistency_score(results: list) -> float:
    if len(results) < 2:
        return 0.0

    titles = [set(normalize_words(x["title"])) for x in results[:3]]
    if len(titles) < 2:
        return 0.0

    total = 0
    count = 0
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            inter = len(titles[i] & titles[j])
            total += inter
            count += 1

    return total / count if count else 0.0


# -----------------------------
# 4단계: 최적 후보 찾기
# -----------------------------
def search_candidate_bundle(candidate: str):
    shopping_results = search_google_shopping(candidate)
    filtered_results = filter_relevant_results(candidate, shopping_results)

    if not filtered_results:
        return {
            "candidate": candidate,
            "results": [],
            "top_match": 0,
            "avg_match": 0,
            "consistency": 0
        }

    top_match = filtered_results[0]["match_score"]
    avg_match = sum(x["match_score"] for x in filtered_results[:3]) / min(len(filtered_results), 3)
    consistency = result_consistency_score(filtered_results)

    return {
        "candidate": candidate,
        "results": filtered_results,
        "top_match": top_match,
        "avg_match": avg_match,
        "consistency": consistency
    }


def find_best_product_results(original_image: Image.Image):
    candidates, mode = get_combined_candidates(original_image)

    if not candidates:
        return None, [], [], mode

    best_title = None
    best_results = []
    best_score_tuple = (-999, -999, -999)

    all_candidate_logs = []

    with ThreadPoolExecutor(max_workers=min(3, len(candidates))) as executor:
        future_map = {
            executor.submit(search_candidate_bundle, candidate): candidate
            for candidate in candidates
        }

        for future in as_completed(future_map):
            candidate = future_map[future]
            try:
                bundle = future.result()
                filtered_results = bundle["results"]
                top_match = bundle["top_match"]
                avg_match = bundle["avg_match"]
                consistency = bundle["consistency"]

                all_candidate_logs.append({
                    "candidate": candidate,
                    "top_match": top_match,
                    "count": len(filtered_results),
                    "consistency": round(consistency, 2)
                })

                current_tuple = (top_match, avg_match, consistency)

                if current_tuple > best_score_tuple:
                    best_score_tuple = current_tuple
                    best_title = candidate
                    best_results = filtered_results

            except Exception:
                all_candidate_logs.append({
                    "candidate": candidate,
                    "top_match": 0,
                    "count": 0,
                    "consistency": 0
                })

    if not best_title:
        return candidates[0], [], all_candidate_logs, mode

    best_results.sort(key=lambda x: x["price_num"])
    return best_title, best_results, all_candidate_logs, mode


# -----------------------------
# 입력
# -----------------------------
cam_data = st.camera_input("제품 사진을 찍으세요")
uploaded_file = st.file_uploader("또는 사진 업로드", type=["jpg", "jpeg", "png"])

query_image = None
if cam_data is not None:
    query_image = Image.open(cam_data)
elif uploaded_file is not None:
    query_image = Image.open(uploaded_file)

if not SERPAPI_KEY or not IMGBB_API_KEY:
    st.error("SERPAPI_KEY와 IMGBB_API_KEY를 환경변수에 넣어주세요.")
    st.stop()

# -----------------------------
# 실행
# -----------------------------
if query_image is not None:
    query_image = preprocess_image(query_image)
    query_image = resize_for_search(query_image, max_side=1400)

    st.image(query_image, caption="원본 이미지", use_container_width=True)

    with st.expander("검색 보조용 중심 영역 보기"):
        cropped_preview = center_crop_image(query_image, 0.75)
        st.image(cropped_preview, caption="중앙 crop 이미지", use_container_width=True)

    if st.button("같은 제품 찾기", use_container_width=True):
        with st.status("제품 탐지 및 쇼핑 검색 중...", expanded=True) as status:
            try:
                status.write("1) 이미지 전처리 및 업로드 준비 중...")
                status.write("2) Lens로 제품 후보 탐지 중...")
                status.write("3) 후보별 쇼핑 검색 및 비교 중...")

                best_title, shopping_results, candidate_logs, mode = find_best_product_results(query_image)

                if not best_title:
                    status.update(label="제품명을 찾지 못했습니다.", state="error")
                    st.warning("제품 후보를 찾지 못했습니다.")
                elif not shopping_results:
                    status.update(label="같은 제품에 가까운 쇼핑 결과를 찾지 못했습니다.", state="error")
                    st.warning("가까운 쇼핑 결과가 없습니다.")
                    st.info(f"탐지된 대표 후보: {best_title}")
                else:
                    status.write(f"선택된 대표 제품명: **{best_title}**")
                    status.write(f"검색 방식: **{mode}**")
                    status.update(label="검색 완료", state="complete")

                    st.subheader("탐지된 제품")
                    st.info(f"이 사진은 **{best_title}** 와(과) 가장 가깝습니다.")

                    st.subheader("🛒 가장 똑같은 쇼핑 결과")
                    for i, item in enumerate(shopping_results[:10], start=1):
                        with st.container():
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                if item["thumbnail"]:
                                    st.image(item["thumbnail"], use_container_width=True)

                            with col2:
                                st.markdown(f"### {i}. {item['title']}")
                                st.write(f"**가격:** {item['price']}")
                                if item["source"]:
                                    st.write(f"쇼핑몰: {item['source']}")
                                st.write(f"일치도 점수: {item['match_score']}")

                                st.link_button(
                                    f"구매 페이지 이동 {i}",
                                    item["link"],
                                    use_container_width=True
                                )

                            st.divider()

                    with st.expander("후보 비교 보기"):
                        for log in sorted(candidate_logs, key=lambda x: x["top_match"], reverse=True):
                            st.write(
                                f"- {log['candidate']} | 최고 일치도: {log['top_match']} | "
                                f"결과 수: {log['count']} | 일관성: {log['consistency']}"
                            )

            except Exception as e:
                status.update(label="오류 발생", state="error")
                st.error(f"실행 중 오류: {e}")

else:
    st.info("사진을 찍거나 업로드한 뒤 버튼을 누르세요.")
