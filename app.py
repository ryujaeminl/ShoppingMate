import os
import re
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import streamlit as st
from PIL import Image, ImageOps

# --- 설정 및 API 키 (Streamlit Secrets 권장) ---
# 로컬 테스트 시에는 os.getenv를 사용하거나 secrets.toml을 활용하세요.
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")
IMGBB_API_KEY = st.secrets.get("IMGBB_API_KEY") or os.getenv("IMGBB_API_KEY")

st.set_page_config(page_title="사진으로 같은 제품 찾기", layout="centered")
st.title("🛒 Shoppingmate")
st.write("제품은 넓게 탐지하고, 최종 결과는 가장 똑같은 쇼핑 상품만 보여줍니다.")

TIMEOUT_UPLOAD = 30
TIMEOUT_SEARCH = 45
MAX_WORKERS = 4
MAX_LENS_CANDIDATES = 6
MAX_COMBINED_CANDIDATES = 8
MAX_RESULTS_TO_SHOW = 10

session = requests.Session()
session.headers.update({"User-Agent": "Shoppingmate/1.1"})

# --- 유틸리티 함수 ---

def preprocess_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")

def resize_for_upload(image: Image.Image, max_size=(800, 800)) -> Image.Image:
    """업로드 속도 향상을 위한 이미지 리사이징"""
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def center_crop_image(image: Image.Image, crop_ratio=0.7) -> Image.Image:
    width, height = image.size
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image.crop((left, top, right, bottom))

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

def request_json(url, *, method="GET", params=None, data=None, timeout=30):
    last_error = None
    for attempt in range(3):
        try:
            if method == "POST":
                response = session.post(url, data=data, timeout=timeout)
            else:
                response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
            if attempt == 2: break
    raise RuntimeError(f"요청 실패: {last_error}")

def upload_to_imgbb(image: Image.Image) -> str:
    if not IMGBB_API_KEY:
        raise RuntimeError("IMGBB_API_KEY가 설정되지 않았습니다.")
    
    # 최적화된 리사이징 적용
    optimized_img = resize_for_upload(image)
    buffer = BytesIO()
    optimized_img.save(buffer, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    data = request_json(
        "https://api.imgbb.com/1/upload",
        method="POST",
        data={"key": IMGBB_API_KEY, "image": img_b64, "name": "query_photo"},
        timeout=TIMEOUT_UPLOAD,
    )
    url = data.get("data", {}).get("url")
    if not url:
        raise RuntimeError("ImgBB URL 추출 실패")
    return url

def clean_product_title(title: str) -> str:
    if not title: return ""
    title = title.strip()
    title = re.sub(r"\[[^\]]*\]", " ", title)
    title = re.sub(r"\([^)]{15,}\)", " ", title)
    title = re.sub(r"[|/]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title

def score_specificity(title: str) -> int:
    if not title: return -999
    score = 0
    length = len(title)
    if length < 6: return -999
    score += min(length, 40)
    if re.search(r"\d", title): score += 12
    if re.search(r"\b(ml|l|oz|mm|cm|gb|tb|w|inch|in|호|kg)\b", title.lower()): score += 15
    if re.search(r"[A-Z]{2,}\d*", title): score += 8
    
    generic_words = {"스타벅스", "삼성", "애플", "나이키", "텀블러", "컵", "신발", "가방", "상품", "물병"}
    if title in generic_words: score -= 30
    return score

def normalize_words(text: str):
    text = text.lower()
    words = re.findall(r"[a-zA-Z가-힣0-9]+", text)
    stopwords = {"the", "and", "for", "with", "new", "best", "official", "상품", "정품", "국내", "해외"}
    return [w for w in words if len(w) >= 2 and w not in stopwords]

def get_lens_product_candidates(image_url: str, max_candidates=MAX_LENS_CANDIDATES):
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_KEY,
        "hl": "ko", "gl": "kr", "no_cache": "true"
    }
    data = request_json("https://serpapi.com/search.json", params=params, timeout=TIMEOUT_SEARCH)
    candidates = []
    for item in data.get("products", []) + data.get("visual_matches", []):
        title = clean_product_title(item.get("title", ""))
        if title: candidates.append(title)
    
    unique = []
    seen = set()
    for title in candidates:
        if title.lower() not in seen:
            seen.add(title.lower())
            unique.append(title)
    unique.sort(key=score_specificity, reverse=True)
    return unique[:max_candidates]

def search_google_shopping(query: str):
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "ko", "gl": "kr", "no_cache": "true"
    }
    data = request_json("https://serpapi.com/search.json", params=params, timeout=TIMEOUT_SEARCH)
    results = []
    for item in data.get("shopping_results", []):
        title = item.get("title", "").strip()
        link = item.get("product_link") or item.get("link") or ""
        price_raw = item.get("price")
        price_num = extract_price_number(item.get("extracted_price") or price_raw)
        
        if not title or not link or price_num >= 999999999: continue
        results.append({
            "title": title, "link": link, "source": item.get("source", ""),
            "thumbnail": item.get("thumbnail", ""), "price": str(price_raw), "price_num": price_num
        })
    return results

def relevance_score(target_title: str, candidate_title: str) -> int:
    target_words = set(normalize_words(target_title))
    candidate_words = set(normalize_words(candidate_title))
    if not target_words: return 0
    common = target_words & candidate_words
    score = len(common) * 10
    # 숫자/모델명 매칭 강조
    t_nums = set(re.findall(r"\d+", target_title))
    c_nums = set(re.findall(r"\d+", candidate_title))
    score += len(t_nums & c_nums) * 20
    return score

def filter_relevant_results(target_title: str, results: list):
    scored = []
    for item in results:
        item["match_score"] = relevance_score(target_title, item["title"])
        scored.append(item)
    
    filtered = [x for x in scored if x["match_score"] >= 10]
    if not filtered: filtered = sorted(scored, key=lambda x: x["match_score"], reverse=True)[:5]
    
    # 중복 제거 (판매처와 가격이 같은 경우)
    unique = []
    seen = set()
    for item in filtered:
        key = (item["price_num"], item["source"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return sorted(unique, key=lambda x: (-x["match_score"], x["price_num"]))

def evaluate_candidate(candidate: str):
    results = search_google_shopping(candidate)
    filtered = filter_relevant_results(candidate, results)
    if not filtered: return {"candidate": candidate, "top_match": 0, "avg_match": 0, "count": 0, "results": []}
    
    avg_match = sum(x["match_score"] for x in filtered[:3]) / min(len(filtered), 3)
    return {
        "candidate": candidate, "top_match": filtered[0]["match_score"],
        "avg_match": avg_match, "count": len(filtered), "results": filtered
    }

def find_best_product_results(orig_img, crop_img):
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(upload_to_imgbb, orig_img)
        f2 = ex.submit(upload_to_imgbb, crop_img)
        urls = [f1.result(), f2.result()]

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(get_lens_product_candidates, urls[0])
        f2 = ex.submit(get_lens_product_candidates, urls[1])
        candidates = list(dict.fromkeys(f1.result() + f2.result()))[:MAX_COMBINED_CANDIDATES]

    evaluated = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(evaluate_candidate, c): c for c in candidates}
        for f in as_completed(futures):
            evaluated.append(f.result())

    if not evaluated: return None, [], []
    evaluated.sort(key=lambda x: (x["top_match"], x["avg_match"]), reverse=True)
    best = evaluated[0]
    return best["candidate"], best["results"], evaluated

def show_result_card(i, item, is_cheapest=False):
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            if item["thumbnail"]: st.image(item["thumbnail"], use_container_width=True)
        with col2:
            title_prefix = "✨ [최저가] " if is_cheapest else f"{i}. "
            st.markdown(f"### {title_prefix}{item['title']}")
            st.write(f"**가격:** {item['price']}")
            st.write(f"**판매처:** {item['source']}")
            st.link_button(f"구매 페이지 이동", item["link"], use_container_width=True)
        st.divider()

# --- UI 레이아웃 ---

cam_data = st.camera_input("제품 사진 촬영")
uploaded_file = st.file_uploader("또는 사진 업로드", type=["jpg", "jpeg", "png"])

query_image = None
if cam_data: query_image = Image.open(cam_data)
elif uploaded_file: query_image = Image.open(uploaded_file)

if not SERPAPI_KEY or not IMGBB_API_KEY:
    st.error("API 키가 설정되지 않았습니다. Secrets 설정을 확인하세요.")
    st.stop()

if query_image:
    query_image = preprocess_image(query_image)
    cropped_image = center_crop_image(query_image, 0.7)
    
    st.image(query_image, caption="원본 이미지", use_container_width=True)
    
    if st.button("🔍 같은 제품 찾기", type="primary", use_container_width=True):
        with st.status("분석 중...", expanded=True) as status:
            try:
                best_title, shopping_results, logs = find_best_product_results(query_image, cropped_image)
                
                if not shopping_results:
                    status.update(label="검색 결과가 없습니다.", state="error")
                else:
                    status.update(label="검색 완료!", state="complete")
                    st.subheader(f"🏷️ 예상 제품: {best_title}")
                    
                    # 가격순 정렬 (이미 필터링에서 일치도 순이므로, 상위권 내에서 가격 재정렬 가능)
                    # 여기서는 상위 10개 중 진짜 최저가를 찾습니다.
                    display_items = shopping_results[:MAX_RESULTS_TO_SHOW]
                    min_price = min(x["price_num"] for x in display_items)
                    
                    for i, item in enumerate(display_items, 1):
                        show_result_card(i, item, is_cheapest=(item["price_num"] == min_price))
            except Exception as e:
                status.update(label="오류 발생", state="error")
                st.error(f"에러 내용: {e}")
