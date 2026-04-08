import os
import re
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
from PIL import Image, ImageOps

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

st.set_page_config(page_title="사진으로 같은 제품 찾기", layout="centered")
st.title("Shoppingmate")
st.write("제품은 넓게 탐지하고, 최종 결과는 가장 똑같은 쇼핑 상품만 보여줍니다.")

TIMEOUT_UPLOAD = 30
TIMEOUT_SEARCH = 45
MAX_WORKERS = 4
MAX_LENS_CANDIDATES = 6
MAX_COMBINED_CANDIDATES = 8
MAX_RESULTS_TO_SHOW = 10

session = requests.Session()
session.headers.update({"User-Agent": "Shoppingmate/1.1"})


def preprocess_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


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
            if attempt == 2:
                break
    raise RuntimeError(f"요청 실패: {last_error}")


def upload_to_imgbb(image: Image.Image) -> str:
    if not IMGBB_API_KEY:
        raise RuntimeError("IMGBB_API_KEY가 없습니다.")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    data = request_json(
        "https://api.imgbb.com/1/upload",
        method="POST",
        data={"key": IMGBB_API_KEY, "image": img_b64, "name": "photo"},
        timeout=TIMEOUT_UPLOAD,
    )

    if not data.get("success"):
        raise RuntimeError(f"ImgBB 업로드 실패: {data}")

    url = data.get("data", {}).get("url")
    if not url:
        raise RuntimeError("ImgBB URL이 없습니다.")
    return url


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
        score += 12
    if re.search(r"\b(ml|l|oz|mm|cm|gb|tb|w|inch|in)\b", title.lower()):
        score += 10
    if re.search(r"[A-Z]{2,}\d*", title):
        score += 8

    generic_words = {
        "스타벅스", "삼성", "애플", "나이키", "텀블러", "컵",
        "신발", "가방", "상품", "물병", "마우스", "키보드"
    }
    if title in generic_words:
        score -= 30

    vague_patterns = [r"^상품", r"^제품", r"^굿즈", r"케이스$", r"액세서리$"]
    if any(re.search(p, title) for p in vague_patterns):
        score -= 12

    return score


def normalize_words(text: str):
    text = text.lower()
    words = re.findall(r"[a-zA-Z가-힣0-9]+", text)
    stopwords = {
        "the", "and", "for", "with", "new", "best", "official",
        "상품", "정품", "국내", "해외", "무료", "배송", "판매", "구매",
        "쇼핑", "스토어", "브랜드", "특가", "행사", "최저가"
    }
    return [w for w in words if len(w) >= 2 and w not in stopwords]


def get_lens_product_candidates(image_url: str, max_candidates=MAX_LENS_CANDIDATES):
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_KEY,
        "hl": "ko",
        "country": "kr",
        "no_cache": "true",
        "output": "json"
    }

    data = request_json("https://serpapi.com/search.json", params=params, timeout=TIMEOUT_SEARCH)
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


def get_combined_candidates(original_image: Image.Image, cropped_image: Image.Image):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_original = executor.submit(upload_to_imgbb, original_image)
        future_cropped = executor.submit(upload_to_imgbb, cropped_image)
        original_url = future_original.result()
        cropped_url = future_cropped.result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_original = executor.submit(get_lens_product_candidates, original_url, MAX_LENS_CANDIDATES)
        future_cropped = executor.submit(get_lens_product_candidates, cropped_url, MAX_LENS_CANDIDATES)
        original_candidates = future_original.result()
        cropped_candidates = future_cropped.result()

    merged = []
    seen = set()
    for title in original_candidates + cropped_candidates:
        key = title.lower()
        if key not in seen:
            seen.add(key)
            merged.append(title)

    merged.sort(key=score_specificity, reverse=True)
    return merged[:MAX_COMBINED_CANDIDATES]


def search_google_shopping(query: str):
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "ko",
        "gl": "kr",
        "no_cache": "true"
    }

    data = request_json("https://serpapi.com/search.json", params=params, timeout=TIMEOUT_SEARCH)
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


def relevance_score(target_title: str, candidate_title: str) -> int:
    target_words = set(normalize_words(target_title))
    candidate_words = set(normalize_words(candidate_title))
    if not target_words or not candidate_words:
        return 0

    common = target_words & candidate_words
    score = len(common) * 10

    target_nums = set(re.findall(r"[a-zA-Z]*\d+[a-zA-Z]*", target_title.lower()))
    cand_nums = set(re.findall(r"[a-zA-Z]*\d+[a-zA-Z]*", candidate_title.lower()))
    if target_nums and cand_nums:
        score += len(target_nums & cand_nums) * 18

    if candidate_title.lower().startswith(target_title.lower()[:12]):
        score += 6

    return score


def filter_relevant_results(target_title: str, results: list):
    scored = []
    for item in results:
        score = relevance_score(target_title, item["title"])
        new_item = item.copy()
        new_item["match_score"] = score
        scored.append(new_item)

    filtered = [x for x in scored if x["match_score"] >= 10]
    if not filtered:
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        filtered = scored[:5]

    deduped = []
    seen = set()
    for item in filtered:
        key = (clean_product_title(item["title"]).lower(), item["price_num"], item["source"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    deduped.sort(key=lambda x: (-x["match_score"], x["price_num"]))
    return deduped


def evaluate_candidate(candidate: str):
    shopping_results = search_google_shopping(candidate)
    filtered_results = filter_relevant_results(candidate, shopping_results)

    if not filtered_results:
        return {
            "candidate": candidate,
            "top_match": 0,
            "avg_match": 0,
            "count": 0,
            "results": [],
            "error": ""
        }

    top_match = filtered_results[0]["match_score"]
    avg_match = sum(x["match_score"] for x in filtered_results[:3]) / min(len(filtered_results), 3)
    return {
        "candidate": candidate,
        "top_match": top_match,
        "avg_match": avg_match,
        "count": len(filtered_results),
        "results": filtered_results,
        "error": ""
    }


def find_best_product_results(original_image: Image.Image, cropped_image: Image.Image):
    candidates = get_combined_candidates(original_image, cropped_image)
    if not candidates:
        return None, [], []

    all_candidate_logs = []
    evaluated = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(evaluate_candidate, candidate): candidate for candidate in candidates}
        for future in as_completed(future_map):
            candidate = future_map[future]
            try:
                result = future.result()
                evaluated.append(result)
                all_candidate_logs.append({
                    "candidate": result["candidate"],
                    "top_match": result["top_match"],
                    "count": result["count"],
                    "error": result["error"]
                })
            except Exception as e:
                all_candidate_logs.append({
                    "candidate": candidate,
                    "top_match": 0,
                    "count": 0,
                    "error": str(e)
                })

    if not evaluated:
        return candidates[0], [], all_candidate_logs

    evaluated.sort(key=lambda x: (x["top_match"], x["avg_match"], x["count"]), reverse=True)
    best = evaluated[0]
    best_results = sorted(best["results"], key=lambda x: x["price_num"])
    return best["candidate"], best_results, all_candidate_logs


def show_result_card(i, item):
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
            st.link_button(f"구매 페이지 이동 {i}", item["link"], use_container_width=True)
        st.divider()


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

if query_image is not None:
    query_image = preprocess_image(query_image)
    cropped_image = center_crop_image(query_image, 0.7)

    st.image(query_image, caption="원본 이미지", use_container_width=True)

    with st.expander("검색에 사용할 중심 영역 보기"):
        st.image(cropped_image, caption="중앙 crop 이미지", use_container_width=True)

    if st.button("같은 제품 찾기", use_container_width=True):
        with st.status("제품 탐지 및 쇼핑 검색 중...", expanded=True) as status:
            try:
                status.write("1) 원본/중앙 영역 이미지를 병렬 업로드 중...")
                status.write("2) Lens 후보를 병렬 탐지 중...")
                status.write("3) 후보별 쇼핑 검색을 병렬 평가 중...")

                best_title, shopping_results, candidate_logs = find_best_product_results(query_image, cropped_image)

                if not best_title:
                    status.update(label="제품명을 찾지 못했습니다.", state="error")
                    st.warning("제품 후보를 찾지 못했습니다.")
                elif not shopping_results:
                    status.update(label="같은 제품에 가까운 쇼핑 결과를 찾지 못했습니다.", state="error")
                    st.warning("가까운 쇼핑 결과가 없습니다.")
                    st.info(f"탐지된 대표 후보: {best_title}")
                else:
                    status.write(f"선택된 대표 제품명: **{best_title}**")
                    status.update(label="검색 완료", state="complete")

                    st.subheader("탐지된 제품")
                    st.info(f"이 사진은 **{best_title}** 와(과) 가장 가깝습니다.")

                    st.subheader("🛒 가장 똑같은 쇼핑 결과")
                    for i, item in enumerate(shopping_results[:MAX_RESULTS_TO_SHOW], start=1):
                        show_result_card(i, item)

                    with st.expander("탐지 후보 비교 보기"):
                        sorted_logs = sorted(candidate_logs, key=lambda x: (x["top_match"], x["count"]), reverse=True)
                        for log in sorted_logs:
                            msg = f"- {log['candidate']} | 최고 일치도: {log['top_match']} | 결과 수: {log['count']}"
                            if log["error"]:
                                msg += f" | 오류: {log['error']}"
                            st.write(msg)

            except Exception as e:
                status.update(label="오류 발생", state="error")
                st.error(f"실행 중 오류: {e}")
else:
    st.info("사진을 찍거나 업로드한 뒤 버튼을 누르세요.")
