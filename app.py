import os
import re
import base64
from io import BytesIO

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
# 유틸
# -----------------------------
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


def upload_to_imgbb(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = requests.post(
        "https://api.imgbb.com/1/upload",
        data={
            "key": IMGBB_API_KEY,
            "image": img_b64,
            "name": "photo"
        },
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(f"ImgBB 업로드 실패: {response.status_code} / {response.text}")

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(f"ImgBB 업로드 실패: {data}")

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
    """
    제품명처럼 구체적일수록 높은 점수
    """
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

    generic_words = {
        "스타벅스", "삼성", "애플", "나이키", "텀블러", "컵",
        "신발", "가방", "상품", "물병", "마우스", "키보드"
    }
    if title in generic_words:
        score -= 30

    return score


def normalize_words(text: str):
    text = text.lower()
    words = re.findall(r"[a-zA-Z가-힣0-9]+", text)
    stopwords = {
        "the", "and", "for", "with", "new", "best", "official",
        "상품", "정품", "국내", "해외", "무료", "배송", "판매", "구매",
        "쇼핑", "스토어", "브랜드"
    }
    return [w for w in words if len(w) >= 2 and w not in stopwords]


# -----------------------------
# 1단계: 제품 탐지(넓게)
# -----------------------------
def get_lens_product_candidates(image_url: str, max_candidates=6):
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_KEY,
        "hl": "ko",
        "country": "kr",
        "no_cache": "true",
        "output": "json"
    }

    response = requests.get(
        "https://serpapi.com/search.json",
        params=params,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Lens 요청 실패: {response.status_code} / {response.text}")

    data = response.json()
    candidates = []

    # products 먼저
    for item in data.get("products", []):
        title = clean_product_title(item.get("title", ""))
        if title:
            candidates.append(title)

    # visual_matches도 같이 사용
    for item in data.get("visual_matches", []):
        title = clean_product_title(item.get("title", ""))
        if title:
            candidates.append(title)

    # 중복 제거
    unique = []
    for title in candidates:
        if title not in unique:
            unique.append(title)

    unique.sort(key=score_specificity, reverse=True)
    return unique[:max_candidates]


def get_combined_candidates(original_image: Image.Image, cropped_image: Image.Image):
    original_url = upload_to_imgbb(original_image)
    cropped_url = upload_to_imgbb(cropped_image)

    original_candidates = get_lens_product_candidates(original_url, max_candidates=6)
    cropped_candidates = get_lens_product_candidates(cropped_url, max_candidates=6)

    merged = []
    for title in original_candidates + cropped_candidates:
        if title not in merged:
            merged.append(title)

    merged.sort(key=score_specificity, reverse=True)
    return merged[:8]


# -----------------------------
# 2단계: 쇼핑 결과 가져오기
# -----------------------------
def search_google_shopping(query: str):
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "ko",
        "gl": "kr",
        "no_cache": "true"
    }

    response = requests.get(
        "https://serpapi.com/search.json",
        params=params,
        timeout=60
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

        # 가격 있는 쇼핑 결과만
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
# 3단계: 후보와 결과의 유사도 측정
# -----------------------------
def relevance_score(target_title: str, candidate_title: str) -> int:
    target_words = set(normalize_words(target_title))
    candidate_words = set(normalize_words(candidate_title))

    if not target_words or not candidate_words:
        return 0

    common = target_words & candidate_words
    score = len(common) * 10

    # 숫자/모델명/용량 비슷하면 추가점수
    target_nums = set(re.findall(r"[a-zA-Z]*\d+[a-zA-Z]*", target_title.lower()))
    cand_nums = set(re.findall(r"[a-zA-Z]*\d+[a-zA-Z]*", candidate_title.lower()))
    if target_nums and cand_nums:
        score += len(target_nums & cand_nums) * 15

    return score


def filter_relevant_results(target_title: str, results: list):
    scored = []

    for item in results:
        score = relevance_score(target_title, item["title"])
        new_item = item.copy()
        new_item["match_score"] = score
        scored.append(new_item)

    # 너무 관련 없는 결과 제거
    filtered = [x for x in scored if x["match_score"] >= 10]

    # 하나도 없으면 상위 5개라도 유지
    if not filtered:
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        filtered = scored[:5]

    # 가장 똑같은 제품 우선, 그 다음 가격
    filtered.sort(key=lambda x: (-x["match_score"], x["price_num"]))
    return filtered


# -----------------------------
# 4단계: 후보 여러 개 중 "가장 똑같은 제품" 고르기
# -----------------------------
def find_best_product_results(original_image: Image.Image, cropped_image: Image.Image):
    candidates = get_combined_candidates(original_image, cropped_image)

    if not candidates:
        return None, [], []

    best_title = None
    best_results = []
    best_top_match = -1
    best_average = -1

    all_candidate_logs = []

    for candidate in candidates:
        try:
            shopping_results = search_google_shopping(candidate)
            filtered_results = filter_relevant_results(candidate, shopping_results)

            if not filtered_results:
                all_candidate_logs.append({
                    "candidate": candidate,
                    "top_match": 0,
                    "count": 0
                })
                continue

            top_match = filtered_results[0]["match_score"]
            avg_match = sum(x["match_score"] for x in filtered_results[:3]) / min(len(filtered_results), 3)

            all_candidate_logs.append({
                "candidate": candidate,
                "top_match": top_match,
                "count": len(filtered_results)
            })

            # 가장 똑같은 것 우선
            if (top_match > best_top_match) or (top_match == best_top_match and avg_match > best_average):
                best_top_match = top_match
                best_average = avg_match
                best_title = candidate
                best_results = filtered_results

        except Exception:
            all_candidate_logs.append({
                "candidate": candidate,
                "top_match": 0,
                "count": 0
            })
            continue

    if not best_title:
        return candidates[0], [], all_candidate_logs

    # 최종 출력은 최저가순
    best_results.sort(key=lambda x: x["price_num"])
    return best_title, best_results, all_candidate_logs


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
    st.error("SERPAPI_KEY와 IMGBB_API_KEY를 코드에 넣어주세요.")
    st.stop()

# -----------------------------
# 실행
# -----------------------------
if query_image is not None:
    query_image = preprocess_image(query_image)
    cropped_image = center_crop_image(query_image, 0.7)

    st.image(query_image, caption="원본 이미지", use_container_width=True)

    with st.expander("검색에 사용할 중심 영역 보기"):
        st.image(cropped_image, caption="중앙 crop 이미지", use_container_width=True)

    if st.button("같은 제품 찾기", use_container_width=True):
        with st.status("제품 탐지 및 쇼핑 검색 중...", expanded=True) as status:
            try:
                status.write("1) 원본/중앙 영역 이미지 업로드 중...")
                status.write("2) Lens로 제품 후보를 넓게 탐지 중...")
                status.write("3) 각 후보를 쇼핑 검색 후 가장 똑같은 결과를 찾는 중...")

                best_title, shopping_results, candidate_logs = find_best_product_results(
                    query_image,
                    cropped_image
                )

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

                    with st.expander("탐지 후보 비교 보기"):
                        for log in candidate_logs:
                            st.write(
                                f"- {log['candidate']} | 최고 일치도: {log['top_match']} | 결과 수: {log['count']}"
                            )

            except Exception as e:
                status.update(label="오류 발생", state="error")
                st.error(f"실행 중 오류: {e}")

else:
    st.info("사진을 찍거나 업로드한 뒤 버튼을 누르세요.")
