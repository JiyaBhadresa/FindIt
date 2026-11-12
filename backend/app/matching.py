import io
import re
import math
import asyncio
import urllib.request
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# ============================================================================
# 1. TEXT MATCHING (NLP SBERT)
# ============================================================================

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_item_text(item: dict) -> str:
    parts = []
    title = (item.get("title") or "").strip()
    if title:
        parts.append(title)
    category = (item.get("category") or "").strip()
    if category:
        parts.append(f"Category: {category}")
    brand = (item.get("brand") or "").strip()
    if brand:
        parts.append(f"Brand: {brand}")
    color = (item.get("color") or "").strip()
    if color:
        parts.append(f"Color: {color}")
    desc = (item.get("description") or "").strip()
    if desc:
        parts.append(desc)
    loc = (item.get("location") or "").strip()
    if loc:
        parts.append(f"Location: {loc}")
    return " | ".join(parts) if parts else "Unknown item"


def calculate_matches(
    target_item: dict,
    candidates: List[dict],
    threshold: float = 0.3,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    model = get_model()
    target_text = build_item_text(target_item)
    candidate_texts = [build_item_text(c) for c in candidates]

    target_emb = model.encode(target_text, convert_to_tensor=True)
    candidate_embs = model.encode(candidate_texts, convert_to_tensor=True)

    scores = util.cos_sim(target_emb, candidate_embs)[0].tolist()

    matches = []
    for candidate, score in zip(candidates, scores):
        score_val = round(float(score), 4)
        if score_val >= threshold:
            item_data = dict(candidate)
            item_data["similarity_score"] = score_val
            matches.append(item_data)

    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matches[:top_k]


async def find_matches(
    target_item: dict,
    candidates: List[dict],
    threshold: float = 0.3,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        calculate_matches,
        target_item,
        candidates,
        threshold,
        top_k,
    )


# ============================================================================
# 2. IMAGE MATCHING (CLIP ViT)
# ============================================================================

_image_model: Optional[SentenceTransformer] = None
IMAGE_DOWNLOAD_TIMEOUT = 8.0
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


def get_image_model() -> SentenceTransformer:
    global _image_model
    if _image_model is None:
        _image_model = SentenceTransformer("clip-ViT-B-32")
    return _image_model


def download_image(url: str) -> Optional[Image.Image]:
    if not url or not isinstance(url, str):
        return None
    if not (url.startswith("http://") or url.startswith("https://")):
        return None
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FindIt-AI-ImageMatcher/1.0"},
        )
        with urllib.request.urlopen(req, timeout=IMAGE_DOWNLOAD_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            data = resp.read(MAX_IMAGE_SIZE_BYTES + 1)
            if len(data) > MAX_IMAGE_SIZE_BYTES:
                return None
            img = Image.open(io.BytesIO(data))
            return img.convert("RGB")
    except Exception:
        return None


def calculate_image_matches(
    target_image: Image.Image,
    candidate_pairs: List[Tuple[dict, Image.Image]],
    threshold: float = 0.2,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not candidate_pairs:
        return []

    model = get_image_model()
    candidate_images = [img for _, img in candidate_pairs]
    candidates = [item for item, _ in candidate_pairs]

    target_emb = model.encode(target_image, convert_to_tensor=True)
    candidate_embs = model.encode(candidate_images, convert_to_tensor=True)

    scores = util.cos_sim(target_emb, candidate_embs)[0].tolist()

    matches = []
    for candidate, score in zip(candidates, scores):
        score_val = round(float(score), 4)
        if score_val >= threshold:
            matches.append({
                "id": candidate["id"],
                "title": candidate["title"],
                "image_url": candidate["image_url"],
                "similarity_score": score_val,
            })

    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matches[:top_k]


async def process_image_matching(
    target_image_url: str,
    candidates: List[dict],
    threshold: float = 0.2,
    top_k: int = 5,
) -> Tuple[int, List[Dict[str, Any]]]:
    target_image = await asyncio.to_thread(download_image, target_image_url)
    if target_image is None:
        raise ValueError("Could not download or decode target image.")

    candidate_pairs: List[Tuple[dict, Image.Image]] = []
    for candidate in candidates:
        img_url = candidate.get("image_url")
        if not img_url:
            continue
        cand_img = await asyncio.to_thread(download_image, img_url)
        if cand_img is not None:
            candidate_pairs.append((candidate, cand_img))

    valid_candidates_count = len(candidate_pairs)
    if valid_candidates_count == 0:
        return 0, []

    matches = await asyncio.to_thread(
        calculate_image_matches,
        target_image,
        candidate_pairs,
        threshold,
        top_k,
    )
    return valid_candidates_count, matches


# ============================================================================
# 3. ATTRIBUTE MATCHING (COLOR, CATEGORY, BRAND, FEATURES)
# ============================================================================

COLOR_FAMILIES = {
    "blue": {"blue", "navy", "dark blue", "midnight blue", "royal blue", "light blue", "sky blue", "cyan", "cobalt", "indigo", "teal"},
    "black": {"black", "jet black", "matte black", "pitch black", "obsidian", "dark"},
    "gray": {"gray", "grey", "silver", "space gray", "space grey", "charcoal", "slate", "metallic", "aluminum", "gunmetal"},
    "white": {"white", "pure white", "off white", "ivory", "cream", "snow"},
    "red": {"red", "dark red", "crimson", "scarlet", "ruby", "maroon", "burgundy", "cherry", "rose red"},
    "green": {"green", "dark green", "olive", "emerald", "lime", "mint", "forest green", "sage"},
    "yellow": {"yellow", "gold", "golden", "amber", "mustard", "lemon"},
    "orange": {"orange", "coral", "peach", "rust", "tangerine"},
    "brown": {"brown", "tan", "beige", "khaki", "chocolate", "coffee", "caramel"},
    "purple": {"purple", "violet", "lavender", "plum", "magenta"},
    "pink": {"pink", "rose", "blush", "salmon"},
}

NOUN_SYNONYMS = {
    "mobile": "phone",
    "cell phone": "phone",
    "cellphone": "phone",
    "cellular phone": "phone",
    "cellular device": "phone",
    "smartphone": "phone",
    "iphone": "phone",
    "android": "phone",
    "purse": "wallet_purse",
    "wallet": "wallet_purse",
    "billfold": "wallet_purse",
    "pocketbook": "wallet_purse",
    "handbag": "wallet_purse",
    "clutch": "wallet_purse",
    "laptop": "laptop",
    "notebook": "laptop",
    "macbook": "laptop",
    "chromebook": "laptop",
}


def normalize_text(val: Optional[str]) -> str:
    if not val:
        return ""
    text = val.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_canonical_color(color_str: str) -> Optional[str]:
    norm = normalize_text(color_str)
    if not norm:
        return None
    for canonical, variants in COLOR_FAMILIES.items():
        if norm in variants:
            return canonical
        for v in variants:
            if v in norm or norm in v:
                return canonical
    return norm


def compare_category(t_cat: Optional[str], c_cat: Optional[str]) -> Tuple[float, str, str]:
    t = normalize_text(t_cat)
    c = normalize_text(c_cat)
    if not t or not c or t == "other" or c == "other":
        return 0.5, "neutral", "Category missing or unspecified (neutral)"
    if t == c:
        return 1.0, "match", f"Exact category match ({t_cat})"
    return 0.0, "conflict", f"Category conflict ({t_cat} vs {c_cat})"


def compare_color(t_col: Optional[str], c_col: Optional[str]) -> Tuple[Optional[float], str, str]:
    t = normalize_text(t_col)
    c = normalize_text(c_col)
    if not t or not c:
        return None, "missing", "Color missing on one or both items"

    if t == c:
        return 1.0, "match", f"Exact color match ({t_col})"

    canon_t = get_canonical_color(t)
    canon_c = get_canonical_color(c)
    if canon_t and canon_c and canon_t == canon_c:
        return 0.85, "synonym_match", f"Color shade match ({t_col} vs {c_col})"

    if t in c or c in t:
        return 0.85, "synonym_match", f"Color variant match ({t_col} vs {c_col})"

    return 0.0, "conflict", f"Color conflict ({t_col} vs {c_col})"


def compare_brand(t_brand: Optional[str], c_brand: Optional[str]) -> Tuple[Optional[float], str, str]:
    t = normalize_text(t_brand)
    c = normalize_text(c_brand)
    if not t or not c:
        return None, "missing", "Brand missing on one or both items"

    t = re.sub(r"\b(inc|co|corp|ltd|llc)\b", "", t).strip()
    c = re.sub(r"\b(inc|co|corp|ltd|llc)\b", "", c).strip()

    if t == c:
        return 1.0, "match", f"Exact brand match ({t_brand})"
    if t in c or c in t:
        return 0.95, "close_match", f"Close brand match ({t_brand} vs {c_brand})"

    return 0.0, "conflict", f"Brand conflict ({t_brand} vs {c_brand})"


def compare_features(target: dict, candidate: dict) -> Tuple[float, str, str]:
    t_desc = (target.get("description") or "").strip()
    c_desc = (candidate.get("description") or "").strip()
    if not t_desc or not c_desc:
        return 0.5, "neutral", "Insufficient description to compare distinctive features"

    model = get_model()
    t_emb = model.encode(t_desc, convert_to_tensor=True)
    c_emb = model.encode(c_desc, convert_to_tensor=True)
    score = round(max(0.0, float(util.cos_sim(t_emb, c_emb)[0][0])), 4)

    if score >= 0.70:
        return score, "strong_match", f"Highly similar distinctive features (score: {score})"
    elif score >= 0.40:
        return score, "moderate_match", f"Moderately similar features (score: {score})"
    else:
        return score, "low_match", f"Distinctive features differ (score: {score})"


def evaluate_attributes(target: dict, candidate: dict) -> Tuple[float, Dict[str, Any], str]:
    cat_score, cat_status, cat_note = compare_category(target.get("category"), candidate.get("category"))
    col_score, col_status, col_note = compare_color(target.get("color"), candidate.get("color"))
    brand_score, brand_status, brand_note = compare_brand(target.get("brand"), candidate.get("brand"))
    feat_score, feat_status, feat_note = compare_features(target, candidate)

    breakdown = {
        "category": {"target": target.get("category"), "candidate": candidate.get("category"), "score": cat_score, "status": cat_status, "note": cat_note},
        "color": {"target": target.get("color"), "candidate": candidate.get("color"), "score": col_score, "status": col_status, "note": col_note},
        "brand": {"target": target.get("brand"), "candidate": candidate.get("brand"), "score": brand_score, "status": brand_status, "note": brand_note},
        "distinctive_features": {"score": feat_score, "status": feat_status, "note": feat_note},
    }

    weights = []
    scores = []

    weights.append(0.35)
    scores.append(cat_score)

    weights.append(0.25)
    scores.append(feat_score)

    if brand_score is not None:
        weights.append(0.25)
        scores.append(brand_score)

    if col_score is not None:
        weights.append(0.15)
        scores.append(col_score)

    total_weight = sum(weights)
    weighted_sum = sum(w * s for w, s in zip(weights, scores))
    raw_score = weighted_sum / total_weight if total_weight > 0 else 0.5

    multiplier = 1.0
    notes = []

    if cat_status == "conflict":
        multiplier *= 0.1
        notes.append("conflicting category")
    elif cat_status == "match":
        notes.append("matching category")

    if brand_status == "conflict":
        multiplier *= 0.4
        notes.append("conflicting brand")
    elif brand_status in ("match", "close_match"):
        notes.append("matching brand")

    if col_status == "conflict":
        multiplier *= 0.6
        notes.append("conflicting color")
    elif col_status in ("match", "synonym_match"):
        notes.append("compatible color")

    final_score = round(max(0.0, min(1.0, raw_score * multiplier)), 4)
    explanation = "; ".join(notes) if notes else "Attributes evaluated."
    if multiplier < 1.0:
        explanation += " (score reduced due to conflicts)"

    return final_score, breakdown, explanation


def calculate_attribute_matches(
    target_item: dict,
    candidates: List[dict],
    threshold: float = 0.3,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    matches = []
    for candidate in candidates:
        score, breakdown, explanation = evaluate_attributes(target_item, candidate)
        if score >= threshold:
            matches.append({
                "id": candidate["id"],
                "item_type": candidate.get("item_type", ""),
                "title": candidate.get("title", ""),
                "category": candidate.get("category", ""),
                "color": candidate.get("color"),
                "brand": candidate.get("brand"),
                "attribute_score": score,
                "breakdown": breakdown,
                "explanation": explanation,
            })

    matches.sort(key=lambda x: x["attribute_score"], reverse=True)
    return matches[:top_k]


async def find_attribute_matches(
    target_item: dict,
    candidates: List[dict],
    threshold: float = 0.3,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        calculate_attribute_matches,
        target_item,
        candidates,
        threshold,
        top_k,
    )


# ============================================================================
# 4. LOCATION & TIME MATCHING
# ============================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 3)


def normalize_location_text(loc: Optional[str]) -> List[str]:
    if not loc:
        return []
    cleaned = re.sub(r"[^\w\s]", " ", loc.lower()).strip()
    stopwords = {"in", "at", "the", "near", "by", "on", "of", "and", "a", "an", "floor", "building", "room"}
    tokens = [t for t in cleaned.split() if t not in stopwords and len(t) > 1]
    return tokens


def evaluate_location(target: dict, candidate: dict) -> Tuple[float, Optional[float], str]:
    t_lat = target.get("latitude")
    t_lon = target.get("longitude")
    c_lat = candidate.get("latitude")
    c_lon = candidate.get("longitude")

    if (
        t_lat is not None
        and t_lon is not None
        and c_lat is not None
        and c_lon is not None
    ):
        dist_km = haversine_km(float(t_lat), float(t_lon), float(c_lat), float(c_lon))
        if dist_km <= 0.1:
            score = 1.0
            note = f"Exact coordinates match (within {int(dist_km * 1000)}m)"
        elif dist_km <= 0.5:
            score = 0.90
            note = f"Very close proximity ({int(dist_km * 1000)}m apart)"
        elif dist_km <= 1.5:
            score = 0.75
            note = f"Nearby locations ({dist_km} km apart)"
        elif dist_km <= 5.0:
            score = 0.50
            note = f"Moderate distance ({dist_km} km apart)"
        elif dist_km <= 15.0:
            score = 0.25
            note = f"Distanced locations ({dist_km} km apart)"
        else:
            score = 0.0
            note = f"Far apart ({dist_km} km apart)"
        return score, dist_km, note

    t_loc = (target.get("location") or "").strip()
    c_loc = (candidate.get("location") or "").strip()

    if not t_loc or not c_loc:
        return 0.5, None, "Location not specified on one or both items (neutral)"

    if t_loc.lower() == c_loc.lower():
        return 1.0, None, f"Identical location description ('{t_loc}')"

    t_tokens = set(normalize_location_text(t_loc))
    c_tokens = set(normalize_location_text(c_loc))

    if not t_tokens or not c_tokens:
        return 0.5, None, "Insufficient location text for detailed comparison"

    intersection = t_tokens & c_tokens
    union = t_tokens | c_tokens
    jaccard = len(intersection) / len(union) if union else 0.0

    if t_loc.lower() in c_loc.lower() or c_loc.lower() in t_loc.lower():
        jaccard = max(jaccard, 0.85)

    if jaccard >= 0.70:
        return round(jaccard, 4), None, f"High location wording overlap ({', '.join(intersection)})"
    elif jaccard >= 0.30:
        return round(jaccard, 4), None, f"Partial location overlap ({', '.join(intersection)})"
    else:
        return round(jaccard, 4), None, f"Different locations ('{t_loc}' vs '{c_loc}')"


def parse_date_safe(d_str: Optional[str]) -> Optional[date]:
    if not d_str or not isinstance(d_str, str):
        return None
    d_clean = d_str.strip()[:10]
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(d_clean, fmt).date()
        except ValueError:
            pass
    return None


def evaluate_time(target: dict, candidate: dict, max_days: int = 30) -> Tuple[float, Optional[float], str]:
    t_date = parse_date_safe(target.get("date_lost_or_found"))
    c_date = parse_date_safe(candidate.get("date_lost_or_found"))

    if not t_date or not c_date:
        return 0.0, None, "Missing or unparseable date"

    diff_days = abs((t_date - c_date).days)

    if diff_days > max_days:
        return 0.0, float(diff_days), f"Reported {diff_days} days apart (exceeds max window of {max_days} days)"

    if diff_days == 0:
        score = 1.0
        note = "Reported on the exact same date"
    elif diff_days <= 1:
        score = 0.95
        note = "Reported 1 day apart"
    elif diff_days <= 3:
        score = 0.85
        note = f"Reported {diff_days} days apart (very close)"
    elif diff_days <= 7:
        score = 0.70
        note = f"Reported {diff_days} days apart (within 1 week)"
    elif diff_days <= 14:
        score = 0.50
        note = f"Reported {diff_days} days apart (within 2 weeks)"
    else:
        score = round(max(0.05, 0.50 - (0.45 * (diff_days - 14) / max(1, max_days - 14))), 4)
        note = f"Reported {diff_days} days apart"

    return score, float(diff_days), note


def evaluate_location_time(target: dict, candidate: dict, max_days: int = 30) -> Dict[str, Any]:
    loc_score, dist_km, loc_note = evaluate_location(target, candidate)
    time_score, diff_days, time_note = evaluate_time(target, candidate, max_days=max_days)

    combined_score = round(0.5 * loc_score + 0.5 * time_score, 4)
    explanation = f"{loc_note}. {time_note}."

    return {
        "location_score": loc_score,
        "time_score": time_score,
        "combined_score": combined_score,
        "distance_km": dist_km,
        "time_difference_days": diff_days,
        "explanation": explanation,
    }


def calculate_location_time_matches(
    target_item: dict,
    candidates: List[dict],
    threshold: float = 0.2,
    max_days: int = 30,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    matches = []
    for candidate in candidates:
        res = evaluate_location_time(target_item, candidate, max_days=max_days)
        if res["combined_score"] >= threshold:
            matches.append({
                "id": candidate["id"],
                "item_type": candidate.get("item_type", ""),
                "title": candidate.get("title", ""),
                "location": candidate.get("location", ""),
                "date_lost_or_found": candidate.get("date_lost_or_found", ""),
                "location_score": res["location_score"],
                "time_score": res["time_score"],
                "combined_score": res["combined_score"],
                "distance_km": res["distance_km"],
                "time_difference_days": res["time_difference_days"],
                "explanation": res["explanation"],
            })

    matches.sort(key=lambda x: x["combined_score"], reverse=True)
    return matches[:top_k]


async def find_location_time_matches(
    target_item: dict,
    candidates: List[dict],
    threshold: float = 0.2,
    max_days: int = 30,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        calculate_location_time_matches,
        target_item,
        candidates,
        threshold,
        max_days,
        top_k,
    )


# ============================================================================
# 5. UNIFIED HYBRID MATCHING ENGINE
# ============================================================================

DEFAULT_WEIGHTS = {
    "text": 0.25,
    "image": 0.35,
    "attribute": 0.20,
    "location_time": 0.20,
}


def compute_text_scores(target_item: dict, candidates: List[dict]) -> List[float]:
    if not candidates:
        return []
    model = get_model()
    target_text = build_item_text(target_item)
    cand_texts = [build_item_text(c) for c in candidates]

    t_emb = model.encode(target_text, convert_to_tensor=True)
    c_embs = model.encode(cand_texts, convert_to_tensor=True)

    scores = util.cos_sim(t_emb, c_embs)[0].tolist()
    return [round(max(0.0, float(s)), 4) for s in scores]


def compute_image_scores(
    target_item: dict, candidates: List[dict]
) -> List[Tuple[Optional[float], Optional[str]]]:
    t_url = target_item.get("image_url")
    if not t_url:
        return [(None, "no target image")] * len(candidates)

    t_img = download_image(t_url)
    if t_img is None:
        return [(None, "target image download failed")] * len(candidates)

    clip_model = get_image_model()
    t_emb = clip_model.encode(t_img, convert_to_tensor=True)

    results = []
    for cand in candidates:
        c_url = cand.get("image_url")
        if not c_url:
            results.append((None, "candidate has no image"))
            continue

        c_img = download_image(c_url)
        if c_img is None:
            results.append((None, "candidate image download failed"))
            continue

        try:
            c_emb = clip_model.encode(c_img, convert_to_tensor=True)
            sim = float(util.cos_sim(t_emb, c_emb)[0][0])
            score_val = round(max(0.0, min(1.0, sim)), 4)
            results.append((score_val, None))
        except Exception:
            results.append((None, "image embedding calculation failed"))

    return results


def build_hybrid_explanation(
    hybrid_score: float,
    text_score: Optional[float],
    image_score: Optional[float],
    attr_score: Optional[float],
    loc_score: Optional[float],
    time_score: Optional[float],
    signals_used: List[str],
    signals_unavailable: List[str],
    attr_breakdown: dict,
    conflict_multiplier: float,
) -> str:
    parts = []

    if hybrid_score >= 0.80:
        parts.append("Strong potential match:")
    elif hybrid_score >= 0.55:
        parts.append("Moderate potential match:")
    elif hybrid_score >= 0.35:
        parts.append("Low to moderate potential match:")
    else:
        parts.append("Weak potential match:")

    descriptors = []
    if text_score is not None and text_score >= 0.60:
        descriptors.append(f"highly similar text descriptions ({text_score})")
    elif text_score is not None and text_score >= 0.40:
        descriptors.append(f"moderately related descriptions ({text_score})")

    if image_score is not None:
        if image_score >= 0.80:
            descriptors.append(f"strong visual image agreement ({image_score})")
        elif image_score >= 0.50:
            descriptors.append(f"moderate visual similarity ({image_score})")
        else:
            descriptors.append(f"low visual resemblance ({image_score})")

    if attr_score is not None:
        cat_status = attr_breakdown.get("category", {}).get("status")
        brand_status = attr_breakdown.get("brand", {}).get("status")
        col_status = attr_breakdown.get("color", {}).get("status")
        if cat_status == "match" and brand_status in ("match", "close_match"):
            descriptors.append("matching category and brand")
        elif cat_status == "match":
            descriptors.append("matching category")

        if col_status in ("match", "synonym_match"):
            descriptors.append("compatible color")

    if loc_score is not None and loc_score >= 0.80:
        descriptors.append("reported at or near the same location")
    elif loc_score is not None and loc_score < 0.30:
        descriptors.append("different reported locations")

    if time_score is not None and time_score >= 0.85:
        descriptors.append("reported within 1 to 3 days")
    elif time_score is not None and time_score <= 0.20:
        descriptors.append("significant time gap between reports")

    if descriptors:
        parts.append(", ".join(descriptors) + ".")

    if conflict_multiplier < 1.0:
        cat_note = attr_breakdown.get("category", {}).get("note", "")
        if "conflict" in cat_note.lower():
            parts.append(f"Score heavily reduced due to category conflict ({cat_note}).")
        else:
            parts.append("Score reduced due to conflicting attributes.")

    if signals_unavailable:
        unavail_str = "; ".join(signals_unavailable)
        parts.append(f"Note: {unavail_str}.")

    return " ".join(parts)


def calculate_hybrid_matches(
    target_item: dict,
    candidates: List[dict],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.35,
    max_days: int = 30,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    active_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        active_weights.update(weights)

    text_scores = compute_text_scores(target_item, candidates)
    image_results = compute_image_scores(target_item, candidates)

    matches = []
    for idx, candidate in enumerate(candidates):
        t_score = text_scores[idx] if idx < len(text_scores) else 0.5
        img_score, img_unavail = image_results[idx] if idx < len(image_results) else (None, "image unavailable")

        attr_score, attr_breakdown, _ = evaluate_attributes(target_item, candidate)
        loc_time_res = evaluate_location_time(target_item, candidate, max_days=max_days)
        loc_score = loc_time_res["location_score"]
        time_score = loc_time_res["time_score"]
        loc_time_combined = loc_time_res["combined_score"]

        available_signals = {}
        signals_used = []
        signals_unavailable = []

        available_signals["text"] = (active_weights["text"], t_score)
        signals_used.append("text")

        if img_score is not None:
            available_signals["image"] = (active_weights["image"], img_score)
            signals_used.append("image")
        else:
            signals_unavailable.append(f"image ({img_unavail})")

        available_signals["attribute"] = (active_weights["attribute"], attr_score)
        signals_used.append("attribute")

        available_signals["location_time"] = (active_weights["location_time"], loc_time_combined)
        signals_used.append("location_time")

        total_available_weight = sum(w for w, _ in available_signals.values())
        if total_available_weight > 0:
            raw_hybrid_score = sum((w / total_available_weight) * s for w, s in available_signals.values())
        else:
            raw_hybrid_score = 0.0

        conflict_multiplier = 1.0
        cat_status = attr_breakdown.get("category", {}).get("status")
        brand_status = attr_breakdown.get("brand", {}).get("status")
        col_status = attr_breakdown.get("color", {}).get("status")

        if cat_status == "conflict":
            conflict_multiplier *= 0.15
        if brand_status == "conflict":
            conflict_multiplier *= 0.70
        if col_status == "conflict":
            conflict_multiplier *= 0.85

        final_hybrid_score = round(max(0.0, min(1.0, raw_hybrid_score * conflict_multiplier)), 4)

        explanation = build_hybrid_explanation(
            hybrid_score=final_hybrid_score,
            text_score=t_score,
            image_score=img_score,
            attr_score=attr_score,
            loc_score=loc_score,
            time_score=time_score,
            signals_used=signals_used,
            signals_unavailable=signals_unavailable,
            attr_breakdown=attr_breakdown,
            conflict_multiplier=conflict_multiplier,
        )

        if final_hybrid_score >= threshold:
            matches.append({
                "id": candidate["id"],
                "item_type": candidate.get("item_type", ""),
                "title": candidate.get("title", ""),
                "image_url": candidate.get("image_url"),
                "hybrid_score": final_hybrid_score,
                "text_score": t_score,
                "image_score": img_score,
                "attribute_score": attr_score,
                "location_score": loc_score,
                "time_score": time_score,
                "signals_used": signals_used,
                "signals_unavailable": signals_unavailable,
                "explanation": explanation,
            })

    matches.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return matches[:top_k]


async def find_hybrid_matches(
    target_item: dict,
    candidates: List[dict],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.35,
    max_days: int = 30,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        calculate_hybrid_matches,
        target_item,
        candidates,
        weights,
        threshold,
        max_days,
        top_k,
    )
