"""
LangChain 工具模块。

工具 a: 知识库参考
工具 c1: 搜索引擎结果列表
工具 c2: 多网页提取并按搜索词筛片段
工具 c3: 单网页全文读取
工具 d: 信源信誉度查询
save_result: 仅负责写入数据库与向量索引
"""

import json
import re
from urllib.parse import urlparse
from typing import Any

from bs4 import BeautifulSoup
import requests
from tavily import TavilyClient

from app.config import TAVILY_API_KEY
from app.database import get_all_news, insert_news, search_news
from app.source_credibility import get_source_credibility
from app.vector_store import search_similar_news, upsert_vector_entry


tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def _parse_json(text: str) -> dict:
    """从 LLM 回复中提取 JSON（兼容 markdown 代码块）。"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def _clip(text: Any, limit: int = 5000) -> str:
    value = "" if text is None else str(text)
    return value[:limit]


def _safe_float(value: Any, default: float = 5.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip())


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered


# ════════════════════════════════════════════════════
#  知识库相似度匹配
# ════════════════════════════════════════════════════

_SIMILARITY_STOP_TOKENS = {
    "目前", "现在", "仅剩", "仅有", "只有", "可正常", "正常", "使用", "消息", "新闻", "信息",
    "报道", "表示", "称", "指出", "已经", "正在", "相关", "情况", "问题", "进行", "因为",
    "是否", "这个", "那个", "我们", "你们", "他们", "并且", "如果", "可以", "需要", "一个",
    "一些", "一种", "其中", "通过", "关于", "对于", "以及", "没有", "无法", "仍然", "继续",
}


def _tokenize_for_similarity(text: str) -> set[str]:
    normalized = _normalize_query(text).lower()
    latin_tokens = re.findall(r"[a-z0-9]{2,}", normalized)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    cjk_ngrams = []
    for size in (2, 3):
        cjk_ngrams.extend("".join(cjk_chars[index:index + size]) for index in range(len(cjk_chars) - size + 1))

    normalized_tokens = set()
    for token in latin_tokens + cjk_ngrams:
        if token in _SIMILARITY_STOP_TOKENS:
            continue
        if token.isdigit():
            continue
        normalized_tokens.add(token)
    return normalized_tokens


def _extract_key_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _similarity_score(query_text: str, candidate_text: str) -> float:
    query_tokens = _tokenize_for_similarity(query_text)
    candidate_tokens = _tokenize_for_similarity(candidate_text)
    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = query_tokens & candidate_tokens
    union = query_tokens | candidate_tokens
    meaningful_overlap = [token for token in overlap if len(token) >= 2]
    if not meaningful_overlap:
        return 0.0
    jaccard = len(overlap) / len(union) if union else 0.0

    query_numbers = _extract_key_numbers(query_text)
    candidate_numbers = _extract_key_numbers(candidate_text)
    number_score = 0.0
    if query_numbers:
        number_score = len(query_numbers & candidate_numbers) / len(query_numbers)

    contains_bonus = 0.15 if _normalize_query(query_text) in _normalize_query(candidate_text) else 0.0
    overlap_bonus = min(len(meaningful_overlap), 4) * 0.05
    return jaccard * 0.75 + number_score * 0.05 + overlap_bonus + contains_bonus


def _select_reference_candidates(news_text: str, all_news: list[dict], limit: int = 20) -> list[dict]:
    scored_candidates = []
    for item in all_news:
        candidate_text = item.get("news_text", "")
        score = _similarity_score(news_text, candidate_text)
        if score < 0.12:
            continue
        scored_candidates.append(
            {
                "id": item.get("id"),
                "score": round(score, 4),
                "news_text": candidate_text,
                "is_real": item.get("is_real"),
                "reason": item.get("reason", ""),
                "evidence_url": item.get("evidence_url", ""),
            }
        )

    scored_candidates.sort(key=lambda item: item["score"], reverse=True)
    return scored_candidates[:limit]


def _normalize_bool_label(value: Any) -> str:
    if value is None:
        return "未知"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric >= 7:
        return "较可信"
    if numeric <= 3:
        return "较不可信"
    return "待定"


def _extract_reference_focuses(news_text: str, candidates: list[dict]) -> list[str]:
    focuses: list[str] = []
    has_numbers = bool(_extract_key_numbers(news_text))
    if has_numbers:
        focuses.append("核实关键数字和数量口径")

    lowered_text = _normalize_query(news_text)
    if any(token in lowered_text for token in ["称", "表示", "消息", "曝", "据说", "网传"]):
        focuses.append("确认原始发布者和首发渠道")

    reasons = " ".join(_normalize_query(item.get("reason", "")) for item in candidates[:5])
    if any(token in reasons for token in ["标题", "夸大", "断章取义", "误导", "宣传", "战果"]):
        focuses.append("检查是否存在夸大、误导或宣传性表述")

    if any(item.get("evidence_url") for item in candidates[:5]):
        focuses.append("优先复核历史记录引用过的证据来源")

    focuses.append("核实核心事实是否有独立来源交叉支持")

    deduped: list[str] = []
    for item in focuses:
        if item not in deduped:
            deduped.append(item)
    return deduped[:4]


def _build_reference_summary(news_text: str, candidates: list[dict]) -> tuple[str, list[str], int | None, bool, bool]:
    if not candidates:
        return (
            "知识库中没有找到足够相似的历史记录，本次不提供参考样本。",
            ["确认原始发布者", "核实核心事实", "补足独立来源"],
            None,
            False,
            False,
        )

    top_candidates = candidates[:3]
    top_candidate = top_candidates[0]
    top_score = top_candidate.get("score", 0.0)
    focuses = _extract_reference_focuses(news_text, top_candidates)

    summary_parts = [
        f"知识库已召回 {len(candidates)} 条候选，最高相似度 {top_score:.4f}。",
        "建议优先参考最相近的历史记录，重点复核其相似点是否真的落在同一主体、同一事件和同一数字口径上。",
    ]

    verdict_labels = []
    for item in top_candidates:
        verdict_labels.append(
            f"ID {item.get('id')} 为{_normalize_bool_label(item.get('is_real'))}，相似度 {item.get('score', 0.0):.4f}"
        )
    if verdict_labels:
        summary_parts.append("候选概况: " + "；".join(verdict_labels) + "。")

    can_determine = False
    matched_id = None
    found = True
    if top_score >= 0.72:
        numbers_match = bool(_extract_key_numbers(news_text) & _extract_key_numbers(top_candidate.get("news_text", "")))
        if numbers_match or not _extract_key_numbers(news_text):
            can_determine = True
            matched_id = top_candidate.get("id")
            summary_parts.append("最高相似候选与当前输入高度接近，可作为近似同条声明的直接参考。")
    elif top_score < 0.2:
        found = False
        summary_parts = ["候选历史记录虽被召回，但整体相似度偏低，已忽略知识库参考以避免误导。"]

    return ("".join(summary_parts), focuses, matched_id, found, can_determine)


# ════════════════════════════════════════════════════
#  网页抓取辅助
# ════════════════════════════════════════════════════

def _domain_from_url(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _domain_distribution(results: list[dict]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for item in results:
        domain = item.get("domain") or _domain_from_url(item.get("url", ""))
        if not domain:
            continue
        distribution[domain] = distribution.get(domain, 0) + 1
    return distribution


def _extract_number_mentions(text: str, limit: int = 8) -> list[str]:
    mentions: list[str] = []
    for value in re.findall(r"\d+(?:\.\d+)?%?", text or ""):
        if value not in mentions:
            mentions.append(value)
        if len(mentions) >= limit:
            break
    return mentions


def _fetch_page_with_requests(url: str) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        )
        response.raise_for_status()
    except Exception as exc:
        return {"url": url, "title": "", "raw_content": "", "content_length": 0, "error": str(exc)}

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    text_parts = [segment.strip() for segment in soup.stripped_strings]
    raw_content = "\n".join(part for part in text_parts if part)
    return {
        "url": response.url or url,
        "title": _clip(title, 300),
        "raw_content": raw_content,
        "content_length": len(raw_content),
        "fetch_source": "requests_bs4",
    }


# ════════════════════════════════════════════════════
#  文本匹配辅助
# ════════════════════════════════════════════════════

def _normalize_match_term(term: str) -> str:
    return re.sub(r"\s+", " ", (term or "").strip().lower())


def _split_text_blocks(text: str) -> list[str]:
    if not text:
        return []
    chunks = re.split(r"(?:\n{2,}|(?<=[。！？!?])|(?<=[.;])\s+)", text)
    blocks: list[str] = []
    for chunk in chunks:
        cleaned = re.sub(r"\s+", " ", chunk).strip()
        if len(cleaned) < 20:
            continue
        blocks.append(cleaned)
    return blocks


def _text_matches_term(text: str, term: str) -> bool:
    normalized_text = _normalize_match_term(text)
    normalized_term = _normalize_match_term(term)
    if not normalized_text or not normalized_term:
        return False
    if len(normalized_term) <= 2:
        return normalized_term in normalized_text
    if re.search(r"[\u4e00-\u9fff]", normalized_term):
        return normalized_term in normalized_text
    return bool(re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text))


def _find_matching_segments(raw_text: str, search_terms: list[str], limit: int = 8) -> list[str]:
    if not raw_text:
        return []
    valid_terms = [term for term in search_terms if _normalize_match_term(term)]
    if not valid_terms:
        return []

    matched: list[str] = []
    for block in _split_text_blocks(raw_text):
        if any(_text_matches_term(block, term) for term in valid_terms):
            if block not in matched:
                matched.append(_clip(block, 500))
        if len(matched) >= limit:
            break
    return matched


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not normalized_text:
        return []

    sentences: list[str] = []
    for chunk in re.findall(r"[^。！？!?\.]+[。！？!?\.]?", normalized_text):
        cleaned = chunk.strip()
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _normalize_keyword_pairs(
    keyword_pairs: list[list[str]] | None = None,
    search_terms: list[str] | None = None,
) -> list[list[str]]:
    normalized_pairs: list[list[str]] = []

    for pair in keyword_pairs or []:
        normalized_pair: list[str] = []
        for term in pair or []:
            cleaned = re.sub(r"\s+", " ", str(term or "")).strip()
            if cleaned and cleaned not in normalized_pair:
                normalized_pair.append(cleaned)
        if normalized_pair:
            normalized_pairs.append(normalized_pair)

    if normalized_pairs:
        return normalized_pairs

    for term in search_terms or []:
        cleaned = re.sub(r"\s+", " ", str(term or "")).strip()
        if cleaned:
            normalized_pairs.append([cleaned])
    return normalized_pairs


def _flatten_keyword_pairs(keyword_pairs: list[list[str]]) -> list[str]:
    flattened: list[str] = []
    for pair in keyword_pairs:
        for term in pair:
            if term not in flattened:
                flattened.append(term)
    return flattened


def _find_matching_sentences(raw_text: str, keyword_pairs: list[list[str]], limit: int = 50) -> list[str]:
    if not raw_text or not keyword_pairs:
        return []

    matched_sentences: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(raw_text):
        if any(any(_text_matches_term(sentence, term) for term in pair) for pair in keyword_pairs):
            normalized_sentence = _normalize_match_term(sentence)
            if normalized_sentence in seen:
                continue
            seen.add(normalized_sentence)
            matched_sentences.append(_clip(sentence, 500))
        if len(matched_sentences) >= limit:
            break
    return matched_sentences


# ════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════

def tool_a_knowledge_base_lookup(news_text: str, img_base64: str | None = None) -> dict:
    """对比数据库，抽取可复用的判别参考信息。"""
    try:
        candidate_entries = search_similar_news(news_text, top_k=20)
    except Exception:
        candidate_entries = []

    all_news = get_all_news()
    if not all_news:
        return {
            "is_news": True,
            "found": False,
            "can_determine": False,
            "matched_id": None,
            "matched_news": None,
            "reference_items": [],
            "reference_summary": "数据库为空，暂无可参考历史记录。",
            "suggested_focuses": ["确认原始发布者", "核实核心事实", "补足独立来源"],
        }

    if not candidate_entries:
        candidate_entries = _select_reference_candidates(news_text, all_news, limit=20)
    if not candidate_entries:
        return {
            "is_news": True,
            "found": False,
            "can_determine": False,
            "matched_id": None,
            "matched_news": None,
            "reference_items": [],
            "reference_summary": "知识库中没有找到足够相似的历史记录，本次不提供参考样本。",
            "suggested_focuses": ["确认原始发布者", "核实核心事实", "补足独立来源"],
            "candidate_count": 0,
            "top_similarity_score": 0.0,
        }

    reference_summary, suggested_focuses, matched_id, found, can_determine = _build_reference_summary(
        news_text,
        candidate_entries,
    )
    result = {
        "is_news": True,
        "found": found,
        "can_determine": can_determine,
        "matched_id": matched_id,
        "reference_ids": [item["id"] for item in candidate_entries[:3] if item.get("id") is not None] if found else [],
        "reference_summary": reference_summary,
        "suggested_focuses": suggested_focuses,
    }

    result.setdefault("is_news", True)
    result.setdefault("found", False)
    result.setdefault("can_determine", False)
    result.setdefault("matched_id", None)
    result.setdefault("reference_ids", [])
    result.setdefault("reference_summary", "")
    result.setdefault("suggested_focuses", [])
    result["candidate_count"] = len(candidate_entries)
    result["top_similarity_score"] = candidate_entries[0]["score"] if candidate_entries else 0.0

    top_score = candidate_entries[0]["score"] if candidate_entries else 0.0
    if top_score < 0.12:
        result["found"] = False
        result["can_determine"] = False
        result["matched_id"] = None
        result["reference_ids"] = []
        result["reference_summary"] = "候选历史记录的相似度仍然偏低，已忽略知识库参考，避免误导后续搜索计划。"

    matched_news = None
    if result.get("matched_id"):
        matched_rows = search_news(news_id=result["matched_id"])
        if matched_rows:
            matched_news = matched_rows[0]
            result["matched_news"] = matched_news
            matched_score = next((item["score"] for item in candidate_entries if item["id"] == result["matched_id"]), 0.0)
            if matched_news.get("is_real") is not None and result.get("can_determine") and matched_score >= 0.45:
                result["found"] = True
            else:
                result["can_determine"] = False

    reference_items = []
    for ref_id in result.get("reference_ids", []):
        ref_item = next((candidate for candidate in candidate_entries if candidate["id"] == ref_id), None)
        if ref_item:
            reference_items.append(
                {
                    "id": ref_item.get("id"),
                    "news_text": ref_item.get("news_text", ""),
                    "is_real": ref_item.get("is_real"),
                    "reason": ref_item.get("reason", ""),
                    "evidence_url": ref_item.get("evidence_url", ""),
                    "similarity_score": ref_item.get("score", 0.0),
                }
            )

    result["reference_items"] = reference_items
    return result


def tool_c1_search_results(
    queries: list[str] | None = None,
    max_results: int = 6,
    exclude_urls: list[str] | None = None,
) -> dict:
    """纯搜索工具：支持一轮内执行多个 Tavily 查询，返回聚合后的候选结果与分查询摘要。"""
    normalized_queries = _dedupe_urls([
        _normalize_query(item)
        for item in [*(queries or [])]
        if _normalize_query(item)
    ])[:4]
    if not normalized_queries:
        return {
            "queries": [],
            "query_summaries": [],
            "results": [],
            "result_count": 0,
            "domain_distribution": {},
            "search_summary": "空查询，未执行搜索。",
            "search_errors": [],
        }

    exclude_set = set(exclude_urls or [])
    limit = max(1, min(max_results, 12))
    search_errors: list[str] = []

    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    query_summaries: list[dict[str, Any]] = []
    limit_per_query = max(1, min(limit, 8))
    total_limit = min(24, max(limit_per_query, len(normalized_queries) * limit_per_query))

    for current_query in normalized_queries:
        try:
            search_result = tavily_client.search(
                query=current_query,
                max_results=limit_per_query,
                include_raw_content=False,
            )
        except Exception as exc:
            error_text = str(exc)
            search_errors.append(f"{current_query}: {error_text}")
            query_summaries.append(
                {
                    "query": current_query,
                    "result_count": 0,
                    "domains": {},
                    "error": error_text,
                }
            )
            continue

        raw_results = search_result.get("results", [])
        query_results: list[dict[str, Any]] = []
        for item in raw_results:
            url = item.get("url", "")
            if not url or url in exclude_set or url in seen_urls:
                continue
            seen_urls.add(url)
            snippet = item.get("content") or item.get("snippet") or item.get("raw_content") or ""
            normalized_item = {
                "title": _clip(item.get("title", ""), 300),
                "snippet": _clip(snippet, 400),
                "url": url,
                "domain": item.get("domain") or _domain_from_url(url),
                "matched_queries": [current_query],
            }
            results.append(normalized_item)
            query_results.append(normalized_item)
            if len(results) >= total_limit:
                break

        query_summaries.append(
            {
                "query": current_query,
                "result_count": len(query_results),
                "domains": _domain_distribution(query_results),
                "error": "",
            }
        )
        if len(results) >= total_limit:
            break

    summary = f"批量 Tavily 搜索共执行 {len(normalized_queries)} 个查询，返回 {len(results)} 条去重候选结果。"
    if search_errors and not results:
        summary = f"批量 Tavily 搜索失败: {search_errors[0]}"
    elif search_errors:
        summary = f"批量 Tavily 搜索部分失败，但仍返回 {len(results)} 条候选结果。"

    return {
        "queries": normalized_queries,
        "query_summaries": query_summaries,
        "results": results,
        "result_count": len(results),
        "domain_distribution": _domain_distribution(results),
        "search_summary": summary,
        "search_errors": search_errors,
    }


def tool_d_source_credibility_lookup(urls: list[str]) -> dict:
    """比对信源知识库，返回候选网址的信誉画像。"""
    deduped_urls = _dedupe_urls(urls)[:10]
    profiles = []
    tier_distribution: dict[str, int] = {}
    country_distribution: dict[str, int] = {}

    for url in deduped_urls:
        profile = get_source_credibility(url)
        profile_item = {
            "url": url,
            "domain": profile.get("domain") or _domain_from_url(url),
            "name": profile.get("name", "未知来源"),
            "tier": profile.get("tier", "unknown"),
            "tier_label": profile.get("tier_label", "❓ 未知来源"),
            "credibility_score": profile.get("credibility_score", 3),
            "country": profile.get("country", "未知"),
            "region": profile.get("region", ""),
            "geo_group": profile.get("geo_group", "unknown"),
            "reason": profile.get("reason", ""),
        }
        profiles.append(profile_item)
        tier_distribution[profile_item["tier"]] = tier_distribution.get(profile_item["tier"], 0) + 1
        country_distribution[profile_item["country"]] = country_distribution.get(profile_item["country"], 0) + 1

    return {
        "urls": deduped_urls,
        "profiles": profiles,
        "profile_count": len(profiles),
        "tier_distribution": tier_distribution,
        "country_distribution": country_distribution,
    }


def tool_c2_extract_relevant_segments(
    urls: list[str],
    keyword_pairs: list[list[str]] | None = None,
    search_terms: list[str] | None = None,
) -> dict:
    """批量提取网页正文，按关键词对保留命中任一词的句子。"""
    deduped_urls = _dedupe_urls(urls)[:5]
    normalized_pairs = _normalize_keyword_pairs(keyword_pairs=keyword_pairs, search_terms=search_terms)
    normalized_terms = _flatten_keyword_pairs(normalized_pairs)
    if not deduped_urls:
        return {"urls": [], "keyword_pairs": normalized_pairs, "search_terms": normalized_terms, "results": [], "matched_url_count": 0, "sentence_list": []}

    try:
        extract_result = tavily_client.extract(urls=deduped_urls)
        extracted_items = extract_result.get("results", [])
    except Exception as exc:
        extract_result = None
        extracted_items = []
        error_text = str(exc)
    else:
        error_text = ""

    if not extracted_items:
        extracted_items = [_fetch_page_with_requests(url) for url in deduped_urls]

    matched_results = []
    sentence_list: list[str] = []
    for item in extracted_items:
        raw_content = item.get("raw_content", "") or ""
        matched_sentences = _find_matching_sentences(raw_content, normalized_pairs)
        if not matched_sentences:
            continue
        for sentence in matched_sentences:
            if sentence not in sentence_list:
                sentence_list.append(sentence)
        matched_results.append(
            {
                "url": item.get("url", ""),
                "title": _clip(item.get("title", ""), 300),
                "keyword_pairs": normalized_pairs,
                "sentences": matched_sentences,
                "sentence_count": len(matched_sentences),
                "matched_segments": matched_sentences,
                "matched_segment_count": len(matched_sentences),
                "extracted_numbers": _extract_number_mentions(raw_content),
            }
        )

    return {
        "urls": deduped_urls,
        "keyword_pairs": normalized_pairs,
        "search_terms": normalized_terms,
        "results": matched_results,
        "matched_url_count": len(matched_results),
        "sentence_list": sentence_list,
        **({"error": error_text} if error_text else {}),
    }


def tool_c3_read_full_page(url: str) -> dict:
    """返回目标网址的全文内容。"""
    try:
        extract_result = tavily_client.extract(urls=[url])
        items = extract_result.get("results", [])
        if items:
            item = items[0]
            raw_content = item.get("raw_content", "") or ""
            if raw_content.strip():
                return {
                    "url": item.get("url", url),
                    "title": _clip(item.get("title", ""), 300),
                    "raw_content": raw_content,
                    "content_length": len(raw_content),
                }
    except Exception as exc:
        fallback = _fetch_page_with_requests(url)
        if fallback.get("content_length", 0) > 0:
            fallback["warning"] = f"Tavily 提取失败，已降级到 requests 抓取: {exc}"
            return fallback
        return {"url": url, "title": "", "raw_content": "", "content_length": 0, "error": str(exc)}

    fallback = _fetch_page_with_requests(url)
    if fallback.get("content_length", 0) > 0:
        fallback["warning"] = "Tavily 未返回正文，已降级到 requests 抓取。"
        return fallback
    return {"url": url, "title": "", "raw_content": "", "content_length": 0}


def tool_save_result(
    news_text: str,
    image_path: str | None,
    classification: float | int | None,
    reason: str,
    evidence_url: str = "",
) -> dict:
    """仅负责写入数据库和向量索引，不参与最终裁决。"""
    new_id = insert_news(
        news_text=news_text,
        image_path=image_path,
        is_real=classification,
        reason=reason,
        evidence_url=evidence_url,
    )
    result = {"db_updated": True, "new_id": new_id}
    try:
        upsert_vector_entry(
            news_id=new_id,
            news_text=news_text,
            is_real=classification,
            reason=reason,
            evidence_url=evidence_url,
        )
    except Exception as exc:
        result["vector_store_warning"] = f"向量索引增量更新失败: {exc}"
    return result
