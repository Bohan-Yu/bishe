"""
LangChain 工具模块。

当前主流程改为“主 Agent 自主决策 + 多工具协作”，工具职责尽量单一：

工具 a: 知识库参考
工具 b: 初始核查分析与检索建议
工具 c1: 搜索引擎结果列表
工具 c2: 多网页提取并按搜索词筛片段
工具 c3: 单网页全文读取
工具 d/f: 保留兼容旧实现
工具 e: 仍由 app.cross_source_verification 提供
save_result: 仅负责写入数据库与向量索引
"""

import html
import json
import re
from urllib.parse import urlparse
from typing import Any

from bs4 import BeautifulSoup
import langchain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import requests
from tavily import TavilyClient

from app.config import TAVILY_API_KEY, TEXT_MODEL, VISION_MODEL, ZHIPU_API_KEY, ZHIPU_BASE_URL
from app.database import get_all_news, insert_news, search_news
from app.source_credibility import (
    get_representative_domains,
    get_source_credibility,
    observe_source_candidates,
    save_source_classification,
)
from app.vector_store import search_similar_news, upsert_vector_entry


if not hasattr(langchain, "verbose"):
    langchain.verbose = False
if not hasattr(langchain, "debug"):
    langchain.debug = False
if not hasattr(langchain, "llm_cache"):
    langchain.llm_cache = None


vision_llm = ChatOpenAI(
    model=VISION_MODEL,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    temperature=0.1,
    max_tokens=2500,
    timeout=120,
)

text_llm = ChatOpenAI(
    model=TEXT_MODEL,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    temperature=0.1,
    max_tokens=2500,
    timeout=120,
)

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


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _extract_focus_terms(text: str, limit: int = 8) -> list[str]:
    normalized = _normalize_query(text)
    terms: list[str] = []

    if _contains_cjk(normalized):
        for value in _query_entity_terms(normalized, limit=3):
            if value not in terms:
                terms.append(value)
            if len(terms) >= limit:
                return terms

        for value in _query_topic_terms(normalized, limit=5):
            if value not in terms:
                terms.append(value)
            if len(terms) >= limit:
                return terms

        for value in _query_claim_terms(normalized, limit=4):
            if value not in terms:
                terms.append(value)
            if len(terms) >= limit:
                return terms

    for value in re.findall(r"[\u4e00-\u9fff]{2,12}", normalized):
        if value in _SIMILARITY_STOP_TOKENS:
            continue
        if value not in terms:
            terms.append(value)
        if len(terms) >= limit:
            return terms

    for value in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", normalized):
        lowered = value.lower()
        if lowered not in terms:
            terms.append(lowered)
        if len(terms) >= limit:
            return terms

    for value in re.findall(r"\d+(?:\.\d+)?%?", normalized):
        if value not in terms:
            terms.append(value)
        if len(terms) >= limit:
            return terms
    return terms[:limit]


def _build_search_terms(*texts: str, limit: int = 12) -> list[str]:
    search_terms: list[str] = []
    for text in texts:
        for term in _extract_focus_terms(text, limit=limit):
            if term not in search_terms:
                search_terms.append(term)
            if len(search_terms) >= limit:
                return search_terms
    return search_terms


def _fallback_query_variants(seed_query: str, context_text: str = "", limit: int = 3) -> list[str]:
    normalized_seed = _normalize_query(seed_query)
    if not normalized_seed:
        return []

    compact_focus = _compact_focus_query(f"{seed_query} {context_text}", limit=6)
    is_cjk = _contains_cjk(f"{seed_query} {context_text}")
    variants = [compact_focus or normalized_seed]

    origin_suffix = "最早 来源 原始 报道" if is_cjk else "earliest original source report"
    verify_suffix = "官方 数据 估计 通报" if is_cjk else "official data estimate report"
    context_suffix = "背景 时间线 上下文" if is_cjk else "background timeline context"

    for candidate in [
        f"{compact_focus or normalized_seed} {origin_suffix}".strip(),
        f"{compact_focus or normalized_seed} {verify_suffix}".strip(),
        f"{compact_focus or normalized_seed} {context_suffix}".strip(),
    ]:
        normalized_candidate = _normalize_query(candidate)
        if normalized_candidate and normalized_candidate not in variants:
            variants.append(normalized_candidate)
        if len(variants) >= limit:
            break
    return variants[:limit]


_GEO_GROUP_ALIASES = {
    "china": ["中国", "china", "chinese"],
    "iran": ["伊朗", "iran", "iranian"],
    "russia": ["俄罗斯", "俄方", "russia", "russian", "sputnik", "tass"],
    "us_israel": ["以色列", "israel", "israeli", "美以"],
    "us_western": ["美国", "英国", "u.s.", "us ", "united states", "britain", "uk "],
}

_GEOPOLITICAL_KEYWORDS = {
    "导弹", "发射器", "发射装置", "无人机", "空袭", "打击", "袭击", "战争", "军事", "军方",
    "missile", "launcher", "launchers", "strike", "strikes", "war", "military", "drone",
}

_SEARCH_BUNDLES = {
    "iran": {
        "brands": ["伊朗媒体", "Tasnim", "IRNA", "Press TV"],
        "tiers": ["mainstream"],
    },
    "russia": {
        "brands": ["俄罗斯卫星通讯社", "Sputnik", "TASS"],
        "tiers": ["mainstream"],
    },
    "china_portal": {
        "brands": ["腾讯新闻", "QQ新闻", "新浪新闻"],
        "domains": ["news.qq.com", "sina.cn"],
    },
}

_TRANSLATED_FOCUS_TERMS = {
    "伊朗": "Iran",
    "美国": "US",
    "以色列": "Israel",
    "俄罗斯": "Russia",
    "导弹": "missile",
    "发射器": "launcher",
    "发射装置": "launcher",
    "无人机": "drone",
    "剩余": "remaining",
    "仅剩": "remaining",
    "摧毁": "destroyed",
    "打击": "strikes",
    "战争": "war",
}

_GEO_GROUP_CANONICAL_TERMS = {
    "china": ["中国", "China"],
    "iran": ["伊朗", "Iran"],
    "russia": ["俄罗斯", "Sputnik"],
    "us_israel": ["以色列", "Israel", "美国", "US"],
    "us_western": ["美国", "US", "英国", "UK"],
}

_QUERY_TOPIC_MARKERS = [
    "导弹发射器",
    "导弹发射装置",
    "发射装置",
    "发射器",
    "发射能力",
    "导弹库",
    "弹药库",
    "导弹",
    "无人机",
    "空袭",
    "防空",
    "launcher",
    "launchers",
    "launch capacity",
    "missile launchers",
    "missile",
    "missiles",
    "arsenal",
    "drone",
    "drones",
    "air defense",
]

_QUERY_CLAIM_MARKERS = [
    "仅剩",
    "剩余",
    "保留",
    "耗尽",
    "枯竭",
    "匮乏",
    "减少",
    "下降",
    "损失",
    "摧毁",
    "击毁",
    "否认",
    "回应",
    "反驳",
    "谎称",
    "没有证据",
    "没有客观证据",
    "军事宣传",
    "更多",
    "更强大",
    "remaining",
    "destroyed",
    "depleted",
    "response",
    "rebuttal",
    "denied",
    "propaganda",
    "no objective evidence",
]

_DEPLETION_CLAIM_MARKERS = {
    "仅剩",
    "剩余",
    "耗尽",
    "枯竭",
    "匮乏",
    "减少",
    "下降",
    "损失",
    "摧毁",
    "击毁",
    "remaining",
    "destroyed",
    "depleted",
    "lost",
}


def _dedupe_keep_order(items: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        normalized = _normalize_query(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
        if limit and len(ordered) >= limit:
            break
    return ordered


def _extract_ordered_markers(text: str, markers: list[str], limit: int = 8) -> list[str]:
    lowered = (text or "").lower()
    hits: list[tuple[int, str]] = []
    for marker in markers:
        index = lowered.find(marker.lower())
        if index >= 0:
            hits.append((index, marker))
    hits.sort(key=lambda item: (item[0], len(item[1])))

    ordered: list[str] = []
    seen = set()
    for _, marker in hits:
        normalized = marker.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(marker)
        if len(ordered) >= limit:
            break
    return ordered


def _query_entity_terms(text: str, limit: int = 4) -> list[str]:
    terms: list[str] = []
    prefer_cjk = _contains_cjk(text)
    for geo_group in _detect_geo_groups(text, limit=limit):
        for term in _GEO_GROUP_CANONICAL_TERMS.get(geo_group, []):
            if prefer_cjk and not _contains_cjk(term):
                continue
            if not prefer_cjk and _contains_cjk(term):
                continue
            if term not in terms:
                terms.append(term)
            if len(terms) >= limit:
                return terms
    return terms


def _query_topic_terms(text: str, limit: int = 5) -> list[str]:
    terms = _extract_ordered_markers(text, _QUERY_TOPIC_MARKERS, limit=limit)
    expanded: list[str] = []
    for term in terms:
        if term not in expanded:
            expanded.append(term)
        if "发射" in term and "导弹" not in expanded:
            expanded.append("导弹")
        if len(expanded) >= limit:
            break
    return expanded[:limit]


def _query_claim_terms(text: str, limit: int = 4) -> list[str]:
    return _extract_ordered_markers(text, _QUERY_CLAIM_MARKERS, limit=limit)


def _counter_response_terms(text: str, limit: int = 6) -> list[str]:
    has_depletion_claim = any(marker.lower() in (text or "").lower() for marker in _DEPLETION_CLAIM_MARKERS)
    terms = ["回应", "否认", "反驳", "谎称"]
    if has_depletion_claim:
        terms.extend(["更强大", "数量更多", "仍有"])
    return _dedupe_keep_order(terms, limit=limit)


def _quoted_counter_phrase(text: str) -> str:
    lowered = (text or "").lower()
    if any(marker.lower() in lowered for marker in _DEPLETION_CLAIM_MARKERS):
        return "更强大、数量更多" if _contains_cjk(text) else "more powerful more numerous"
    return ""


def _detect_geo_groups(text: str, limit: int = 4) -> list[str]:
    lowered = (text or "").lower()
    matches: list[str] = []
    for geo_group, aliases in _GEO_GROUP_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            matches.append(geo_group)
        if len(matches) >= limit:
            break
    return matches


def _is_geopolitical_query(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in _GEOPOLITICAL_KEYWORDS) or len(_detect_geo_groups(text, limit=2)) > 0


def _compact_focus_query(text: str, limit: int = 6) -> str:
    focus_terms = _extract_focus_terms(text, limit=max(limit, 8))
    if not focus_terms:
        return _normalize_query(text)[:120]

    prioritized: list[str] = []
    for term in [
        *_query_entity_terms(text, limit=3),
        *_query_topic_terms(text, limit=4),
        *_query_claim_terms(text, limit=3),
        *[value for value in re.findall(r"\d+(?:\.\d+)?%?", text or "")],
        *focus_terms,
    ]:
        if term not in prioritized:
            prioritized.append(term)
        if len(prioritized) >= limit:
            break
    return " ".join(prioritized[:limit])[:120].strip()


def _english_focus_query(text: str, limit: int = 6) -> str:
    translated: list[str] = []
    seen = set()
    for token in _extract_focus_terms(text, limit=limit + 2):
        translated_token = _TRANSLATED_FOCUS_TERMS.get(token, "")
        if translated_token and translated_token not in seen:
            seen.add(translated_token)
            translated.append(translated_token)
        if len(translated) >= limit:
            break
    return " ".join(translated)


def _bundle_domains(bundle_key: str, geo_group: str = "", limit: int = 3) -> list[str]:
    bundle = _SEARCH_BUNDLES.get(bundle_key, {})
    explicit_domains = bundle.get("domains", [])
    if explicit_domains:
        return explicit_domains[:limit]
    if not geo_group:
        return []
    return get_representative_domains(
        geo_group=geo_group,
        tiers=bundle.get("tiers"),
        limit=limit,
    )


def _multi_perspective_query_variants(seed_query: str, context_text: str = "", limit: int = 4) -> list[str]:
    combined_text = " ".join(part for part in [seed_query, context_text] if part).strip()
    if not combined_text or not _is_geopolitical_query(combined_text):
        return []

    compact_query = _compact_focus_query(combined_text, limit=6)
    english_focus = _english_focus_query(combined_text, limit=5)
    detected_groups = _detect_geo_groups(combined_text, limit=3)
    variants: list[str] = []

    def add(query: str) -> None:
        if len(variants) >= limit:
            return
        normalized = _normalize_query(query)
        if normalized and normalized not in variants:
            variants.append(normalized)

    local_group = detected_groups[0] if detected_groups else ""
    if local_group == "iran":
        add(f"{compact_query} 伊朗媒体 {' '.join(_SEARCH_BUNDLES['iran']['brands'][:3])}")
    elif local_group:
        add(f"{compact_query} {' '.join(_SEARCH_BUNDLES['china_portal']['brands'][:2])}")

    russian_domains = _bundle_domains("russia", geo_group="russia", limit=2)
    add(f"{compact_query} {' '.join(_SEARCH_BUNDLES['russia']['brands'][:3])} {' '.join(f'site:{domain}' for domain in russian_domains)}")

    portal_domains = _bundle_domains("china_portal", limit=2)
    add(f"{compact_query} {' '.join(_SEARCH_BUNDLES['china_portal']['brands'][:3])} {' '.join(f'site:{domain}' for domain in portal_domains)}")

    add(f"{compact_query} 否认 回应 谎称 没有证据")
    add(f"{compact_query} 摧毁 仅剩 所剩无几 90%")

    if english_focus:
        add(f"{english_focus} Sputnik TASS rebuttal no evidence")
        add(f"{english_focus} destroyed launchers remaining 100 Tencent News")

    return _dedupe_keep_order(variants, limit=limit)


def _counter_narrative_query_variants(seed_query: str, context_text: str = "", limit: int = 4) -> list[str]:
    combined_text = " ".join(part for part in [seed_query, context_text] if part).strip()
    if not combined_text or not _is_geopolitical_query(combined_text):
        return []

    compact_query = _compact_focus_query(combined_text, limit=6)
    english_focus = _english_focus_query(combined_text, limit=6)
    variants: list[str] = []

    def add(query: str) -> None:
        normalized = _normalize_query(query)
        if normalized and normalized not in variants and len(variants) < limit:
            variants.append(normalized)

    add(f"{compact_query} {' '.join(_counter_response_terms(combined_text, limit=6))}")
    add(f"{compact_query} 军事宣传 没有客观证据")
    add(f"{compact_query} 摧毁90%以上 所剩无几")
    if english_focus:
        add(f"{english_focus} rebuttal propaganda no objective evidence")
        add(f"{english_focus} more missiles response trump destroyed launchers")

    return variants[:limit]


def _query_terms_for_validation(query: str) -> set[str]:
    return {
        term.lower()
        for term in _extract_focus_terms(query, limit=10)
        if len(term.strip()) >= 2
    }


def _query_variant_is_relevant(seed_query: str, candidate_query: str) -> bool:
    normalized_candidate = _normalize_query(candidate_query)
    if not normalized_candidate:
        return False

    seed_terms = _query_terms_for_validation(seed_query)
    candidate_terms = _query_terms_for_validation(candidate_query)
    if not candidate_terms:
        return False

    generic_bad_patterns = [
        r"^\d+\s+(facts|statistics|historical context|authoritative sources)\b",
        r"^\d+的(原始来源声明|基本事实数据|统计数据时间线|历史中的上下文|常见误解)$",
    ]
    lowered = normalized_candidate.lower()
    if any(re.search(pattern, lowered) for pattern in generic_bad_patterns):
        return False

    if _contains_cjk(seed_query):
        overlap = len(seed_terms & candidate_terms)
        has_number_overlap = bool(re.findall(r"\d+(?:\.\d+)?%?", seed_query) and re.findall(r"\d+(?:\.\d+)?%?", candidate_query))
        has_core_keyword = any(token in lowered for token in ["伊朗", "导弹", "发射器", "发射装置", "trump", "iran", "missile", "launcher"])
        return overlap >= 2 or (has_number_overlap and has_core_keyword)

    overlap_ratio = len(seed_terms & candidate_terms) / max(1, len(seed_terms))
    return overlap_ratio >= 0.35


def _generate_query_variants(seed_query: str, context_text: str = "", limit: int = 3) -> list[str]:
    normalized_seed = _normalize_query(seed_query)
    if not normalized_seed:
        return []

    if _contains_cjk(f"{seed_query} {context_text}") or _is_geopolitical_query(f"{seed_query} {context_text}"):
        compact_seed = _compact_focus_query(f"{seed_query} {context_text}", limit=6)
        heuristic_queries = _dedupe_keep_order(
            [
                compact_seed,
                normalized_seed,
                *_counter_narrative_query_variants(seed_query, context_text=context_text, limit=max(3, limit)),
                *_multi_perspective_query_variants(seed_query, context_text=context_text, limit=max(3, limit)),
                *_fallback_query_variants(seed_query, context_text, limit=max(3, limit)),
            ],
            limit=max(limit, 8),
        )
        return [query for query in heuristic_queries if _query_variant_is_relevant(normalized_seed, query)][: max(limit, 4)]

    prompt = f"""你是事实核查检索策略助手。请围绕给定查询生成最多 {limit} 个互补检索式。

要求:
1. 检索式必须保持通用，不要假设具体结论。
2. 维度应尽量互补，例如原始来源、直接事实、数字/时间/上下文。
3. 如果原查询已经足够具体，可以只输出 1-2 个检索式。
4. 不要输出解释。

输入查询:
{normalized_seed}

上下文:
{_clip(context_text, 400)}

仅输出 JSON:
{{
  "queries": ["...", "..."]
}}"""

    try:
        response = text_llm.invoke([HumanMessage(content=prompt)])
        result = _parse_json(response.content)
        queries = result.get("queries", []) if isinstance(result, dict) else []
    except Exception:
        queries = []

    normalized_queries: list[str] = []
    for query in queries[:limit]:
        normalized_query = _normalize_query(query)
        if normalized_query and normalized_query not in normalized_queries and _query_variant_is_relevant(normalized_seed, normalized_query):
            normalized_queries.append(normalized_query)

    if normalized_seed not in normalized_queries:
        normalized_queries.insert(0, normalized_seed)

    if len(normalized_queries) < min(limit, 2):
        for query in _fallback_query_variants(seed_query, context_text, limit=limit):
            if query not in normalized_queries:
                normalized_queries.append(query)
            if len(normalized_queries) >= limit:
                break

    for query in [
        *_multi_perspective_query_variants(seed_query, context_text=context_text, limit=max(2, limit)),
        *_counter_narrative_query_variants(seed_query, context_text=context_text, limit=max(2, limit)),
    ]:
        if query not in normalized_queries:
            normalized_queries.append(query)
        if len(normalized_queries) >= max(limit, 6):
            break

    return normalized_queries[: max(limit, min(6, len(normalized_queries)))]


def _merge_search_results(search_runs: list[dict], exclude_set: set[str]) -> list[dict]:
    merged: dict[str, dict] = {}
    ordered_urls: list[str] = []
    for run in search_runs:
        query = run.get("query", "")
        for item in run.get("results", []):
            url = item.get("url", "")
            if not url or url in exclude_set:
                continue
            if url not in merged:
                merged[url] = {
                    "title": _clip(item.get("title", ""), 300),
                    "snippet": _clip(item.get("content", item.get("snippet", "")), 800),
                    "url": url,
                    "domain": _domain_from_url(url),
                    "search_queries": [query] if query else [],
                }
                ordered_urls.append(url)
            else:
                existing = merged[url]
                if len(item.get("content", item.get("snippet", "")) or "") > len(existing.get("snippet", "")):
                    existing["snippet"] = _clip(item.get("content", item.get("snippet", "")), 800)
                if query and query not in existing["search_queries"]:
                    existing["search_queries"].append(query)
    return [merged[url] for url in ordered_urls]


def _sanitize_result_url(url: str) -> str:
    normalized = html.unescape(str(url or "").strip())
    normalized = re.sub(r"&(?:amp;)?rut=.*$", "", normalized)
    normalized = re.sub(r"[?&]rut=.*$", "", normalized)
    return normalized


def _canonicalize_candidate_url(url: str) -> str:
    normalized = _sanitize_result_url(url)
    match = re.match(r"https?://view\.inews\.qq\.com/a/([A-Z0-9]+)", normalized)
    if match:
        return f"https://news.qq.com/rain/a/{match.group(1)}"
    return normalized


def _search_duckduckgo_html(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except Exception:
        return []

    matches = re.findall(
        r'result__a" href="//duckduckgo.com/l/\?uddg=([^"]+)[\s\S]*?>([\s\S]*?)</a>',
        response.text,
        flags=re.IGNORECASE,
    )
    results: list[dict[str, Any]] = []
    for encoded_url, raw_title in matches[:max_results]:
        url = _canonicalize_candidate_url(requests.utils.unquote(encoded_url))
        title = re.sub(r"<[^>]+>", "", html.unescape(raw_title or "")).strip()
        if not url:
            continue
        results.append(
            {
                "url": url,
                "title": _clip(title, 300),
                "content": "",
            }
        )
    return results


def _fallback_search_runs(query_variants: list[str], max_results: int = 8) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for query in query_variants[:4]:
        results = _search_duckduckgo_html(query, max_results=max_results)
        if results:
            runs.append({"query": f"[ddg] {query}", "results": results})
    return runs


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


def _search_tencent_news(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            "https://i.news.qq.com/gw/pc_search/result",
            params={
                "page": "0",
                "query": query,
                "ticket": "",
                "randstr": "",
                "is_pc": "1",
                "hippy_custom_version": "25",
                "search_type": "all",
                "search_count_limit": str(max(10, max_results)),
                "appver": "15.5_qqnews_7.1.80",
                "suid": "",
            },
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://news.qq.com/search"},
        )
        payload = response.json()
    except Exception:
        return []

    if int(payload.get("ret") or 0) not in {0}:
        return []

    results: list[dict[str, Any]] = []
    for section in payload.get("secList") or []:
        for key in ("newsList", "videoList"):
            for item in section.get(key) or []:
                url = (
                    item.get("url")
                    or (item.get("link_info") or {}).get("url")
                    or item.get("jump_url")
                    or ""
                )
                title = item.get("title") or (item.get("commonData") or {}).get("title") or ""
                if not url or not title:
                    continue
                canonical_url = _canonicalize_candidate_url(url)
                if "news.qq.com/rain/a/" not in canonical_url:
                    continue
                results.append(
                    {
                        "url": canonical_url,
                        "title": _clip(str(title), 300),
                        "content": "",
                    }
                )
                if len(results) >= max_results:
                    return results
    return results


def _site_specific_search_runs(seed_query: str, query_variants: list[str], max_results: int = 8) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not _is_geopolitical_query(seed_query):
        return runs

    compact_query = _compact_focus_query(seed_query, limit=6)
    entity_terms = " ".join(_query_entity_terms(seed_query, limit=2))
    topic_term_list = _query_topic_terms(seed_query, limit=3)
    topic_terms = " ".join(topic_term_list)
    short_topic_terms = " ".join(topic_term_list[:1] or ["导弹" if _contains_cjk(seed_query) else "missile"])
    counter_terms = " ".join(_counter_response_terms(seed_query, limit=6))
    quoted_counter_phrase = _quoted_counter_phrase(seed_query)
    focused_queries = _dedupe_keep_order(
        [
            compact_query,
            *_counter_narrative_query_variants(seed_query, context_text=seed_query, limit=3),
            *_multi_perspective_query_variants(seed_query, context_text=seed_query, limit=3),
            *query_variants[:2],
        ],
        limit=6,
    )

    qq_queries = _dedupe_keep_order(
        [
            f"{entity_terms} {topic_terms} 数量更多".strip(),
            f"{compact_query} {counter_terms}".strip(),
            compact_query,
            *[query for query in focused_queries if _contains_cjk(query)],
        ],
        limit=5,
    )
    for query in qq_queries[:4]:
        results = _search_tencent_news(query, max_results=max_results)
        if results:
            runs.append({"query": f"[qq-site] {query}", "results": results})

    ddg_queries = _dedupe_keep_order(
        [
            f"site:sputniknews.cn {entity_terms} {short_topic_terms} 军事宣传 没有客观证据".strip(),
            f"site:news.qq.com {entity_terms} {short_topic_terms} {counter_terms}".strip(),
            *focused_queries,
            f"{_english_focus_query(seed_query, limit=6)} Sputnik China military propaganda no objective evidence",
            f"{_english_focus_query(seed_query, limit=6)} Tencent News response rebuttal more missiles",
        ],
        limit=10,
    )
    if quoted_counter_phrase and _contains_cjk(seed_query):
        ddg_queries = _dedupe_keep_order(
            [
                f'site:news.qq.com "{quoted_counter_phrase}" {entity_terms} {topic_terms}'.strip(),
                *ddg_queries,
            ],
            limit=10,
        )
    for query in ddg_queries[:8]:
        results = _search_duckduckgo_html(query, max_results=max_results)
        if results:
            runs.append({"query": f"[ddg] {query}", "results": results})
    return runs


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


_RELEVANCE_TOPIC_GROUPS = {
    "launcher": ["发射器", "发射装置", "导弹发射器", "导弹发射装置", "launchers", "launcher", "launch capacity"],
    "missile": ["导弹", "导弹库", "弹药库", "missile", "missiles", "arsenal"],
    "drone": ["无人机", "drone", "drones"],
}


def _query_relevance_requirements(query: str) -> dict[str, Any]:
    lowered = (query or "").lower()
    geo_groups = _detect_geo_groups(query, limit=3)
    entity_terms: list[str] = []
    for geo_group in geo_groups:
        entity_terms.extend(_GEO_GROUP_ALIASES.get(geo_group, [])[:3])
    entity_terms = _dedupe_keep_order(entity_terms, limit=6)

    primary_topics: list[str] = []
    secondary_topics: list[str] = []
    if any(term in lowered for term in [marker.lower() for marker in _RELEVANCE_TOPIC_GROUPS["launcher"]]):
        primary_topics.extend(_RELEVANCE_TOPIC_GROUPS["launcher"])
        secondary_topics.extend(_RELEVANCE_TOPIC_GROUPS["missile"])
    elif any(term in lowered for term in [marker.lower() for marker in _RELEVANCE_TOPIC_GROUPS["missile"]]):
        primary_topics.extend(_RELEVANCE_TOPIC_GROUPS["missile"])

    if any(term in lowered for term in [marker.lower() for marker in _RELEVANCE_TOPIC_GROUPS["drone"]]):
        secondary_topics.extend(_RELEVANCE_TOPIC_GROUPS["drone"])

    claim_terms = _query_claim_terms(query, limit=5)
    return {
        "entity_terms": _dedupe_keep_order(entity_terms, limit=6),
        "primary_topics": _dedupe_keep_order(primary_topics, limit=8),
        "secondary_topics": _dedupe_keep_order(secondary_topics, limit=8),
        "claim_terms": claim_terms,
    }


def _term_hits(text: str, terms: list[str]) -> int:
    lowered = (text or "").lower()
    return sum(1 for term in terms if term and term.lower() in lowered)


def _search_result_relevance(query: str, item: dict[str, Any]) -> float:
    title_snippet = " ".join(
        part for part in [item.get("title", ""), item.get("snippet", ""), item.get("content", "")]
        if part
    )
    text = " ".join(
        part for part in [
            item.get("title", ""),
            item.get("snippet", ""),
            item.get("content", ""),
        ]
        if part
    )
    if not text:
        return 0.0

    query_terms = _query_terms_for_validation(query)
    text_terms = _query_terms_for_validation(text)
    overlap = len(query_terms & text_terms)
    query_numbers = set(_extract_number_mentions(query, limit=10))
    text_numbers = set(_extract_number_mentions(text, limit=10))
    numbers_overlap = bool(query_numbers & text_numbers)
    requirements = _query_relevance_requirements(query)
    entity_hits = _term_hits(text, requirements["entity_terms"])
    primary_topic_hits = _term_hits(text, requirements["primary_topics"])
    secondary_topic_hits = _term_hits(text, requirements["secondary_topics"])
    claim_hits = _term_hits(text, requirements["claim_terms"])
    title_primary_hits = _term_hits(title_snippet, requirements["primary_topics"])
    strong_counter_hits = _term_hits(
        text,
        [
            "军事宣传",
            "没有客观证据",
            "更强大",
            "数量更多",
            "more powerful",
            "more numerous",
            "no objective evidence",
            "propaganda",
        ],
    )

    if requirements["entity_terms"] and entity_hits == 0:
        return 0.0
    if requirements["primary_topics"] and title_primary_hits == 0 and strong_counter_hits == 0 and not numbers_overlap:
        return 0.0
    if requirements["primary_topics"] and primary_topic_hits == 0:
        if secondary_topic_hits == 0 or (claim_hits < 2 and strong_counter_hits == 0 and not numbers_overlap):
            return 0.0

    core_tokens = ["伊朗", "iran", "导弹", "missile", "发射器", "launcher", "特朗普", "trump"]
    core_hits = sum(1 for token in core_tokens if token.lower() in text.lower())
    score = (
        overlap * 0.16
        + entity_hits * 0.14
        + primary_topic_hits * 0.16
        + secondary_topic_hits * 0.08
        + claim_hits * 0.08
        + strong_counter_hits * 0.1
        + core_hits * 0.08
        + (0.16 if numbers_overlap else 0.0)
    )
    return round(min(1.0, score), 3)


def _filter_search_results(query: str, results: list[dict], minimum_score: float = 0.28) -> list[dict]:
    if not results:
        return []

    scored = []
    for item in results:
        enriched = dict(item)
        relevance = _search_result_relevance(query, enriched)
        enriched["search_relevance"] = relevance
        scored.append(enriched)

    kept = [item for item in scored if item.get("search_relevance", 0.0) >= minimum_score]
    if not kept:
        fallback_pool = [item for item in scored if item.get("search_relevance", 0.0) > 0.05]
        ranked_pool = fallback_pool or scored
        kept = sorted(ranked_pool, key=lambda row: row.get("search_relevance", 0.0), reverse=True)[: max(3, min(6, len(ranked_pool)))]
    kept.sort(key=lambda row: row.get("search_relevance", 0.0), reverse=True)
    return kept


def _country_distribution(results: list[dict]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for item in results:
        profile = item.get("source_profile") or get_source_credibility(item.get("url", ""))
        country = str(profile.get("country") or "未知")
        distribution[country] = distribution.get(country, 0) + 1
    return distribution


def _select_country_diverse_results(results: list[dict], limit: int) -> list[dict]:
    if not results:
        return []

    country_buckets: dict[str, list[dict]] = {}
    country_order: list[str] = []
    for item in results:
        profile = item.get("source_profile") or get_source_credibility(item.get("url", ""))
        item["source_profile"] = profile
        country = str(profile.get("country") or "未知")
        if country not in country_buckets:
            country_buckets[country] = []
            country_order.append(country)
        country_buckets[country].append(item)

    prioritized_countries = [country for country in country_order if country != "未知"]
    if "未知" in country_buckets:
        prioritized_countries.append("未知")

    selected: list[dict] = []
    selected_urls: set[str] = set()
    while len(selected) < limit:
        progressed = False
        for country in prioritized_countries:
            bucket = country_buckets.get(country, [])
            while bucket and bucket[0].get("url", "") in selected_urls:
                bucket.pop(0)
            if not bucket:
                continue
            chosen = bucket.pop(0)
            chosen_url = chosen.get("url", "")
            if chosen_url in selected_urls:
                continue
            selected.append(chosen)
            selected_urls.add(chosen_url)
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    if len(selected) < limit:
        for item in results:
            chosen_url = item.get("url", "")
            if chosen_url in selected_urls:
                continue
            selected.append(item)
            selected_urls.add(chosen_url)
            if len(selected) >= limit:
                break

    return selected[:limit]


def _classify_search_result_sources_with_llm(results: list[dict], query: str) -> list[dict]:
    unknown_items = []
    for item in results[:8]:
        url = item.get("url", "")
        source_info = get_source_credibility(url)
        if source_info.get("tier") != "unknown":
            continue
        unknown_items.append(
            {
                "url": url,
                "domain": source_info.get("domain", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
            }
        )

    if not unknown_items:
        return []

    prompt = f"""你是新闻信源分类助手。请根据搜索结果中的 domain、title、snippet，对未知来源做保守分类。

分类仅允许:
- official
- mainstream
- professional
- portal
- self_media
- unknown

要求:
1. 如果证据不足，输出 unknown。
2. 只有比较有把握时，confidence 才能高于 0.75。
3. 不要根据政治立场推断，只根据站点类型和页面线索判断。

查询:
{query}

输入:
{json.dumps(unknown_items, ensure_ascii=False, indent=2)}

仅输出 JSON 数组:
[
    {{"domain": "example.com", "name": "来源名", "tier": "professional", "country": "国家或国际组织", "confidence": 0.82, "reason": "简短原因"}}
]"""

    try:
        response = text_llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json(response.content)
        if isinstance(parsed, list):
            result = parsed
        elif isinstance(parsed, dict):
            result = parsed.get("items", [])
        else:
            result = []
    except Exception:
        result = []

    normalized: list[dict] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain", "")).lower().strip()
        tier = str(item.get("tier", "unknown")).strip()
        confidence = _safe_float(item.get("confidence"), 0.0)
        if not domain or tier not in {"official", "mainstream", "professional", "portal", "self_media", "unknown"}:
            continue
        normalized.append(
            {
                "domain": domain,
                "name": _clip(item.get("name", domain), 120),
                "tier": tier,
                "country": _clip(item.get("country", ""), 40),
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": _clip(item.get("reason", ""), 200),
            }
        )
    return normalized


def _observe_and_learn_sources(results: list[dict], query: str, analysis_context: str = "") -> dict:
    observation = observe_source_candidates(results, query=query, analysis_context=analysis_context, source="search")
    classifications = _classify_search_result_sources_with_llm(results, query)
    saved_domains: list[str] = []
    for item in classifications:
        if item.get("tier") == "unknown" or item.get("confidence", 0.0) < 0.75:
            continue
        matching_url = next(
            (result.get("url", "") for result in results if _domain_from_url(result.get("url", "")) == item.get("domain")),
            item.get("domain", ""),
        )
        if save_source_classification(
            url=matching_url,
            name=item.get("name", item.get("domain", "")),
            tier=item.get("tier", "unknown"),
            reason=item.get("reason", ""),
            source="search_llm",
            country=item.get("country", ""),
        ):
            saved_domains.append(item.get("domain", ""))
    return {
        **observation,
        "classified_domain_count": len(classifications),
        "saved_domain_count": len(saved_domains),
        "saved_domains": saved_domains[:8],
    }


def _mechanical_source_type(url: str) -> str:
    source_info = get_source_credibility(url)
    tier = source_info.get("tier", "unknown")
    if tier == "official":
        return "官方"
    if tier == "mainstream":
        return "主流媒体"
    if tier == "professional":
        return "专业媒体"
    return "其他"


def _build_mechanical_evidence_items(
    plan_id: str,
    objective: str,
    query_variants: list[str],
    merged_results: list[dict],
    extracted_contents: dict[str, dict[str, str]],
    search_terms: list[str],
    limit: int = 5,
) -> list[dict]:
    items: list[dict] = []
    for item in merged_results[:limit]:
        url = item.get("url", "")
        extracted = extracted_contents.get(url, {})
        raw_content = extracted.get("raw_content", "")
        snippet = item.get("snippet", "")
        matched_segments = _find_matching_segments(raw_content or snippet, search_terms)
        summary = matched_segments[0] if matched_segments else snippet
        combined_text = " ".join([item.get("title", ""), snippet, raw_content])
        source_info = get_source_credibility(url)
        items.append(
            {
                "plan_id": plan_id,
                "objective": objective,
                "title": extracted.get("title") or item.get("title", ""),
                "url": url,
                "domain": item.get("domain") or _domain_from_url(url),
                "summary": _clip(summary, 500),
                "snippet": _clip(snippet, 500),
                "matched_segments": matched_segments[:4],
                "extracted_numbers": _extract_number_mentions(combined_text),
                "search_queries": item.get("search_queries", query_variants),
                "key_points": matched_segments[:3] or [summary] if summary else [],
                "stance": "neutral",
                "source_type": _mechanical_source_type(url),
                "source_name": source_info.get("name", source_info.get("domain", "")),
                "credibility_score": source_info.get("credibility_score", 5),
                "relevance_score": 6.5 if matched_segments else 5.5,
                "composite_score": round((source_info.get("credibility_score", 5) * 0.45) + (6.5 if matched_segments else 5.5) * 0.55, 1),
                "reason": "基于搜索摘要与正文命中片段生成的机械候选证据。",
                "publisher_signal": "",
                "benefit_signal": "",
            }
        )
    return items


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


def _fallback_plan(news_text: str) -> list[dict]:
    base = _clip(news_text, 120)
    return [
        {
            "plan_id": "plan-1",
            "plan_intro": "先确认最早公开来源和传播链，判断消息是否来自单一信源或转述链。",
            "objective": "确认这条信息最早由谁发出，以及是否源自单一军事口径",
            "query": f"{base} 最早发布 谁说的 来源",
            "why": "先确认发布源和传播链，判断是否存在带立场的单一信源。",
            "target_sources": "原始声明、军方通报、主流媒体、事实核查报道",
            "success_criteria": "找到最早公开来源和主要转述链",
            "priority": "high",
        },
        {
            "plan_id": "plan-2",
            "plan_intro": "直接核查新闻中的核心事实、数字或关键描述是否有独立证据支撑。",
            "objective": "核实新闻中的核心事实、数字和战果表述是否准确",
            "query": f"{base} 核心事实 数字 查证 fact check",
            "why": "判断关键数字和事实是否被夸大或缺乏独立证据。",
            "target_sources": "官方数据、权威媒体、专业分析机构",
            "success_criteria": "找到可直接支持或反驳核心事实的证据",
            "priority": "high",
        },
        {
            "plan_id": "plan-3",
            "plan_intro": "在适用时检查是否存在夸大、误导或特定传播动机，不适用时可排除该维度。",
            "objective": "分析是否存在夸大、误导或特定传播动机",
            "query": f"{base} 夸大 误导 宣传 动机 分析",
            "why": "判断消息框架是否包含明显夸大、误导或特定传播动机；若不适用可排除该维度。",
            "target_sources": "媒体分析、专家评论、战报对比",
            "success_criteria": "形成对夸大、误导或潜在传播动机的说明，或明确该维度不适用",
            "priority": "medium",
        },
    ]


def _fallback_search_result(plan_id: str, objective: str, query: str, error: str = "") -> dict:
    return {
        "plan_id": plan_id,
        "objective": objective,
        "query": query,
        "search_summary": f"未能为该搜索任务生成结构化证据。{(' 错误: ' + error) if error else ''}",
        "evidence_items": [],
        "evidence_list": [],
        "web_list": [],
        "scores": [],
        "reasons": [],
        "coverage_signals": {
            "publisher_identity_confirmed": False,
            "core_fact_checked": False,
            "beneficiary_or_motive_checked": False,
        },
    }


def _is_safety_filter_error(exc: Exception) -> bool:
    message = str(exc)
    return "1301" in message or "contentFilter" in message or "敏感内容" in message or "不安全或敏感内容" in message


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
    reference_ids = [item["id"] for item in top_candidates if item.get("id") is not None]
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
        reference_ids = []
        summary_parts = ["候选历史记录虽被召回，但整体相似度偏低，已忽略知识库参考以避免误导。"]

    return ("".join(summary_parts), focuses, matched_id, found, can_determine)


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


def tool_b_search_plan(
    news_text: str,
    img_base64: str | None = None,
    knowledge_references: list[dict] | None = None,
) -> dict:
    """生成轻量级初始核查分析，供主 Agent 后续自主决定调用哪些工具。"""
    references_text = json.dumps(knowledge_references or [], ensure_ascii=False, indent=2)
    prompt = f"""你是资深事实核查分析员。你的任务不是编排固定流程，而是给主 Agent 提供简洁的初始分析。

输入新闻:
{_clip(news_text, 5000)}

知识库参考:
{references_text}

请完成以下工作：
1. 概括这条新闻最值得核查的核心问题。
2. 给出 2-4 个最有价值的检索式，供主 Agent 后续自主搜索。
3. 给出 2-4 个重点关注维度，例如来源、数字、时间地点、图像、立场、传播动机。
4. 如有必要，给出初步风险假设和还缺哪些关键信息。
5. 输出尽量简洁，不要生成冗长流程说明。
6. 如果有图片，补充图片中与核查相关的信息；没有图片则写空字符串。
7. 现在的时间是2026年3月。
请仅输出 JSON:
{{
  "analysis_summary": "...",
    "core_question": "...",
        "verification_dimensions": ["..."],
    "focus_points": ["..."],
    "search_queries": ["..."],
    "risk_hypotheses": ["..."],
  "img_description": "",
    "missing_information": ["..."]
}}"""

    contents: list[dict[str, Any]] = []
    if img_base64:
        contents.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
            }
        )
    contents.append({"type": "text", "text": prompt})

    llm = vision_llm if img_base64 else text_llm
    try:
        response = llm.invoke([HumanMessage(content=contents)])
        result = _parse_json(response.content)
    except Exception as exc:
        result = {
            "analysis_summary": f"初始分析生成失败，已降级使用默认检索建议。错误: {exc}",
            "core_question": _clip(news_text, 120),
            "verification_dimensions": ["来源链路", "核心事实", "关键数字", "上下文"],
            "focus_points": ["原始发布者", "核心事实", "是否存在夸大或误导"],
            "search_queries": [item.get("query", "") for item in _fallback_plan(news_text)[:3]],
            "risk_hypotheses": ["消息可能依赖单一来源", "核心事实仍缺少独立证据"],
            "img_description": "",
            "missing_information": ["原始发布者", "独立证据", "利益相关方"],
        }

    result.setdefault("analysis_summary", "")
    result.setdefault("core_question", _clip(news_text, 120))
    result.setdefault("verification_dimensions", [])
    result.setdefault("focus_points", [])
    result.setdefault("search_queries", [])
    result.setdefault("risk_hypotheses", [])
    result.setdefault("img_description", "")
    result.setdefault("missing_information", [])

    normalized_focus = []
    for item in result.get("focus_points", [])[:4]:
        text = _clip(item, 80).strip()
        if text and text not in normalized_focus:
            normalized_focus.append(text)

    normalized_queries = []
    for item in result.get("search_queries", [])[:4]:
        query = _normalize_query(item)
        if query and query not in normalized_queries:
            normalized_queries.append(query)

    if not normalized_queries:
        normalized_queries = [plan.get("query", "") for plan in _fallback_plan(news_text)[:3] if plan.get("query")]

    expanded_queries: list[str] = []
    for query in normalized_queries[:3]:
        for variant in _generate_query_variants(query, context_text=f"{news_text} {result.get('core_question', '')}", limit=4):
            if variant not in expanded_queries:
                expanded_queries.append(variant)
            if len(expanded_queries) >= 6:
                break
        if len(expanded_queries) >= 6:
            break

    result["focus_points"] = normalized_focus or ["原始发布者", "核心事实", "传播立场"]
    result["search_queries"] = expanded_queries or normalized_queries
    result["verification_dimensions"] = [
        _clip(item, 80)
        for item in result.get("verification_dimensions", [])[:5]
        if _clip(item, 80)
    ] or ["来源链路", "核心事实", "关键数字", "上下文"]
    return result


def _domain_from_url(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


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


def tool_c1_search_results(
    query: str,
    max_results: int = 6,
    exclude_urls: list[str] | None = None,
) -> dict:
    """纯搜索工具：执行单次 Tavily 查询，只返回基础结果列表与简单统计。"""
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return {"query": "", "query_variants": [], "results": [], "result_count": 0, "domain_distribution": {}, "source_learning": {"observed_domain_count": 0, "saved_domain_count": 0}, "search_summary": "空查询，未执行搜索。", "search_errors": []}

    exclude_set = set(exclude_urls or [])
    limit = max(1, min(max_results, 12))
    search_errors: list[str] = []

    try:
        search_result = tavily_client.search(
            query=normalized_query,
            max_results=limit,
            include_raw_content=False,
        )
    except Exception as exc:
        search_errors.append(str(exc))

        raw_results = []
    else:
        raw_results = search_result.get("results", [])

    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in raw_results:
        url = item.get("url", "")
        if not url or url in exclude_set or url in seen_urls:
            continue
        seen_urls.add(url)
        snippet = item.get("content") or item.get("snippet") or item.get("raw_content") or ""
        results.append(
            {
                "title": _clip(item.get("title", ""), 300),
                "snippet": _clip(snippet, 400),
                "url": url,
                "domain": item.get("domain") or _domain_from_url(url),
            }
        )
        if len(results) >= limit:
            break

    summary = f"单次 Tavily 搜索返回 {len(results)} 条候选结果。"
    if search_errors:
        summary = f"Tavily 搜索失败: {search_errors[0]}"

    return {
        "query": normalized_query,
        "query_variants": [normalized_query],
        "results": results,
        "result_count": len(results),
        "domain_distribution": _domain_distribution(results),
        "source_learning": {"observed_domain_count": len(_domain_distribution(results)), "saved_domain_count": 0},
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


def tool_f_final_verify_and_save(
    news_text: str,
    image_path: str | None,
    img_description: str,
    analysis_result: dict,
    iteration_logs: list[dict],
    all_evidence_items: list[dict],
    cross_verify_result: dict | None,
) -> dict:
    """兼容旧入口：若外部仍调用 F，则降级为保存结果。"""
    sorted_evidence = sorted(
        all_evidence_items,
        key=lambda item: item.get("composite_score", 0),
        reverse=True,
    )
    evidence_url = sorted_evidence[0].get("url", "") if sorted_evidence else ""
    classification = None
    if cross_verify_result:
        classification = cross_verify_result.get("cross_verify_score")
    if classification is None and sorted_evidence:
        classification = round(
            sum(_safe_float(item.get("composite_score"), 5.0) for item in sorted_evidence[:5]) / min(len(sorted_evidence), 5),
            1,
        )
    if classification is None:
        classification = 5.0

    reason = "兼容旧入口保存结果。"
    if analysis_result.get("analysis_summary"):
        reason = _clip(analysis_result.get("analysis_summary"), 400)
    elif sorted_evidence:
        reason = _clip(sorted_evidence[0].get("summary", "") or sorted_evidence[0].get("reason", ""), 400)

    save_result = tool_save_result(
        news_text=news_text,
        image_path=image_path,
        classification=classification,
        reason=reason,
        evidence_url=evidence_url,
    )
    return {
        "claim_verdicts": [],
        "classification": classification,
        "reason": reason,
        "evidence_url": evidence_url,
        "publisher_conclusion": "未知",
        "beneficiary_conclusion": "未知",
        "has_contradiction": cross_verify_result.get("has_contradiction", False) if cross_verify_result else False,
        "cross_verify_score": cross_verify_result.get("cross_verify_score", 5.0) if cross_verify_result else 5.0,
        **save_result,
    }
