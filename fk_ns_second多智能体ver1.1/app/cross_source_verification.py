"""
多源交叉验证子 Agent（Cross-Source Verification Agent）

创新点:
  1. 信源分层: 基于信誉度知识库，将搜索到的来源按 官方/主流/专业/门户/自媒体 分级
  2. 立场对比: 检测不同层级来源是否给出矛盾结论，标记矛盾信号

流程:
    输入: 搜索证据列表 + 原始声明
    处理: 信源分层 → 混合语义相关性筛选 → 立场分析
    输出: 交叉验证报告（信源分布、独立源数、矛盾检测、加权可信度）
"""

import json
import re
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.config import ZHIPU_API_KEY, ZHIPU_BASE_URL, TEXT_MODEL
from app.tools import _parse_json
from app.source_credibility import (
    batch_evaluate_sources,
    infer_source_metadata,
    save_source_classification,
    GEO_GROUP_LABELS,
    TIER_LABELS,
    TIER_SCORES,
)
from app.vector_store import batch_semantic_similarity

import langchain
if not hasattr(langchain, "verbose"):
    langchain.verbose = False
if not hasattr(langchain, "debug"):
    langchain.debug = False
if not hasattr(langchain, "llm_cache"):
    langchain.llm_cache = None

# ────────────── LLM 实例 ──────────────
_cross_verify_llm = ChatOpenAI(
    model=TEXT_MODEL,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    temperature=0.1,
    max_tokens=10000,
    timeout=120,
)

_TRACKING_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "spm", "from", "from_source", "feature", "source", "ref", "ref_src",
    "ved", "ei", "sa", "usg", "fbclid", "gclid", "igshid", "mibextid",
}

_RELEVANCE_STOPWORDS = {
    "目前", "现在", "消息", "新闻", "内容", "信息", "相关", "表示", "指出", "情况", "一个",
    "一些", "已经", "正在", "进行", "对于", "以及", "这个", "那个", "他们", "我们", "你们",
    "about", "report", "reported", "says", "said", "with", "from", "that", "this", "have",
}

_SOURCE_CLUE_PATTERNS = [
    r"据([^，。；：:]{2,30})(?:报道|消息|称|表示)",
    r"([^，。；：:]{2,30})(?:发言人|官员|军方|政府)(?:称|表示)",
    r"according to ([^,.;:]{2,40})",
    r"reported by ([^,.;:]{2,40})",
    r"citing ([^,.;:]{2,40})",
]

_EVIDENCE_CUE_PATTERNS = [
    r"据[^，。；：:]{2,30}(?:报道|消息|称|表示)",
    r"(?:报告|公告|通报|声明|研究|调查|数据显示|资料显示|文件显示)",
    r"(?:采访|证实|否认|披露|指出|提到|援引)",
    r"(?:according to|reported by|citing|data shows|statement|report|study)",
]

_MAX_RELEVANT_RECORDS = 12
_MIN_RELEVANT_RECORDS = 4

_NUMERIC_CONTEXT_PATTERNS = {
    "remaining": ["仅剩", "剩余", "remaining", "left", "still has", "available", "可用", "可正常", "仍有"],
    "active": ["活跃", "active", "operational", "可发射", "运行", "deployable"],
    "destroyed": ["摧毁", "destroyed", "taken out", "neutralized", "中和", "失效", "损失"],
    "total": ["总计", "总数", "initially", "originally", "最初", "起初", "总共有", "before"],
    "ratio": ["%", "百分之", "比例", "ratio", "share"],
}


def _clip(text: Any, limit: int = 5000) -> str:
    value = "" if text is None else str(text)
    return value[:limit]


def _coerce_cross_verify_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        normalized_items = []
        for key, item in value.items():
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            if isinstance(key, str) and key.strip():
                normalized.setdefault("source_name", key.strip())
                normalized.setdefault("title", normalized.get("title") or key.strip())
            normalized_items.append(normalized)
        return normalized_items
    return []


def _coerce_cross_verify_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def cross_source_verify(
    claim_text: str,
    evidence_list: list[str],
    web_list: list[str],
    scores: list,
    reasons: list,
    evidence_items: list[dict] | None = None,
    agent_evidence_catalog: dict | None = None,
) -> dict:
    """
    对搜索返回的证据做多源交叉验证。
    """
    evidence_items = _coerce_cross_verify_items(evidence_items)
    evidence_list = _coerce_cross_verify_list(evidence_list)
    web_list = _coerce_cross_verify_list(web_list)
    scores = _coerce_cross_verify_list(scores)
    reasons = _coerce_cross_verify_list(reasons)

    if not (evidence_items or (evidence_list and web_list)):
        return _empty_result("无证据可供交叉验证")

    evidence_records = _build_evidence_records(evidence_list, web_list, scores, reasons, evidence_items=evidence_items)
    deduped_records = _dedupe_exact_records(evidence_records)

    # ═══════════ Step 1: 信源分层 + 未知域名 LLM 兜底 ═══════════
    source_profiles = batch_evaluate_sources([record["normalized_url"] for record in deduped_records])
    source_profiles = _classify_unknown_sources_with_llm(deduped_records, source_profiles)
    for record, profile in zip(deduped_records, source_profiles):
        profile.update(
            infer_source_metadata(
                record.get("normalized_url") or record.get("url", ""),
                name=str(profile.get("name", "")),
                text=str(record.get("summary", "")),
                existing=profile,
            )
        )
        record["source_profile"] = profile

    tier_distribution = {}
    for sp in source_profiles:
        tier = sp.get("tier", "unknown")
        tier_distribution.setdefault(tier, 0)
        tier_distribution[tier] += 1

    # ═══════════ Step 2: 混合语义相关性筛选 ═══════════
    relevant_records, filtered_out_records = _filter_relevant_records(claim_text, deduped_records)

    for record in relevant_records:
        if "provenance" not in record:
            record["provenance"] = {}

    independent_count = len(relevant_records)
    stance_analysis = _build_simple_stance_analysis(claim_text, relevant_records)
    source_diversity = _analyze_source_diversity(relevant_records, [], stance_analysis)
    traceable_evidence = _build_simple_traceable_evidence(relevant_records, stance_analysis)
    numeric_analysis = _analyze_numeric_consistency(claim_text, relevant_records)
    cross_score = _calculate_weighted_score(
        [record.get("source_profile", {}) for record in relevant_records],
        [record.get("score", 5) for record in relevant_records],
        independent_count,
        stance_analysis,
        len(relevant_records),
        source_diversity,
    )
    summary_parts = []
    if source_diversity.get("summary"):
        summary_parts.append(source_diversity.get("summary", ""))
    if numeric_analysis.get("summary"):
        summary_parts.append(numeric_analysis.get("summary", ""))

    return {
        "source_profiles": source_profiles,
        "tier_distribution": tier_distribution,
        "tier_distribution_readable": {
            TIER_LABELS.get(k, k): v for k, v in tier_distribution.items()
        },
        "independent_source_count": independent_count,
        "total_source_count": len(deduped_records),
        "relevant_source_count": len(relevant_records),
        "filtered_out_count": len(filtered_out_records),
        "filtered_out_samples": [
            {
                "index": record["index"],
                "url": record["url"],
                "reason": record.get("filter_reason", "相关性不足"),
            }
            for record in filtered_out_records[:8]
        ],
        "dedup_groups": [],
        "traceable_evidence": traceable_evidence,
        "stance_analysis": stance_analysis,
        "source_diversity": source_diversity,
        "numeric_analysis": numeric_analysis,
        "cross_verify_score": round(cross_score, 1),
        "cross_verify_summary": " ".join(part for part in summary_parts if part).strip(),
        "has_contradiction": stance_analysis.get("has_contradiction", False),
    }


def _build_evidence_records(
    evidence_list: list[str],
    web_list: list[str],
    scores: list,
    reasons: list,
    evidence_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = []
    if evidence_items:
        for index, item in enumerate(evidence_items):
            if not isinstance(item, dict):
                continue
            url = item.get("url", "")
            normalized_url = _normalize_url(url)
            summary = " ".join(
                part for part in [
                    item.get("summary", ""),
                    item.get("snippet", ""),
                    "；".join(item.get("matched_segments", [])[:3]),
                ]
                if part
            )
            reason = item.get("reason", reasons[index] if index < len(reasons) else "")
            score = item.get("composite_score", scores[index] if index < len(scores) else 5.0)
            records.append(
                {
                    "index": index + 1,
                    "title": str(item.get("title", ""))[:300],
                    "summary": str(summary)[:5000],
                    "snippet": str(item.get("snippet", ""))[:1200],
                    "matched_segments": item.get("matched_segments", [])[:5],
                    "extracted_numbers": item.get("extracted_numbers", [])[:8],
                    "url": url,
                    "normalized_url": normalized_url or url,
                    "score": float(score) if isinstance(score, (int, float)) else 5.0,
                    "reason": str(reason)[:1000],
                    "source_clues": _extract_source_clues(f"{summary} {reason}"),
                    "search_queries": item.get("search_queries", [])[:6],
                    "evidence_granularity": str(item.get("evidence_granularity", "search_result"))[:40],
                    "novelty_score": float(item.get("novelty_score", 0.0) or 0.0),
                    "agent_stance_hint": str(item.get("agent_stance_hint", ""))[:20],
                    "stance_reason": str(item.get("stance_reason", ""))[:300],
                    "source_name_hint": str(item.get("source_name", ""))[:100],
                }
            )
        if records:
            return records

    for index, summary in enumerate(evidence_list):
        url = web_list[index] if index < len(web_list) else ""
        score = scores[index] if index < len(scores) and isinstance(scores[index], (int, float)) else 5.0
        reason = reasons[index] if index < len(reasons) else ""
        normalized_url = _normalize_url(url)
        records.append(
            {
                "index": index + 1,
                "title": "",
                "summary": str(summary)[:5000],
                "snippet": "",
                "matched_segments": [],
                "extracted_numbers": _extract_numbers(str(summary)),
                "url": url,
                "normalized_url": normalized_url or url,
                "score": float(score),
                "reason": str(reason)[:1000],
                "source_clues": _extract_source_clues(f"{summary} {reason}"),
                "search_queries": [],
                "evidence_granularity": "search_result",
                "novelty_score": 0.0,
                "agent_stance_hint": "",
                "stance_reason": "",
                "source_name_hint": "",
            }
        )
    return records


# ════════════════════════════════════════════════════
#  URL 规范化与去重
# ════════════════════════════════════════════════════

def _normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
    except Exception:
        return url

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_KEYS
    ]
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    scheme = parsed.scheme or "https"
    path = parsed.path.rstrip("/") or "/"
    normalized = parsed._replace(
        scheme=scheme.lower(),
        netloc=host,
        path=path,
        query=urlencode(filtered_query),
        fragment="",
    )
    return urlunparse(normalized)


def _dedupe_exact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    for record in records:
        key = record["normalized_url"] or record["url"] or f"index:{record['index']}"
        if key not in merged:
            merged[key] = dict(record)
            ordered_keys.append(key)
            continue

        existing = merged[key]
        if record.get("score", 0) > existing.get("score", 0):
            existing["summary"] = record.get("summary", existing.get("summary", ""))
            existing["score"] = record.get("score", existing.get("score", 5))
            existing["reason"] = record.get("reason", existing.get("reason", ""))
        for clue in record.get("source_clues", []):
            if clue not in existing["source_clues"]:
                existing["source_clues"].append(clue)
    return [merged[key] for key in ordered_keys]


# ════════════════════════════════════════════════════
#  未知信源 LLM 分类
# ════════════════════════════════════════════════════

def _classify_unknown_sources_with_llm(
    records: list[dict[str, Any]],
    source_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unknown_payload = []
    for index, profile in enumerate(source_profiles):
        if profile.get("tier") != "unknown":
            continue
        record = records[index] if index < len(records) else {}
        unknown_payload.append(
            {
                "index": index,
                "url": profile.get("url", record.get("url", "")),
                "domain": profile.get("domain", ""),
                "summary": record.get("summary", "")[:400],
            }
        )

    if not unknown_payload:
        return source_profiles

    prompt = f"""你是新闻信源分类助手。请对以下未命中本地知识库的网址进行来源类型判断。

可选层级仅允许:
- official: 政府、国际组织、官方机构
- mainstream: 主流综合媒体、通讯社
- professional: 专业媒体、行业媒体、研究机构、智库、百科资料站
- portal: 门户、聚合平台、论坛集合页
- self_media: 自媒体、社区帖子、个人博客、UGC 账号页
- unknown: 无法判断

判断要求:
1. 只能根据 domain、url、summary 做保守判断，不确定就输出 unknown。
2. 若是 Reddit、论坛、社区问答、个人账号页，优先判为 self_media 或 portal。
3. 若是研究机构、数据库、年鉴、百科资料页，可判为 professional。
4. 输出中的 index 必须与输入保持一致。
5. 如果能根据 domain、summary 判断来源所属国家或国际组织，请输出 country；否则输出空字符串。

输入:
{json.dumps(unknown_payload, ensure_ascii=False, indent=2)}

仅输出 JSON 数组:
[
    {{"index": 0, "name": "来源名", "tier": "professional", "country": "国家或国际组织", "reason": "简短原因"}}
]"""

    try:
        resp = _cross_verify_llm.invoke([HumanMessage(content=prompt)])
        result = _parse_json_array(resp.content)
        if not isinstance(result, list):
            return source_profiles
    except Exception:
        return source_profiles

    classified = {item.get("index"): item for item in result if isinstance(item, dict)}
    for index, item in classified.items():
        if not isinstance(index, int) or index < 0 or index >= len(source_profiles):
            continue
        tier = item.get("tier")
        if tier not in TIER_SCORES:
            continue
        source_profiles[index] = {
            **source_profiles[index],
            "name": item.get("name") or source_profiles[index].get("name") or source_profiles[index].get("domain", "未知"),
            "tier": tier,
            "tier_label": TIER_LABELS.get(tier, TIER_LABELS["unknown"]),
            "credibility_score": TIER_SCORES[tier],
            "llm_classified": True,
            "llm_reason": str(item.get("reason", ""))[:200],
            "knowledge_base_source": "llm",
            "country": str(item.get("country") or source_profiles[index].get("country", "")).strip()[:40] or source_profiles[index].get("country", "未知"),
        }
        try:
            save_source_classification(
                url=source_profiles[index].get("url") or records[index].get("url", ""),
                name=source_profiles[index].get("name", ""),
                tier=tier,
                reason=source_profiles[index].get("llm_reason", ""),
                source="llm",
                country=str(item.get("country", "")),
            )
        except Exception:
            pass
    return source_profiles


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []
        try:
            parsed = json.loads(match.group())
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []


# ════════════════════════════════════════════════════
#  相关性筛选
# ════════════════════════════════════════════════════

def _filter_relevant_records(
    claim_text: str,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semantic_scores = _estimate_semantic_scores(claim_text, records)
    scored_records = []
    for record, semantic_score in zip(records, semantic_scores):
        relevance_score, relevance_signals = _estimate_relevance(
            claim_text,
            record.get("summary", ""),
            record.get("reason", ""),
            semantic_score,
        )
        enriched = dict(record)
        enriched["relevance_score"] = relevance_score
        enriched["semantic_similarity"] = round(semantic_score, 4)
        enriched["relevance_signals"] = relevance_signals
        scored_records.append(enriched)

    kept, filtered = _partition_relevant_records(scored_records)
    for record in filtered:
        record["filter_reason"] = "; ".join(record.get("relevance_signals", [])) or "与待验证声明的实体、数字和核心描述重合不足"

    if not kept and scored_records:
        fallback = max(scored_records, key=lambda item: item["relevance_score"])
        fallback["filter_reason"] = "无证据达到阈值，保留最高相关项以避免空分析"
        kept = [fallback]
        filtered = [record for record in scored_records if record is not fallback]

    kept.sort(key=lambda item: (item["relevance_score"], item.get("score", 0)), reverse=True)
    return kept[:_MAX_RELEVANT_RECORDS], filtered


def _partition_relevant_records(
    scored_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not scored_records:
        return [], []

    ranked = sorted(
        scored_records,
        key=lambda item: (item.get("relevance_score", 0.0), item.get("score", 0.0)),
        reverse=True,
    )
    head_count = min(_MIN_RELEVANT_RECORDS, len(ranked))
    head_scores = [item.get("relevance_score", 0.0) for item in ranked[:head_count]]
    dynamic_floor = min(head_scores) if head_scores else 0.0
    top_score = ranked[0].get("relevance_score", 0.0)
    dynamic_floor = max(dynamic_floor, top_score * 0.55)

    kept = []
    filtered = []
    for position, record in enumerate(ranked):
        explicit_signal = _record_has_explicit_relevance_signal(record)
        retention_signal = (
            float(record.get("novelty_score", 0.0) or 0.0) >= 1.0
            and len(record.get("matched_segments", []) or []) >= 2
            and record.get("evidence_granularity") in {"segment_match", "full_page"}
        )
        diversity_signal = _record_adds_source_diversity(record, kept, top_score)
        baseline_signal = record.get("relevance_score", 0.0) >= max(0.22, top_score * 0.35)
        should_keep = (
            (position < _MIN_RELEVANT_RECORDS and baseline_signal)
            or explicit_signal
            or retention_signal
            or diversity_signal
            or record.get("relevance_score", 0.0) >= dynamic_floor
        )
        if should_keep and len(kept) < _MAX_RELEVANT_RECORDS:
            kept.append(record)
        else:
            filtered.append(record)
    return kept, filtered


def _record_adds_source_diversity(
    record: dict[str, Any],
    kept_records: list[dict[str, Any]],
    top_score: float,
) -> bool:
    profile = record.get("source_profile", {}) if isinstance(record.get("source_profile", {}), dict) else {}
    geo_group = str(profile.get("geo_group") or "unknown")
    country = str(profile.get("country") or "未知")
    if geo_group == "unknown" and country == "未知":
        return False

    relevance_score = float(record.get("relevance_score", 0.0) or 0.0)
    minimum_relevance = max(0.3, top_score * 0.42)
    if relevance_score < minimum_relevance:
        return False

    kept_geo_groups = {
        str((item.get("source_profile", {}) if isinstance(item.get("source_profile", {}), dict) else {}).get("geo_group") or "unknown")
        for item in kept_records
    }
    kept_countries = {
        str((item.get("source_profile", {}) if isinstance(item.get("source_profile", {}), dict) else {}).get("country") or "未知")
        for item in kept_records
    }
    adds_geo_group = geo_group != "unknown" and geo_group not in kept_geo_groups
    adds_country = country != "未知" and country not in kept_countries
    return adds_geo_group or adds_country


def _record_has_explicit_relevance_signal(record: dict[str, Any]) -> bool:
    joined_signals = " ".join(record.get("relevance_signals", []))
    return any(token in joined_signals for token in ["命中声明实体/主干锚点", "命中关键数字", "包含声明核心短语", "包含证据性表达"])


def _estimate_semantic_scores(claim_text: str, records: list[dict[str, Any]]) -> list[float]:
    candidate_texts = []
    for record in records:
        provenance = record.get("provenance", {})
        candidate_texts.append(
            " ".join(
                part for part in [
                    record.get("summary", ""),
                    record.get("reason", ""),
                    provenance.get("title", ""),
                    provenance.get("lead_excerpt", ""),
                    " ".join(provenance.get("quoted_sources", [])),
                ]
                if part
            )
        )
    try:
        scores = batch_semantic_similarity(claim_text, candidate_texts)
        return scores if len(scores) == len(records) else [0.0 for _ in records]
    except Exception:
        return [0.0 for _ in records]


def _estimate_relevance(
    claim_text: str,
    summary: str,
    reason: str,
    semantic_score: float,
) -> tuple[float, list[str]]:
    claim_anchor_tokens = _extract_anchor_tokens(claim_text)
    summary_anchor_tokens = _extract_anchor_tokens(f"{summary} {reason}")
    anchor_overlap = len(claim_anchor_tokens & summary_anchor_tokens) / len(claim_anchor_tokens) if claim_anchor_tokens else 0.0

    claim_numbers = _extract_numbers(claim_text)
    summary_numbers = _extract_numbers(f"{summary} {reason}")
    number_overlap = len(claim_numbers & summary_numbers) / len(claim_numbers) if claim_numbers else 0.0

    normalized_claim = _normalize_text(claim_text)
    normalized_summary = _normalize_text(f"{summary} {reason}")
    direct_hit = normalized_claim and normalized_claim in normalized_summary
    evidence_cue_hit = _has_evidence_cue(summary, reason)
    lexical_backoff = _lexical_backoff_score(claim_text, f"{summary} {reason}")

    score = (
        semantic_score * 0.55
        + anchor_overlap * 0.15
        + number_overlap * 0.15
        + lexical_backoff * 0.05
        + (0.05 if direct_hit else 0.0)
        + (0.05 if evidence_cue_hit else 0.0)
    )
    score = max(0.0, min(1.0, score))
    signals = []
    signals.append(f"语义相似度 {semantic_score:.3f}")
    if claim_anchor_tokens:
        if claim_anchor_tokens & summary_anchor_tokens:
            signals.append("命中声明实体/主干锚点")
        else:
            signals.append("未命中声明实体/主干锚点")
    if claim_numbers:
        if claim_numbers & summary_numbers:
            signals.append("命中关键数字")
        else:
            signals.append("未命中关键数字")
    if lexical_backoff > 0:
        signals.append(f"词汇回退匹配 {lexical_backoff:.3f}")
    if direct_hit:
        signals.append("包含声明核心短语")
    if evidence_cue_hit:
        signals.append("包含证据性表达")
    return score, signals


# ════════════════════════════════════════════════════
#  文本处理辅助
# ════════════════════════════════════════════════════

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _extract_semantic_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    latin_tokens = re.findall(r"[a-z0-9]{2,}", normalized)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    cjk_ngrams = []
    for size in (2, 3):
        cjk_ngrams.extend("".join(cjk_chars[index:index + size]) for index in range(len(cjk_chars) - size + 1))

    tokens = set()
    for token in latin_tokens + cjk_ngrams:
        if token in _RELEVANCE_STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.add(token)
    return tokens


def _lexical_backoff_score(claim_text: str, evidence_text: str) -> float:
    claim_tokens = _extract_semantic_tokens(claim_text)
    evidence_tokens = _extract_semantic_tokens(evidence_text)
    overlap = claim_tokens & evidence_tokens
    union = claim_tokens | evidence_tokens
    return len(overlap) / len(union) if union else 0.0


def _extract_anchor_tokens(text: str) -> set[str]:
    anchors = set()
    for token in _extract_semantic_tokens(text):
        if re.fullmatch(r"[a-z0-9]+", token):
            if len(token) >= 4:
                anchors.add(token)
            continue
        if len(token) >= 3:
            anchors.add(token)
    return anchors


def _has_evidence_cue(summary: str, reason: str) -> bool:
    combined = f"{summary} {reason}"
    return any(re.search(pattern, combined, flags=re.IGNORECASE) for pattern in _EVIDENCE_CUE_PATTERNS)


def _extract_source_clues(text: str) -> list[str]:
    clues = []
    for pattern in _SOURCE_CLUE_PATTERNS:
        for match in re.findall(pattern, text or "", flags=re.IGNORECASE):
            candidate = _normalize_text(match)
            if 2 <= len(candidate) <= 40 and candidate not in clues:
                clues.append(candidate)
    return clues[:5]


# ════════════════════════════════════════════════════
#  数字口径分析
# ════════════════════════════════════════════════════

def _infer_numeric_context(text: str, number: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return "general"
    if "%" in number:
        return "ratio"
    number_token = str(number).lower()
    number_index = normalized.find(number_token)
    window = normalized[max(0, number_index - 40): number_index + 60] if number_index >= 0 else normalized[:120]
    for label, keywords in _NUMERIC_CONTEXT_PATTERNS.items():
        if any(keyword in window for keyword in keywords):
            return label
    return "general"


def _normalize_numeric_value(number: str) -> float | None:
    cleaned = str(number or "").replace(",", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _numeric_label(label: str) -> str:
    return {
        "remaining": "剩余/可用数量",
        "active": "活跃/可操作数量",
        "destroyed": "被摧毁/失效数量",
        "total": "总量/初始数量",
        "ratio": "比例/百分比",
        "general": "其他数字口径",
    }.get(label, label)


def _analyze_numeric_consistency(claim_text: str, records: list[dict[str, Any]]) -> dict:
    claim_numbers = sorted(_extract_numbers(claim_text))
    clusters: dict[str, dict[str, Any]] = {}
    for record in records:
        provenance = record.get("provenance", {})
        texts = [
            record.get("summary", ""),
            record.get("snippet", ""),
            record.get("reason", ""),
            provenance.get("title", ""),
            provenance.get("lead_excerpt", ""),
            " ".join(record.get("matched_segments", [])),
        ]
        combined_text = " ".join(part for part in texts if part)
        extracted_numbers = record.get("extracted_numbers") or list(_extract_numbers(combined_text))
        for number in extracted_numbers:
            context_key = _infer_numeric_context(combined_text, str(number))
            cluster = clusters.setdefault(
                context_key,
                {
                    "context": context_key,
                    "context_label": _numeric_label(context_key),
                    "values": {},
                    "evidence_indices": set(),
                },
            )
            cluster["evidence_indices"].add(record.get("index"))
            value_bucket = cluster["values"].setdefault(
                str(number),
                {
                    "value": str(number),
                    "normalized_value": _normalize_numeric_value(str(number)),
                    "indices": [],
                    "examples": [],
                },
            )
            if record.get("index") not in value_bucket["indices"]:
                value_bucket["indices"].append(record.get("index"))
            snippet = _clip(combined_text, 180)
            if snippet and snippet not in value_bucket["examples"]:
                value_bucket["examples"].append(snippet)

    cluster_items = []
    inconsistent_contexts = []
    for cluster in clusters.values():
        values = list(cluster.get("values", {}).values())
        values.sort(key=lambda item: (len(item.get("indices", [])), str(item.get("value", ""))), reverse=True)
        unique_values = [item.get("value") for item in values]
        consistency = "consistent" if len(unique_values) <= 1 else "mixed"
        if consistency == "mixed":
            inconsistent_contexts.append(cluster.get("context_label", cluster.get("context", "")))
        cluster_items.append(
            {
                "context": cluster.get("context", "general"),
                "context_label": cluster.get("context_label", _numeric_label(cluster.get("context", "general"))),
                "consistency": consistency,
                "claim_numbers": claim_numbers,
                "evidence_indices": sorted(index for index in cluster.get("evidence_indices", set()) if index is not None),
                "values": values[:6],
            }
        )

    cluster_items.sort(key=lambda item: (0 if item.get("consistency") == "mixed" else 1, -len(item.get("evidence_indices", []))))
    if not cluster_items:
        return {
            "claim_numbers": claim_numbers,
            "clusters": [],
            "has_numeric_conflict": False,
            "summary": "未提取到可比较的数字口径。",
        }

    if inconsistent_contexts:
        summary = f"数字口径存在差异，主要集中在: {'、'.join(inconsistent_contexts[:3])}。"
    else:
        summary = "数字口径整体一致或未发现明显冲突。"
    return {
        "claim_numbers": claim_numbers,
        "clusters": cluster_items,
        "has_numeric_conflict": bool(inconsistent_contexts),
        "summary": summary,
    }


# ════════════════════════════════════════════════════
#  加权评分与来源多样性
# ════════════════════════════════════════════════════

def _calculate_weighted_score(
    source_profiles: list[dict],
    original_scores: list,
    independent_count: int,
    stance_analysis: dict,
    total_count: int,
    source_diversity: dict[str, Any] | None = None,
) -> float:
    if not source_profiles or not original_scores:
        return 5.0

    weighted_sum = 0.0
    weight_total = 0.0
    for i, profile in enumerate(source_profiles):
        if i >= len(original_scores):
            break
        score = original_scores[i] if isinstance(original_scores[i], (int, float)) else 5
        credibility = profile.get("credibility_score", 3)
        weight = credibility / 10.0
        weighted_sum += score * weight
        weight_total += weight

    base_score = weighted_sum / weight_total if weight_total > 0 else 5.0

    if total_count > 0:
        independence_ratio = independent_count / total_count
        if independence_ratio < 0.5:
            base_score = base_score * 0.7 + 5.0 * 0.3

    if stance_analysis.get("has_contradiction"):
        base_score = base_score * 0.6 + 5.0 * 0.4

    diversity_penalty = float((source_diversity or {}).get("score_penalty", 0.0) or 0.0)
    if diversity_penalty > 0:
        diversity_penalty = max(0.0, min(0.45, diversity_penalty))
        base_score = base_score * (1.0 - diversity_penalty) + 5.0 * diversity_penalty

    return max(0.0, min(10.0, base_score))


def _analyze_source_diversity(
    records: list[dict[str, Any]],
    dedup_groups: list[dict[str, Any]],
    stance_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not records:
        return {
            "countries": {},
            "geo_groups": {},
            "unique_country_count": 0,
            "unique_geo_group_count": 0,
            "has_concentration_risk": False,
            "score_penalty": 0.0,
            "summary": "",
        }

    representative_records = list(records)

    country_distribution: dict[str, int] = {}
    geo_group_distribution: dict[str, int] = {}
    for record in representative_records:
        profile = record.get("source_profile", {})
        country = str(profile.get("country") or "未知")
        geo_group = str(profile.get("geo_group") or "unknown")
        country_distribution[country] = country_distribution.get(country, 0) + 1
        geo_group_distribution[geo_group] = geo_group_distribution.get(geo_group, 0) + 1

    total = len(representative_records)
    dominant_country = max(country_distribution.items(), key=lambda item: item[1]) if country_distribution else ("未知", 0)
    dominant_geo_group = max(geo_group_distribution.items(), key=lambda item: item[1]) if geo_group_distribution else ("unknown", 0)
    dominant_country_ratio = dominant_country[1] / total if total else 0.0
    dominant_geo_group_ratio = dominant_geo_group[1] / total if total else 0.0

    stances = [
        str(item.get("stance", "neutral"))
        for item in (stance_analysis or {}).get("stances", [])
        if isinstance(item, dict) and item.get("stance") not in {None, "irrelevant"}
    ]
    distinct_stances = {stance for stance in stances if stance}
    aligned_stance = len(distinct_stances) == 1 and bool(distinct_stances)

    unique_country_count = len([country for country in country_distribution if country != "未知"])
    unique_geo_group_count = len([group for group in geo_group_distribution if group != "unknown"])
    has_concentration_risk = (
        total >= 3 and unique_country_count <= 1
        or total >= 4 and unique_geo_group_count <= 1
        or total >= 4 and dominant_geo_group_ratio >= 0.75 and aligned_stance
    )

    penalty = 0.0
    if has_concentration_risk:
        penalty = 0.12
        if unique_geo_group_count <= 1:
            penalty += 0.10
        if unique_country_count <= 1:
            penalty += 0.08
        if dominant_geo_group_ratio >= 0.75:
            penalty += 0.08
        if aligned_stance:
            penalty += 0.05
        penalty = min(0.38, penalty)

    country_distribution_readable = {
        country: count for country, count in sorted(country_distribution.items(), key=lambda item: (-item[1], item[0]))
    }
    geo_group_distribution_readable = {
        GEO_GROUP_LABELS.get(group, group): count
        for group, count in sorted(geo_group_distribution.items(), key=lambda item: (-item[1], item[0]))
    }

    if has_concentration_risk:
        summary = (
            f"来源多样性不足: 独立来源主要集中在{GEO_GROUP_LABELS.get(dominant_geo_group[0], dominant_geo_group[0])}"
            f"（{dominant_geo_group[1]}/{total}），国家分布为{_format_distribution(country_distribution_readable)}；"
            "这些来源可能共享相近地缘政治立场，已对交叉验证得分做降权。"
        )
    else:
        summary = (
            f"来源国家/阵营分布为{_format_distribution(country_distribution_readable)}，"
            "暂未发现明显的单一阵营集中风险。"
        )

    return {
        "countries": country_distribution,
        "countries_readable": country_distribution_readable,
        "geo_groups": geo_group_distribution,
        "geo_groups_readable": geo_group_distribution_readable,
        "unique_country_count": unique_country_count,
        "unique_geo_group_count": unique_geo_group_count,
        "dominant_country": dominant_country[0],
        "dominant_country_ratio": round(dominant_country_ratio, 3),
        "dominant_geo_group": dominant_geo_group[0],
        "dominant_geo_group_label": GEO_GROUP_LABELS.get(dominant_geo_group[0], dominant_geo_group[0]),
        "dominant_geo_group_ratio": round(dominant_geo_group_ratio, 3),
        "has_concentration_risk": has_concentration_risk,
        "aligned_stance": aligned_stance,
        "score_penalty": round(penalty, 3),
        "summary": summary,
    }


def _format_distribution(distribution: dict[str, int]) -> str:
    if not distribution:
        return "未知"
    return "、".join(f"{key}{value}条" for key, value in distribution.items())


# ════════════════════════════════════════════════════
#  立场分析与可追踪证据
# ════════════════════════════════════════════════════

def _build_simple_stance_analysis(claim_text: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """初始立场分析 - 使用 agent_stance_hint 作为初始值，待分析智能体后续覆写"""
    _HINT_MAP = {"support": "support", "deny": "deny", "mixed": "mixed", "neutral": "neutral"}
    stances = []
    support_count = 0
    deny_count = 0
    mixed_count = 0

    for record in records:
        idx = record.get("index")
        agent_hint = str(record.get("agent_stance_hint", "")).strip().lower()
        stance_value = _HINT_MAP.get(agent_hint, "neutral")

        stances.append({
            "index": idx,
            "stance": stance_value,
            "reason": str(record.get("stance_reason", ""))[:300],
            "source_description": str(record.get("source_name_hint", ""))[:100],
        })
        if stance_value == "support":
            support_count += 1
        elif stance_value == "deny":
            deny_count += 1
        elif stance_value == "mixed":
            mixed_count += 1

    has_contradiction = bool(mixed_count or (support_count and deny_count))
    if mixed_count and mixed_count >= max(support_count, deny_count):
        dominant_stance = "mixed"
    elif support_count > deny_count:
        dominant_stance = "support"
    elif deny_count > support_count:
        dominant_stance = "deny"
    else:
        dominant_stance = "neutral"

    return {
        "stances": stances,
        "has_contradiction": has_contradiction,
        "dominant_stance": dominant_stance,
        "contradiction_detail": "证据中存在支持与反驳口径" if has_contradiction else ""
    }


def _build_simple_traceable_evidence(
    records: list[dict[str, Any]],
    stance_analysis: dict[str, Any]
) -> list[dict[str, Any]]:
    """简化的可追踪证据构建"""
    stance_lookup = {item.get("index"): item for item in stance_analysis.get("stances", [])}
    items = []

    for record in sorted(records, key=lambda item: item.get("index", 0)):
        profile = record.get("source_profile", {})
        provenance = record.get("provenance", {})
        record_index = record.get("index")
        stance = stance_lookup.get(record_index, {"stance": "neutral", "reason": "", "source_description": ""})

        record_stance_reason = stance.get("reason", "") or record.get("stance_reason", "")
        record_source_desc = stance.get("source_description", "") or record.get("source_name_hint", "") or profile.get("name", "")

        items.append({
            "index": record_index,
            "url": record.get("url", ""),
            "source_name": profile.get("name", "未知来源"),
            "domain": profile.get("domain", ""),
            "country": profile.get("country", "未知"),
            "tier": profile.get("tier", "unknown"),
            "tier_label": profile.get("tier_label", TIER_LABELS.get("unknown", "未知")),
            "display_status": "独立来源",
            "relation": "independent",
            "title": record.get("title", "") or provenance.get("title", ""),
            "summary": record.get("summary", ""),
            "publish_time": provenance.get("publish_time", ""),
            "score": float(record.get("score", 0) or 0),
            "stance": stance.get("stance", "neutral"),
            "stance_label": _stance_label(stance.get("stance", "neutral")),
            "stance_reason": record_stance_reason,
            "source_description": record_source_desc,
        })

    return items


def _stance_label(stance: str) -> str:
    return {
        "support": "✅支持",
        "deny": "❌否认",
        "mixed": "⚖️双向",
        "neutral": "➖中立",
        "irrelevant": "⬜无关",
    }.get(stance or "neutral", "➖中立")


def _empty_result(message: str) -> dict:
    """无证据时的空结果"""
    return {
        "source_profiles": [],
        "tier_distribution": {},
        "tier_distribution_readable": {},
        "independent_source_count": 0,
        "total_source_count": 0,
        "dedup_groups": [],
        "traceable_evidence": [],
        "stance_analysis": {"has_contradiction": False, "dominant_stance": "unknown", "stances": []},
        "source_diversity": {
            "countries": {},
            "countries_readable": {},
            "geo_groups": {},
            "geo_groups_readable": {},
            "unique_country_count": 0,
            "unique_geo_group_count": 0,
            "has_concentration_risk": False,
            "score_penalty": 0.0,
            "summary": "",
        },
        "numeric_analysis": {"claim_numbers": [], "clusters": [], "has_numeric_conflict": False, "summary": ""},
        "cross_verify_score": 5.0,
        "cross_verify_summary": message,
        "has_contradiction": False,
        "relevant_source_count": 0,
        "filtered_out_count": 0,
        "filtered_out_samples": [],
    }
