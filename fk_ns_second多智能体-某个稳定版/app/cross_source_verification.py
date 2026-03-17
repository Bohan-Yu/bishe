"""
多源交叉验证子 Agent（Cross-Source Verification Agent）

创新点:
  1. 信源分层: 基于信誉度知识库，将搜索到的来源按 官方/主流/专业/门户/自媒体 分级
  2. 溯源去重: 检测多条搜索结果是否实质引用同一原始报道，消除"假性多源验证"
  3. 立场对比: 检测不同层级来源是否给出矛盾结论，标记矛盾信号

流程:
    输入: 搜索证据列表 + 原始声明
    处理: 信源分层 → 混合语义相关性筛选 → LLM 溯源去重 → LLM 立场分析
    输出: 交叉验证报告（信源分布、独立源数、矛盾检测、加权可信度）
"""

import json
import hashlib
import re
from itertools import combinations
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.config import TAVILY_API_KEY, ZHIPU_API_KEY, ZHIPU_BASE_URL, TEXT_MODEL
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

# ────────────── LLM 实例 ──────────────
_cross_verify_llm = ChatOpenAI(
    model=TEXT_MODEL,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    temperature=0.1,
    max_tokens=10000,
    timeout=120,
)

_cross_verify_extract_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

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

_MAX_PROVENANCE_URLS = 12
_MAX_RELEVANT_RECORDS = 12
_MIN_RELEVANT_RECORDS = 4
_MAX_PAIR_CANDIDATES = 30

_ATTRIBUTION_CUE_TERMS = {
    "据", "援引", "引述", "提到", "表示", "称", "指出", "通报", "声明", "发布",
    "according to", "citing", "reported by", "quoted", "said", "says", "announced", "statement",
}

_CHINESE_SOURCE_SUFFIXES = (
    "网", "报", "台", "社", "部", "局", "厅", "院", "会", "署", "办", "政府", "法院",
    "委员会", "研究院", "研究所", "实验室", "大学", "学院", "智库", "集团", "公司", "平台",
    "中心", "机构", "组织",
)

_LATIN_SOURCE_KEYWORDS = {
    "agency", "news", "times", "post", "journal", "review", "daily", "telegraph", "reuters",
    "associated", "press", "ministry", "department", "office", "government", "court", "university",
    "institute", "commission", "committee", "laboratory", "lab", "center", "centre", "media",
}

_PUBLISH_TIME_PATTERNS = [
    r"(20\d{2}[/-]\d{1,2}[/-]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?)",
    r"(20\d{2}年\d{1,2}月\d{1,2}日(?:\s*\d{1,2}:\d{2}(?::\d{2})?)?)",
    r"(published\s+[A-Z][a-z]+\s+\d{1,2},\s+20\d{2})",
    r"(updated\s+[A-Z][a-z]+\s+\d{1,2},\s+20\d{2})",
]

_SOURCE_PREFIX_PATTERNS = [
    r"^(?:据|援引|引述|根据|来自)",
    r"^(?:according to|reported by|citing)\s+",
]

_SOURCE_SUFFIX_PATTERNS = [
    r"(?:报道|消息|称|表示|指出|通报|声明|发布)$",
    r"(?:report|reports|reported|said|says|statement)$",
]

_LOW_AUTHORITY_ACCOUNT_MARKERS = {
    "博主", "军事博主", "自媒体", "个人账号", "个人主页", "专栏作者", "作者自述",
    "头条号", "百家号", "企鹅号", "网易号", "搜狐号", "个人认证", "creator",
}

_STANCE_SUPPORT_MARKERS = {
    "仅剩", "剩余", "所剩无几", "发射器减少", "摧毁了", "摧毁超90%", "摧毁90%以上",
    "destroyed", "running out", "only about 100", "launchers left",
}

_STANCE_DENY_MARKERS = {
    "纯属军事宣传", "没有客观证据", "无客观证据", "谎称", "否认", "驳斥", "反驳",
    "更强大、数量更多", "数量更多", "prepared many surprises", "military propaganda",
    "no objective evidence", "more powerful, more numerous", "more missiles",
}

_STANCE_STRONG_DENY_MARKERS = {
    "纯属军事宣传", "没有客观证据", "无客观证据", "military propaganda", "no objective evidence",
}

_STANCE_MIXED_BRIDGE_MARKERS = {
    "回应", "同时", "随后", "此前报道", "另一边", "并称", "并指责", "而伊朗", "特朗普", "伊朗方面",
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

    参数:
        claim_text:    被验证的声明文本
        evidence_list: 证据摘要列表
        web_list:      证据 URL 列表
        scores:        原始证据评分列表
        reasons:       原始评分理由列表

    返回:
    {
        "source_profiles": [...],       # 每条来源的信誉画像
        "tier_distribution": {...},     # 按层级分布统计
        "independent_source_count": 2,  # 真正独立的信息源数量
        "dedup_groups": [...],          # 溯源去重分组
        "stance_analysis": {...},       # 立场分析
        "cross_verify_score": 7.5,      # 交叉验证加权得分
        "cross_verify_summary": "..."   # 交叉验证结论
    }
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

    # 统计层级分布
    tier_distribution = {}
    for sp in source_profiles:
        tier = sp.get("tier", "unknown")
        tier_distribution.setdefault(tier, 0)
        tier_distribution[tier] += 1

    # ═══════════ Step 2: 混合语义相关性筛选 ═══════════
    relevant_records, filtered_out_records = _filter_relevant_records(claim_text, deduped_records)

    # ═══════════ Step 3: 网页全文抽取 + 溯源特征标准化 ═══════════
    _apply_agent_catalog_hints(relevant_records, agent_evidence_catalog)
    _attach_provenance_signals(claim_text, relevant_records)
    _apply_record_level_source_overrides(relevant_records)
    same_source_candidates = _merge_same_source_candidates(
        _build_agent_same_source_candidates(relevant_records, agent_evidence_catalog),
        _build_same_source_candidates(relevant_records),
    )
    fallback_dedup_groups = _build_rule_based_dedup_groups(relevant_records, same_source_candidates)

    # ═══════════ Step 4: 仅对高相关证据做 LLM 独立来源 + 立场分析 ═══════════
    evidence_info = []
    for record in relevant_records:
        profile = record.get("source_profile", {})
        provenance = record.get("provenance", {})
        evidence_info.append({
            "index": record["index"],
            "summary": record["summary"],
            "url": record["url"],
            "normalized_url": record["normalized_url"],
            "domain": profile.get("domain", ""),
            "source_name": profile.get("name", "未知"),
            "country": profile.get("country", "未知"),
            "tier": profile.get("tier_label", "未知"),
            "credibility": profile.get("credibility_score", 3),
            "original_score": record.get("score", 5),
            "score_reason": record.get("reason", ""),
            "relevance_score": round(record.get("relevance_score", 0.0), 3),
            "relevance_signals": record.get("relevance_signals", []),
            "source_clues": record.get("source_clues", []),
            "duplicate_indices": record.get("duplicate_indices", []),
            "title": provenance.get("title", ""),
            "lead_excerpt": provenance.get("lead_excerpt", ""),
            "publish_time": provenance.get("publish_time", ""),
            "quoted_sources": provenance.get("quoted_sources", []),
            "source_entities": provenance.get("source_entities", []),
            "attribution_sentences": provenance.get("attribution_sentences", []),
            "provenance_score": round(provenance.get("provenance_score", 0.0), 3),
        })

    analysis_result = _llm_cross_analysis(
        claim_text,
        evidence_info,
        same_source_candidates,
        fallback_dedup_groups,
        agent_evidence_catalog=agent_evidence_catalog,
    )

    # ═══════════ 整合结果 ═══════════
    dedup_groups = _finalize_dedup_groups(
        relevant_records,
        analysis_result.get("dedup_groups", []) or fallback_dedup_groups,
    )
    independent_count = len(dedup_groups)
    stance_analysis = _refine_stance_analysis(relevant_records, analysis_result.get("stance_analysis", {}))
    source_diversity = _analyze_source_diversity(relevant_records, dedup_groups, stance_analysis)
    traceable_evidence = _build_traceable_evidence(relevant_records, dedup_groups, stance_analysis)
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
    if analysis_result.get("summary"):
        summary_parts.append(analysis_result.get("summary", ""))
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
        "dedup_groups": dedup_groups,
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
                    "duplicate_indices": [index + 1],
                    "search_queries": item.get("search_queries", [])[:6],
                    "origin_tools": item.get("origin_tools", [])[:4],
                    "evidence_granularity": str(item.get("evidence_granularity", "search_result"))[:40],
                    "observation_count": int(item.get("observation_count", 1) or 1),
                    "novelty_score": float(item.get("novelty_score", 0.0) or 0.0),
                    "agent_stance_hint": str(item.get("agent_stance_hint", ""))[:20],
                    "agent_source_role": str(item.get("agent_source_role", ""))[:30],
                    "agent_originality_hint": str(item.get("agent_originality_hint", ""))[:20],
                    "agent_same_source_group": str(item.get("agent_same_source_group", ""))[:80],
                    "agent_note": str(item.get("agent_note", ""))[:300],
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
                "duplicate_indices": [index + 1],
                "search_queries": [],
                "origin_tools": [],
                "evidence_granularity": "search_result",
                "observation_count": 1,
                "novelty_score": 0.0,
                "agent_stance_hint": "",
                "agent_source_role": "",
                "agent_originality_hint": "",
                "agent_same_source_group": "",
                "agent_note": "",
            }
        )
    return records


def _adaptive_cap(total: int, minimum: int, multiplier: float, hard_cap: int) -> int:
    if total <= 0:
        return minimum
    estimate = int(total * multiplier)
    return max(minimum, min(hard_cap, estimate))


def _apply_agent_catalog_hints(records: list[dict[str, Any]], agent_evidence_catalog: dict | None) -> None:
    if not records or not isinstance(agent_evidence_catalog, dict):
        return

    note_by_url = {
        note.get("url"): note
        for note in agent_evidence_catalog.get("evidence_notes", [])
        if isinstance(note, dict) and note.get("url")
    }
    for record in records:
        note = note_by_url.get(record.get("url", ""), {})
        if not note:
            continue
        record["agent_hint"] = {
            "stance_hint": note.get("stance_hint", ""),
            "source_role": note.get("source_role", ""),
            "originality_hint": note.get("originality_hint", ""),
            "same_source_group": note.get("same_source_group", ""),
            "reason": note.get("reason", ""),
        }
        if note.get("same_source_group"):
            record["agent_same_source_group"] = str(note.get("same_source_group", ""))[:80]
        merged_clues = list(record.get("source_clues", []))
        for candidate in [note.get("reason", ""), note.get("source_role", "")]:
            normalized = _normalize_text(candidate)
            if 2 <= len(normalized) <= 60 and normalized not in merged_clues:
                merged_clues.append(normalized)
        record["source_clues"] = merged_clues[:10]


_NUMERIC_CONTEXT_PATTERNS = {
    "remaining": ["仅剩", "剩余", "remaining", "left", "still has", "available", "可用", "可正常", "仍有"],
    "active": ["活跃", "active", "operational", "可发射", "运行", "deployable"],
    "destroyed": ["摧毁", "destroyed", "taken out", "neutralized", "中和", "失效", "损失"],
    "total": ["总计", "总数", "initially", "originally", "最初", "起初", "总共有", "before"],
    "ratio": ["%", "百分之", "比例", "ratio", "share"],
}


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
        existing["duplicate_indices"] = existing.get("duplicate_indices", []) + record.get("duplicate_indices", [])
        if record.get("score", 0) > existing.get("score", 0):
            existing["summary"] = record.get("summary", existing.get("summary", ""))
            existing["score"] = record.get("score", existing.get("score", 5))
            existing["reason"] = record.get("reason", existing.get("reason", ""))
        for clue in record.get("source_clues", []):
            if clue not in existing["source_clues"]:
                existing["source_clues"].append(clue)
    return [merged[key] for key in ordered_keys]


def _attach_provenance_signals(query_text: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    extracted_map: dict[str, dict[str, str]] = {}
    urls = [record.get("url", "") for record in records if record.get("url")]
    unique_urls = []
    seen_urls = set()
    for url in urls:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        unique_urls.append(url)

    if _cross_verify_extract_client and unique_urls:
        try:
            extract_limit = _adaptive_cap(len(unique_urls), minimum=4, multiplier=1.0, hard_cap=_MAX_PROVENANCE_URLS)
            extract_result = _cross_verify_extract_client.extract(
                urls=unique_urls[:extract_limit],
                query=query_text,
            )
            for item in extract_result.get("results", []):
                extracted_map[item.get("url", "")] = {
                    "title": item.get("title", ""),
                    "raw_content": item.get("raw_content", ""),
                }
        except Exception:
            extracted_map = {}

    for record in records:
        extracted = extracted_map.get(record.get("url", ""), {})
        provenance = _build_provenance_profile(record, extracted)
        record["provenance"] = provenance
        merged_clues = list(record.get("source_clues", []))
        for clue in provenance.get("quoted_sources", []) + provenance.get("source_entities", []):
            normalized_clue = _normalize_text(clue)
            if 2 <= len(normalized_clue) <= 40 and normalized_clue not in merged_clues:
                merged_clues.append(normalized_clue)
        record["source_clues"] = merged_clues[:8]


def _record_text_blob(record: dict[str, Any]) -> str:
    provenance = record.get("provenance", {})
    parts = [
        record.get("title", ""),
        record.get("summary", ""),
        record.get("snippet", ""),
        " ".join(record.get("matched_segments", [])[:4]),
        provenance.get("title", ""),
        provenance.get("lead_excerpt", ""),
        " ".join(provenance.get("attribution_sentences", [])[:3]),
    ]
    return " ".join(str(part or "") for part in parts if str(part or "").strip())


def _apply_record_level_source_overrides(records: list[dict[str, Any]]) -> None:
    for record in records:
        profile = record.get("source_profile", {})
        if not profile:
            continue
        text_blob = _record_text_blob(record)
        lowered = text_blob.lower()
        if any(marker.lower() in lowered for marker in _LOW_AUTHORITY_ACCOUNT_MARKERS):
            profile["tier"] = "self_media"
            profile["tier_label"] = TIER_LABELS.get("self_media", "📱 自媒体")
            profile["credibility_score"] = TIER_SCORES.get("self_media", 2)
            profile["record_level_override"] = "author_account"
            profile["record_level_reason"] = "页面含博主/自媒体/个人账号等标记，按低权威账号页降权。"


def _heuristic_record_stance(record: dict[str, Any]) -> dict[str, Any]:
    text_blob = _record_text_blob(record)
    lowered = text_blob.lower()
    provenance = record.get("provenance", {})
    support_hits = sorted(marker for marker in _STANCE_SUPPORT_MARKERS if marker.lower() in lowered)
    deny_hits = sorted(marker for marker in _STANCE_DENY_MARKERS if marker.lower() in lowered)
    strong_deny_hits = sorted(marker for marker in _STANCE_STRONG_DENY_MARKERS if marker.lower() in lowered)
    mixed_bridge_hits = sorted(marker for marker in _STANCE_MIXED_BRIDGE_MARKERS if marker.lower() in lowered)
    multi_source_signals = int(len(provenance.get("quoted_sources", []) or []) >= 2) + int(len(provenance.get("attribution_sentences", []) or []) >= 2)

    if strong_deny_hits:
        return {
            "stance": "deny",
            "reason": f"页面存在强反驳表述，如{', '.join(strong_deny_hits[:2])}。",
            "confidence": 0.96,
        }

    if support_hits and deny_hits:
        if mixed_bridge_hits or multi_source_signals:
            return {
                "stance": "mixed",
                "reason": (
                    f"页面同时出现支持口径({', '.join(support_hits[:2])})与反驳口径({', '.join(deny_hits[:2])})，"
                    f"且有双方并列/回应信号({', '.join(mixed_bridge_hits[:2]) or '多方归因'})。"
                ),
                "confidence": 0.95,
            }
        return {
            "stance": "deny",
            "reason": f"页面虽转述原说法，但核心表述是否认/反驳，如{', '.join(deny_hits[:2])}。",
            "confidence": 0.9,
        }
    if deny_hits:
        return {
            "stance": "deny",
            "reason": f"页面存在明确反驳/否认证据，如{', '.join(deny_hits[:2])}。",
            "confidence": 0.88,
        }
    if support_hits:
        return {
            "stance": "support",
            "reason": f"页面存在支持声明的表述，如{', '.join(support_hits[:2])}。",
            "confidence": 0.82,
        }
    return {"stance": "", "reason": "", "confidence": 0.0}


def _refine_stance_analysis(
    records: list[dict[str, Any]],
    stance_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    analysis = dict(stance_analysis or {})
    raw_stances = analysis.get("stances", [])
    stance_lookup = {}
    if isinstance(raw_stances, list):
        for item in raw_stances:
            if isinstance(item, dict) and isinstance(item.get("index"), int):
                stance_lookup[item["index"]] = dict(item)

    finalized: list[dict[str, Any]] = []
    support_count = 0
    deny_count = 0
    mixed_count = 0

    for record in sorted(records, key=lambda item: item.get("index", 0)):
        index = record.get("index")
        existing = stance_lookup.get(index, {"index": index, "stance": "neutral", "reason": ""})
        heuristic = _heuristic_record_stance(record)
        current_stance = str(existing.get("stance", "neutral") or "neutral")
        provenance = record.get("provenance", {})
        record_text = _record_text_blob(record).lower()
        has_multi_source_context = (
            len(provenance.get("quoted_sources", []) or []) >= 2
            or len(provenance.get("attribution_sentences", []) or []) >= 2
            or len(provenance.get("source_entities", []) or []) >= 2
        )
        has_strong_deny_phrase = any(marker.lower() in record_text for marker in _STANCE_STRONG_DENY_MARKERS)
        if has_strong_deny_phrase and not has_multi_source_context:
            existing["stance"] = "deny"
            existing["reason"] = "页面含明确反驳表述，且未呈现双方并列口径。"
            heuristic = {"stance": "deny", "reason": existing["reason"], "confidence": 0.99}
            current_stance = "deny"

        if heuristic.get("stance") == "mixed":
            existing["stance"] = "mixed"
            if heuristic.get("reason"):
                existing["reason"] = heuristic["reason"]
        elif heuristic.get("stance") and current_stance in {"neutral", "irrelevant", ""}:
            existing["stance"] = heuristic["stance"]
            if heuristic.get("reason"):
                existing["reason"] = heuristic["reason"]
        elif heuristic.get("stance") and heuristic.get("stance") != current_stance:
            if has_multi_source_context and current_stance in {"support", "deny"}:
                existing["stance"] = "mixed"
                merged_reason = "；".join(
                    part for part in [existing.get("reason", ""), heuristic.get("reason", "")]
                    if str(part or "").strip()
                )
                existing["reason"] = merged_reason[:300]
            elif float(heuristic.get("confidence", 0.0) or 0.0) >= 0.88:
                existing["stance"] = heuristic["stance"]
                if heuristic.get("reason"):
                    existing["reason"] = heuristic["reason"]
            else:
                existing["stance"] = "mixed"
                merged_reason = "；".join(
                    part for part in [existing.get("reason", ""), heuristic.get("reason", "")]
                    if str(part or "").strip()
                )
                existing["reason"] = merged_reason[:300]

        stance_value = str(existing.get("stance", "neutral") or "neutral")
        if stance_value == "support":
            support_count += 1
        elif stance_value == "deny":
            deny_count += 1
        elif stance_value == "mixed":
            mixed_count += 1
        finalized.append(existing)

    analysis["stances"] = finalized
    has_contradiction = bool(mixed_count or (support_count and deny_count))
    analysis["has_contradiction"] = has_contradiction

    if mixed_count and mixed_count >= max(support_count, deny_count):
        analysis["dominant_stance"] = "mixed"
    elif support_count > deny_count:
        analysis["dominant_stance"] = "support"
    elif deny_count > support_count:
        analysis["dominant_stance"] = "deny"
    else:
        analysis["dominant_stance"] = analysis.get("dominant_stance") or "neutral"

    if has_contradiction and not analysis.get("contradiction_detail"):
        analysis["contradiction_detail"] = "证据中同时存在支持与反驳口径，或单篇页面并列展示双方互相冲突的说法。"
    return analysis


def _build_provenance_profile(record: dict[str, Any], extracted: dict[str, str]) -> dict[str, Any]:
    title = _clip_text(extracted.get("title", ""), 300)
    raw_content = _normalize_extracted_content(extracted.get("raw_content", ""))
    lead_excerpt = _extract_lead_excerpt(raw_content, record.get("summary", ""), title)
    content_window = _clip_text(raw_content, 5000)
    publish_time = _extract_publish_time(content_window)
    source_entities = _extract_source_entities("\n".join([title, lead_excerpt, content_window]))
    attribution_blocks = _extract_attribution_blocks(content_window)
    quoted_sources = _merge_ranked_sources(
        [source for block in attribution_blocks for source in block.get("sources", [])] + source_entities
    )
    fingerprint = _build_provenance_fingerprint(title, lead_excerpt, quoted_sources)
    provenance_score = _estimate_provenance_quality(title, lead_excerpt, publish_time, attribution_blocks, raw_content)

    return {
        "title": title,
        "lead_excerpt": lead_excerpt,
        "publish_time": publish_time,
        "source_entities": source_entities[:8],
        "quoted_sources": quoted_sources[:8],
        "attribution_sentences": [block.get("text", "") for block in attribution_blocks[:4]],
        "fingerprint": fingerprint,
        "content_hash": hashlib.md5(fingerprint.encode("utf-8")).hexdigest() if fingerprint else "",
        "provenance_score": provenance_score,
    }


def _normalize_extracted_content(text: str) -> str:
    cleaned = re.sub(r"\r\n?", "\n", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _clip_text(text: Any, limit: int = 5000) -> str:
    value = "" if text is None else str(text)
    return value[:limit]


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for chunk in re.split(r"\n+", text or ""):
        normalized = re.sub(r"\s+", " ", chunk).strip()
        if len(normalized) < 20:
            continue
        paragraphs.append(normalized)
    return paragraphs


def _extract_lead_excerpt(raw_content: str, fallback_summary: str, title: str = "") -> str:
    paragraphs = _split_paragraphs(raw_content)
    kept: list[str] = []
    normalized_title = _normalize_text(title)
    for paragraph in paragraphs:
        normalized_paragraph = _normalize_text(paragraph)
        if normalized_title and normalized_paragraph == normalized_title:
            continue
        kept.append(paragraph)
        if len(" ".join(kept)) >= 500 or len(kept) >= 3:
            break
    lead = " ".join(kept).strip()
    return lead or _clip_text(fallback_summary, 500)


def _extract_publish_time(text: str) -> str:
    for pattern in _PUBLISH_TIME_PATTERNS:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return _normalize_text(match.group(1))[:40]
    return ""


def _extract_source_entities(text: str) -> list[str]:
    candidates: list[str] = []
    suffix_pattern = "|".join(sorted(_CHINESE_SOURCE_SUFFIXES, key=len, reverse=True))
    chinese_matches = re.findall(
        rf"[\u4e00-\u9fffA-Za-z0-9·（）()\-]{{2,30}}(?:{suffix_pattern})",
        text or "",
    )
    latin_matches = re.findall(
        r"\b[A-Z][A-Za-z&.\-]{1,}(?:\s+[A-Z][A-Za-z&.\-]{1,}){0,4}\b",
        text or "",
    )
    acronym_matches = re.findall(r"\b[A-Z]{2,10}\b", text or "")

    for raw_candidate in chinese_matches + latin_matches + acronym_matches:
        candidate = _normalize_source_name(raw_candidate)
        if not _is_meaningful_source_name(candidate):
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates[:12]


def _normalize_source_name(name: str) -> str:
    cleaned = re.sub(r"^[\s\-,:;，。：；]+|[\s\-,:;，。：；]+$", "", name or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    for pattern in _SOURCE_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    for pattern in _SOURCE_SUFFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        cleaned = re.sub(r"(社|网|台)报$", r"\1", cleaned)
    cleaned = re.sub(r"^[\s\-,:;，。：；]+|[\s\-,:;，。：；]+$", "", cleaned)
    return cleaned.strip()


def _is_meaningful_source_name(name: str) -> bool:
    if len(name) < 2:
        return False
    lowered = name.lower()
    if lowered in {"image", "video", "share", "login", "menu"}:
        return False
    if re.fullmatch(r"\d+", name):
        return False
    if re.search(r"[\u4e00-\u9fff]", name):
        return any(name.endswith(suffix) for suffix in _CHINESE_SOURCE_SUFFIXES) or len(name) <= 8
    return any(keyword in lowered for keyword in _LATIN_SOURCE_KEYWORDS) or len(name.split()) >= 2 or name.isupper()


def _extract_attribution_blocks(text: str) -> list[dict[str, Any]]:
    sentences = re.split(r"(?<=[。！？!?\.])\s+|\n", text or "")
    blocks: list[dict[str, Any]] = []
    for sentence in sentences:
        normalized_sentence = re.sub(r"\s+", " ", sentence).strip()
        if len(normalized_sentence) < 20:
            continue
        lowered = normalized_sentence.lower()
        if not any(term in lowered or term in normalized_sentence for term in _ATTRIBUTION_CUE_TERMS):
            continue
        sources = _extract_source_entities(normalized_sentence)
        if not sources:
            continue
        blocks.append(
            {
                "text": _clip_text(normalized_sentence, 220),
                "sources": sources[:3],
            }
        )
    return blocks[:8]


def _merge_ranked_sources(sources: list[str]) -> list[str]:
    ranked: list[str] = []
    for source in sources:
        normalized = _normalize_source_name(source)
        if not _is_meaningful_source_name(normalized):
            continue
        if normalized not in ranked:
            ranked.append(normalized)
    return ranked


def _build_provenance_fingerprint(title: str, lead_excerpt: str, quoted_sources: list[str]) -> str:
    parts = [
        _normalize_text(title),
        _normalize_text(lead_excerpt),
        " ".join(_normalize_text(item) for item in quoted_sources[:5]),
    ]
    return " | ".join(part for part in parts if part)


def _estimate_provenance_quality(
    title: str,
    lead_excerpt: str,
    publish_time: str,
    attribution_blocks: list[dict[str, Any]],
    raw_content: str,
) -> float:
    score = 0.0
    if title:
        score += 0.2
    if lead_excerpt:
        score += 0.3
    if publish_time:
        score += 0.15
    if attribution_blocks:
        score += 0.2
    if len(raw_content) >= 500:
        score += 0.15
    return min(1.0, score)


def _build_same_source_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for left, right in combinations(records, 2):
        left_profile = left.get("provenance", {})
        right_profile = right.get("provenance", {})

        url_exact = 1.0 if left.get("normalized_url") == right.get("normalized_url") else 0.0
        title_similarity = _hybrid_text_similarity(left_profile.get("title", ""), right_profile.get("title", ""))
        lead_similarity = _hybrid_text_similarity(left_profile.get("lead_excerpt", ""), right_profile.get("lead_excerpt", ""))
        summary_similarity = _hybrid_text_similarity(left.get("summary", ""), right.get("summary", ""))
        attribution_overlap = _overlap_ratio(
            set(left_profile.get("quoted_sources", [])),
            set(right_profile.get("quoted_sources", [])),
        )
        entity_overlap = _overlap_ratio(
            set(left_profile.get("source_entities", [])),
            set(right_profile.get("source_entities", [])),
        )
        time_match = _time_match_score(left_profile.get("publish_time", ""), right_profile.get("publish_time", ""))
        domain_match = 1.0 if _extract_domain_key(left) == _extract_domain_key(right) else 0.0

        same_source_score = min(
            1.0,
            url_exact * 0.30
            + title_similarity * 0.18
            + lead_similarity * 0.18
            + summary_similarity * 0.10
            + attribution_overlap * 0.10
            + entity_overlap * 0.07
            + time_match * 0.05
            + domain_match * 0.02,
        )

        reasons = []
        if url_exact:
            reasons.append("规范化 URL 完全一致")
        if title_similarity >= 0.8:
            reasons.append(f"标题高度相似 {title_similarity:.2f}")
        if lead_similarity >= 0.75:
            reasons.append(f"导语高度相似 {lead_similarity:.2f}")
        if attribution_overlap >= 0.5:
            reasons.append("共享关键归因来源")
        if entity_overlap >= 0.5:
            reasons.append("共享主要机构/媒体实体")
        if time_match:
            reasons.append("发布时间口径接近")

        strong_signal_count = sum(
            1 for flag in [
                url_exact == 1.0,
                title_similarity >= 0.88 and lead_similarity >= 0.70,
                attribution_overlap >= 0.60,
                entity_overlap >= 0.65,
                time_match == 1.0 and title_similarity >= 0.75,
            ] if flag
        )
        likely_same = (
            url_exact == 1.0
            or strong_signal_count >= 2
            or (strong_signal_count >= 1 and same_source_score >= 0.58)
            or same_source_score >= 0.78
        )
        if not likely_same:
            continue

        candidates.append(
            {
                "left_index": left.get("index"),
                "right_index": right.get("index"),
                "left_title": left_profile.get("title", ""),
                "right_title": right_profile.get("title", ""),
                "same_source_score": round(same_source_score, 3),
                "title_similarity": round(title_similarity, 3),
                "lead_similarity": round(lead_similarity, 3),
                "attribution_overlap": round(attribution_overlap, 3),
                "entity_overlap": round(entity_overlap, 3),
                "domain_match": bool(domain_match),
                "time_match": bool(time_match),
                "likely_same_source": url_exact == 1.0 or strong_signal_count >= 2 or same_source_score >= 0.78,
                "reason_signals": reasons[:4],
            }
        )

    candidates.sort(key=lambda item: item.get("same_source_score", 0.0), reverse=True)
    return candidates[:_adaptive_cap(len(records), minimum=8, multiplier=3.0, hard_cap=_MAX_PAIR_CANDIDATES)]


def _build_agent_same_source_candidates(
    records: list[dict[str, Any]],
    agent_evidence_catalog: dict | None,
) -> list[dict[str, Any]]:
    if not records or not isinstance(agent_evidence_catalog, dict):
        return []

    url_to_record = {record.get("url", ""): record for record in records if record.get("url")}
    pair_map: dict[tuple[int, int], dict[str, Any]] = {}

    def register_pair(left_url: str, right_url: str, score: float, reason: str, likely_same: bool) -> None:
        left = url_to_record.get(left_url)
        right = url_to_record.get(right_url)
        if not left or not right or left.get("index") == right.get("index"):
            return
        key = tuple(sorted((left.get("index"), right.get("index"))))
        existing = pair_map.get(key)
        candidate = {
            "left_index": key[0],
            "right_index": key[1],
            "left_title": (left.get("provenance", {}) or {}).get("title", left.get("title", "")),
            "right_title": (right.get("provenance", {}) or {}).get("title", right.get("title", "")),
            "same_source_score": round(score, 3),
            "title_similarity": 0.0,
            "lead_similarity": 0.0,
            "attribution_overlap": 0.0,
            "entity_overlap": 0.0,
            "domain_match": _extract_domain_key(left) == _extract_domain_key(right),
            "time_match": False,
            "likely_same_source": likely_same,
            "reason_signals": [reason],
            "agent_hint": True,
        }
        if existing:
            existing["same_source_score"] = max(existing.get("same_source_score", 0.0), candidate["same_source_score"])
            existing["likely_same_source"] = existing.get("likely_same_source", False) or likely_same
            existing["reason_signals"] = list(dict.fromkeys(existing.get("reason_signals", []) + [reason]))[:4]
            return
        pair_map[key] = candidate

    for group in agent_evidence_catalog.get("group_hints", []):
        if not isinstance(group, dict):
            continue
        urls = [url for url in group.get("urls", []) if url in url_to_record]
        confidence = str(group.get("confidence", "medium")).lower()
        score = 0.9 if confidence == "high" else 0.72
        likely_same = confidence == "high"
        for left_index in range(len(urls)):
            for right_index in range(left_index + 1, len(urls)):
                register_pair(urls[left_index], urls[right_index], score, str(group.get("reason", "主智能体分组提示"))[:120], likely_same)

    bucket_map: dict[str, list[str]] = {}
    for record in records:
        group_id = str(record.get("agent_same_source_group", "") or ((record.get("agent_hint") or {}).get("same_source_group", ""))).strip()
        if not group_id or not record.get("url"):
            continue
        bucket_map.setdefault(group_id, []).append(record.get("url", ""))
    for group_id, urls in bucket_map.items():
        deduped_urls = []
        for url in urls:
            if url and url not in deduped_urls:
                deduped_urls.append(url)
        if len(deduped_urls) < 2:
            continue
        for left_index in range(len(deduped_urls)):
            for right_index in range(left_index + 1, len(deduped_urls)):
                register_pair(deduped_urls[left_index], deduped_urls[right_index], 0.7, f"共享主智能体同源组 {group_id[:40]}", False)

    for pair in agent_evidence_catalog.get("ambiguous_pairs", []):
        if not isinstance(pair, dict):
            continue
        register_pair(
            str(pair.get("left_url", "")),
            str(pair.get("right_url", "")),
            0.55,
            str(pair.get("reason", "主智能体标记为歧义对"))[:120],
            False,
        )

    candidates = list(pair_map.values())
    candidates.sort(key=lambda item: item.get("same_source_score", 0.0), reverse=True)
    return candidates[:_adaptive_cap(len(records), minimum=6, multiplier=2.0, hard_cap=_MAX_PAIR_CANDIDATES)]


def _merge_same_source_candidates(*candidate_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate_list in candidate_lists:
        for candidate in candidate_list or []:
            if not isinstance(candidate, dict):
                continue
            left_index = candidate.get("left_index")
            right_index = candidate.get("right_index")
            if not isinstance(left_index, int) or not isinstance(right_index, int) or left_index == right_index:
                continue
            key = tuple(sorted((left_index, right_index)))
            existing = merged.get(key)
            if not existing:
                merged[key] = dict(candidate)
                merged[key]["left_index"] = key[0]
                merged[key]["right_index"] = key[1]
                continue
            existing["same_source_score"] = max(existing.get("same_source_score", 0.0), candidate.get("same_source_score", 0.0))
            existing["likely_same_source"] = existing.get("likely_same_source", False) or candidate.get("likely_same_source", False)
            existing["reason_signals"] = list(dict.fromkeys(existing.get("reason_signals", []) + candidate.get("reason_signals", [])))[:5]
            for field in ["title_similarity", "lead_similarity", "attribution_overlap", "entity_overlap"]:
                existing[field] = max(existing.get(field, 0.0), candidate.get(field, 0.0))
            existing["domain_match"] = existing.get("domain_match", False) or candidate.get("domain_match", False)
            existing["time_match"] = existing.get("time_match", False) or candidate.get("time_match", False)
            existing["agent_hint"] = existing.get("agent_hint", False) or candidate.get("agent_hint", False)

    merged_list = list(merged.values())
    merged_list.sort(key=lambda item: item.get("same_source_score", 0.0), reverse=True)
    return merged_list


def _extract_domain_key(record: dict[str, Any]) -> str:
    profile = record.get("source_profile", {})
    return (profile.get("domain") or urlparse(record.get("normalized_url") or record.get("url") or "").hostname or "").lower()


def _hybrid_text_similarity(left_text: str, right_text: str) -> float:
    if not left_text or not right_text:
        return 0.0
    lexical = _lexical_backoff_score(left_text, right_text)
    left_numbers = _extract_numbers(left_text)
    right_numbers = _extract_numbers(right_text)
    number_overlap = _overlap_ratio(left_numbers, right_numbers) if left_numbers or right_numbers else 0.0
    normalized_left = _normalize_text(left_text)
    normalized_right = _normalize_text(right_text)
    direct_match = 1.0 if normalized_left == normalized_right or normalized_left in normalized_right or normalized_right in normalized_left else 0.0
    return min(1.0, lexical * 0.7 + number_overlap * 0.15 + direct_match * 0.15)


def _overlap_ratio(left_set: set[str], right_set: set[str]) -> float:
    if not left_set or not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _time_match_score(left_time: str, right_time: str) -> float:
    if not left_time or not right_time:
        return 0.0
    left_date = left_time[:10]
    right_date = right_time[:10]
    return 1.0 if left_date == right_date else 0.0


def _build_rule_based_dedup_groups(
    records: list[dict[str, Any]],
    pair_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        return []

    index_to_record = {record.get("index"): record for record in records}
    adjacency = {record.get("index"): set() for record in records}
    for pair in pair_candidates:
        left_index = pair.get("left_index")
        right_index = pair.get("right_index")
        if left_index not in adjacency or right_index not in adjacency:
            continue
        if not pair.get("likely_same_source"):
            continue
        adjacency[left_index].add(right_index)
        adjacency[right_index].add(left_index)

    visited = set()
    groups = []
    ordered_indices = [record.get("index") for record in records]
    group_id = 1
    for index in ordered_indices:
        if index in visited:
            continue
        stack = [index]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency.get(current, set()) - visited, reverse=True))

        component.sort()
        description = "独立来源，暂未发现明显转载链"
        original_index = None
        if len(component) > 1:
            description = _describe_rule_based_group(component, pair_candidates)
            original_index = _select_group_original(component, index_to_record)
        groups.append(
            {
                "group_id": group_id,
                "group_type": "independent" if len(component) == 1 else "same_source_cluster",
                "original_index": original_index,
                "is_original": bool(original_index),
                "member_indices": component,
                "members": _build_group_members(component, index_to_record, original_index),
                "description": description,
            }
        )
        group_id += 1

    return groups


def _select_group_original(component: list[int], index_to_record: dict[int, dict[str, Any]]) -> int:
    ranked = []
    for index in component:
        record = index_to_record.get(index, {})
        provenance = record.get("provenance", {})
        source_profile = record.get("source_profile", {})
        publish_time = provenance.get("publish_time", "")
        ranked.append(
            (
                0 if publish_time else 1,
                publish_time or "9999-99-99",
                -float(source_profile.get("credibility_score", 0)),
                -float(provenance.get("provenance_score", 0.0)),
                index,
            )
        )
    ranked.sort()
    return ranked[0][-1] if ranked else component[0]


def _describe_rule_based_group(component: list[int], pair_candidates: list[dict[str, Any]]) -> str:
    related_pairs = [
        pair for pair in pair_candidates
        if pair.get("left_index") in component and pair.get("right_index") in component
    ]
    signals = []
    if any(pair.get("title_similarity", 0.0) >= 0.8 for pair in related_pairs):
        signals.append("标题相似")
    if any(pair.get("lead_similarity", 0.0) >= 0.75 for pair in related_pairs):
        signals.append("导语相似")
    if any(pair.get("attribution_overlap", 0.0) >= 0.5 for pair in related_pairs):
        signals.append("共享归因来源")
    if any(pair.get("time_match") for pair in related_pairs):
        signals.append("发布时间接近")
    if not signals:
        signals.append("综合溯源信号接近")
    return "疑似同源传播链: " + "、".join(signals[:3])


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


def _llm_cross_analysis(
    claim_text: str,
    evidence_info: list[dict],
    same_source_candidates: list[dict[str, Any]],
    fallback_dedup_groups: list[dict[str, Any]],
    agent_evidence_catalog: dict | None = None,
) -> dict:
    """用 LLM 做溯源去重 + 立场对比分析"""

    if not evidence_info:
        return _default_analysis(0, error="没有进入交叉验证的高相关证据", dedup_groups=[])

    system_prompt = """你是一位资深的事实核查专家，擅长分析多个信息来源之间的关系。

你需要完成两项分析任务：

【任务1: 溯源去重】
检查多条证据是否来自同一个原始报道/信息源。如果多条结果只是互相转载，则它们不算独立来源。
将证据按"实际来源"分组，标明哪些是原始报道、哪些是转载。
只有在标题、导语、发布时间、归因语句、quoted_sources、source_entities、same_source_candidates 中出现明确同源线索时，才能把多条证据归为同组；如果证据不足以证明同源，应保守地视为独立来源。
不要把“同一主题”误判为“同一原始报道”。同源要求至少体现为：转载关系、相同原始信源、显著相同的标题/导语、或明确共享同一首发来源。

【任务2: 立场对比】
对比不同来源对该声明的态度：
- support（支持/证实声明为真）
- deny（否认/辟谣/认为声明为假）
- mixed（同一页面并列呈现支持与反驳口径，不能只归为单一支持或单一否认）
- neutral（中立/仅报道事实不做判断）
- irrelevant（与声明无关）

特别注意：
1. 如果官方/权威来源与自媒体来源结论不同，这是重要的矛盾信号。
2. 如果同一篇转载/汇总页同时引用“支持声明”的说法和“反驳声明”的说法，应优先标为 mixed，并在 reason 中点明双方各自的口径。"""

    evidence_text = json.dumps(evidence_info, ensure_ascii=False, indent=2)
    pair_text = json.dumps(same_source_candidates, ensure_ascii=False, indent=2)
    fallback_text = json.dumps(fallback_dedup_groups, ensure_ascii=False, indent=2)
    agent_catalog_text = json.dumps(agent_evidence_catalog or {}, ensure_ascii=False, indent=2)

    user_prompt = f"""请分析以下证据来源的独立性和立场。

【待验证声明】
{claim_text[:5000]}

【证据来源列表】
{evidence_text}

【候选同源证据对】
{pair_text}

【规则回退分组（仅供参考，不可盲从）】
{fallback_text}

【主智能体证据编目先验（仅供参考，不可盲从）】
{agent_catalog_text}

字段理解要求：
- dedup_groups 用于说明哪些证据是独立来源，哪些只是转载或二次转述
- independent_source_count 只统计真正独立的信息源数量
- stance_analysis.stances 用于逐条说明每条证据的立场及理由
- contradiction_detail 只在存在明确矛盾时填写，否则留空或省略
- summary 需要总结“独立性 + 立场分布 + 是否存在矛盾”这三个方面
- evidence_info 中的 title、lead_excerpt、publish_time、quoted_sources、source_entities、attribution_sentences 和 same_source_candidates 是主要溯源依据
- 主智能体编目先验可用于优先发现明显同源或明显独立的证据，但如果与正文溯源信号冲突，应以正文与传播链证据为准
- source_clues、relevance_signals、score_reason 只是辅助线索，可用于补充判断，但不能据此臆造传播链
- 如果某条证据只是背景材料或与声明弱相关，应将其 stance 设为 irrelevant 或 neutral，而不是强行参与主要结论
- 如果同一页面既转述支持方说法，又明确转述反方反驳、否认或相反数字口径，不要只选其中一边；应输出 mixed

仅输出 JSON，格式:
{{
  "dedup_groups": [
    {{
      "group_id": 1,
            "group_type": "independent",
            "original_index": null,
      "member_indices": [1],
            "description": "独立来源"
    }},
    {{
      "group_id": 2,
            "group_type": "same_source_cluster",
            "original_index": 2,
      "member_indices": [2, 3],
      "description": "转载自同一篇报道"
    }}
  ],
  "independent_source_count": 2,
  "stance_analysis": {{
    "stances": [
      {{"index": 1, "stance": "support", "reason": "..."}},
      {{"index": 2, "stance": "mixed", "reason": "..."}}
    ],
    "has_contradiction": true,
    "contradiction_detail": "官方来源辟谣，但自媒体仍在传播",
    "dominant_stance": "deny"
  }},
  "summary": "交叉验证结论总结"
}}"""

    try:
        resp = _cross_verify_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        result = _parse_json(resp.content)

        if not isinstance(result, dict) or not result:
            return _default_analysis(len(evidence_info), dedup_groups=fallback_dedup_groups)

        if not isinstance(result.get("dedup_groups", []), list):
            result["dedup_groups"] = []
        if not isinstance(result.get("stance_analysis", {}), dict):
            result["stance_analysis"] = {}

        # 校验 & 补全
        result.setdefault("dedup_groups", [])
        result.setdefault("independent_source_count", len(evidence_info))
        result.setdefault("stance_analysis", {})
        result["stance_analysis"].setdefault("has_contradiction", False)
        result["stance_analysis"].setdefault("dominant_stance", "neutral")
        result["stance_analysis"].setdefault("stances", [])
        result.setdefault("summary", "")
        if not result.get("dedup_groups"):
            result["dedup_groups"] = fallback_dedup_groups
            result["independent_source_count"] = len(fallback_dedup_groups)

        return result

    except Exception as e:
        return _default_analysis(len(evidence_info), error=str(e), dedup_groups=fallback_dedup_groups)


def _calculate_weighted_score(
    source_profiles: list[dict],
    original_scores: list,
    independent_count: int,
    stance_analysis: dict,
    total_count: int,
    source_diversity: dict[str, Any] | None = None,
) -> float:
    """
    计算交叉验证加权得分。

    加权因素：
    1. 信源信誉度权重（官方来源权重更高）
    2. 独立来源数量（独立来源越多越可信）
    3. 矛盾惩罚（有矛盾则降低确信度）
    """
    if not source_profiles or not original_scores:
        return 5.0  # 无法判断

    # 1. 按信源信誉度加权的证据分数
    weighted_sum = 0.0
    weight_total = 0.0
    for i, profile in enumerate(source_profiles):
        if i >= len(original_scores):
            break
        score = original_scores[i] if isinstance(original_scores[i], (int, float)) else 5
        credibility = profile.get("credibility_score", 3)
        weight = credibility / 10.0  # 归一化到 0-1
        weighted_sum += score * weight
        weight_total += weight

    base_score = weighted_sum / weight_total if weight_total > 0 else 5.0

    # 2. 独立来源数量奖惩
    if total_count > 0:
        independence_ratio = independent_count / total_count
        # 独立性高 → 微调加分; 独立性低（都是转载）→ 向 5 靠拢
        if independence_ratio < 0.5:
            base_score = base_score * 0.7 + 5.0 * 0.3  # 向不确定靠拢
    
    # 3. 矛盾检测惩罚
    if stance_analysis.get("has_contradiction"):
        # 有矛盾 → 向 5.0（不确定）靠拢
        base_score = base_score * 0.6 + 5.0 * 0.4

    # 4. 来源国家/阵营集中惩罚
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

    index_to_record = {record.get("index"): record for record in records}
    representative_records: list[dict[str, Any]] = []
    for group in dedup_groups or []:
        member_indices = group.get("member_indices", [])
        if not member_indices:
            continue
        representative_index = group.get("original_index") or member_indices[0]
        representative = index_to_record.get(representative_index)
        if representative:
            representative_records.append(representative)
    if not representative_records:
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


def _default_analysis(count: int, error: str = "", dedup_groups: Optional[list[dict[str, Any]]] = None) -> dict:
    """LLM 调用失败时的降级结果"""
    return {
        "dedup_groups": dedup_groups or [],
        "independent_source_count": len(dedup_groups or []) or count,
        "stance_analysis": {
            "has_contradiction": False,
            "dominant_stance": "neutral",
            "stances": [],
        },
        "summary": f"交叉验证分析降级处理。{(' 错误: ' + error) if error else ''}",
    }


def _record_fact_signal_sets(record: dict[str, Any]) -> dict[str, set[str]]:
    provenance = record.get("provenance", {})
    text_parts = [
        record.get("summary", ""),
        record.get("snippet", ""),
        " ".join(record.get("matched_segments", [])[:4]),
        provenance.get("title", ""),
        provenance.get("lead_excerpt", ""),
    ]

    entities = set()
    for raw_item in provenance.get("source_entities", []) + provenance.get("quoted_sources", []):
        normalized = _normalize_source_name(str(raw_item or ""))
        if normalized:
            entities.add(normalized)

    numbers = set(str(value) for value in (record.get("extracted_numbers", []) or []) if str(value).strip())
    numbers |= _extract_numbers(" ".join(text_parts))
    segments = {_normalize_text(segment) for segment in record.get("matched_segments", [])[:4] if _normalize_text(segment)}

    anchors = set()
    for text in text_parts:
        anchors |= _extract_anchor_tokens(text)

    return {
        "entities": entities,
        "numbers": numbers,
        "segments": segments,
        "anchors": anchors,
    }


def _same_source_supplement_info(record: dict[str, Any], original_record: dict[str, Any]) -> dict[str, Any]:
    if not original_record:
        return {
            "is_supplementary": False,
            "additional_fact_score": 0.0,
            "additional_fact_count": 0,
            "supplementary_facts": [],
            "summary_label": "",
        }

    current_signals = _record_fact_signal_sets(record)
    original_signals = _record_fact_signal_sets(original_record)

    novel_entities = sorted(current_signals["entities"] - original_signals["entities"])[:3]
    novel_numbers = sorted(current_signals["numbers"] - original_signals["numbers"])[:3]
    novel_segments = sorted(current_signals["segments"] - original_signals["segments"])[:2]
    novel_anchors = sorted(current_signals["anchors"] - original_signals["anchors"])[:6]

    supplementary_facts = []
    if novel_entities:
        supplementary_facts.append("new entities: " + ", ".join(novel_entities))
    if novel_numbers:
        supplementary_facts.append("new numbers: " + ", ".join(novel_numbers))
    if novel_segments:
        supplementary_facts.append("new segments: " + " | ".join(novel_segments))
    elif len(novel_anchors) >= 3:
        supplementary_facts.append("new terms: " + ", ".join(novel_anchors[:4]))

    additional_fact_score = 0.0
    if novel_entities:
        additional_fact_score += min(len(novel_entities), 3) * 0.35
    if novel_numbers:
        additional_fact_score += min(len(novel_numbers), 3) * 0.25
    if novel_segments:
        additional_fact_score += min(len(novel_segments), 2) * 0.4
    elif len(novel_anchors) >= 3:
        additional_fact_score += 0.3
    additional_fact_score += min(float(record.get("novelty_score", 0.0) or 0.0), 2.0) * 0.1

    additional_fact_count = len(novel_entities) + len(novel_numbers) + len(novel_segments)
    is_supplementary = additional_fact_score >= 0.55 and bool(supplementary_facts)
    return {
        "is_supplementary": is_supplementary,
        "additional_fact_score": additional_fact_score,
        "additional_fact_count": additional_fact_count,
        "supplementary_facts": supplementary_facts[:3],
        "summary_label": "same-source supplement" if is_supplementary else "",
    }


def _build_group_members(
    component: list[int],
    index_to_record: dict[int, dict[str, Any]],
    original_index: Optional[int],
) -> list[dict[str, Any]]:
    members = []
    original_record = index_to_record.get(original_index, {}) if original_index else {}
    for index in component:
        record = index_to_record.get(index, {})
        profile = record.get("source_profile", {})
        provenance = record.get("provenance", {})
        supplement_info = _same_source_supplement_info(record, original_record if index != original_index else {})
        if len(component) == 1:
            relation = "independent"
            status_label = "独立来源"
        elif index == original_index:
            relation = "original"
            status_label = "原始报道"
        else:
            relation = "repost"
            status_label = "转载/转述"
        if relation == "repost" and supplement_info.get("is_supplementary"):
            relation = "supplementary_repost"
            status_label = supplement_info.get("summary_label", "same-source supplement")
        members.append(
            {
                "index": index,
                "url": record.get("url", ""),
                "normalized_url": record.get("normalized_url", ""),
                "source_name": profile.get("name", "未知来源"),
                "domain": profile.get("domain", ""),
                "country": profile.get("country", "未知"),
                "tier": profile.get("tier", "unknown"),
                "tier_label": profile.get("tier_label", TIER_LABELS["unknown"]),
                "title": provenance.get("title", ""),
                "publish_time": provenance.get("publish_time", ""),
                "relation": relation,
                "display_status": status_label,
                "additional_fact_score": round(float(supplement_info.get("additional_fact_score", 0.0) or 0.0), 3),
                "additional_fact_count": int(supplement_info.get("additional_fact_count", 0) or 0),
                "supplementary_facts": supplement_info.get("supplementary_facts", []),
                "has_additional_reporting": bool(supplement_info.get("is_supplementary")),
            }
        )
    return members


def _finalize_dedup_groups(
    records: list[dict[str, Any]],
    raw_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        return []

    index_to_record = {record.get("index"): record for record in records}
    normalized_groups = []
    covered_indices = set()

    for raw_group in raw_groups or []:
        if not isinstance(raw_group, dict):
            continue
        member_indices = []
        for index in raw_group.get("member_indices", []):
            if index in index_to_record and index not in member_indices:
                member_indices.append(index)
        if not member_indices:
            continue

        member_indices.sort()
        if len(member_indices) == 1:
            original_index = None
            group_type = "independent"
            description = raw_group.get("description") or "独立来源，暂未发现明显转载链"
        else:
            chosen_original = raw_group.get("original_index")
            original_index = chosen_original if chosen_original in member_indices else _select_group_original(member_indices, index_to_record)
            group_type = "same_source_cluster"
            description = raw_group.get("description") or "疑似同源传播链"

        normalized_groups.append(
            {
                "group_id": len(normalized_groups) + 1,
                "group_type": group_type,
                "original_index": original_index,
                "is_original": bool(original_index),
                "member_indices": member_indices,
                "members": _build_group_members(member_indices, index_to_record, original_index),
                "description": description,
            }
        )
        covered_indices.update(member_indices)

    for record in records:
        index = record.get("index")
        if index in covered_indices:
            continue
        normalized_groups.append(
            {
                "group_id": len(normalized_groups) + 1,
                "group_type": "independent",
                "original_index": None,
                "is_original": False,
                "member_indices": [index],
                "members": _build_group_members([index], index_to_record, None),
                "description": "独立来源，暂未发现明显转载链",
            }
        )

    return normalized_groups


def _build_traceable_evidence(
    records: list[dict[str, Any]],
    dedup_groups: list[dict[str, Any]],
    stance_analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    trace_lookup = {}
    stance_lookup = {}
    stance_map = (stance_analysis or {}).get("stances", [])
    if isinstance(stance_map, list):
        for item in stance_map:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if isinstance(index, int):
                stance_lookup[index] = item

    for group in dedup_groups:
        for member in group.get("members", []):
            trace_lookup[member.get("index")] = {
                "group_id": group.get("group_id"),
                "group_type": group.get("group_type", "independent"),
                "group_description": group.get("description", ""),
                **member,
            }

    items = []
    for record in sorted(records, key=lambda item: item.get("index", 0)):
        profile = record.get("source_profile", {})
        provenance = record.get("provenance", {})
        record_index = record.get("index")
        trace = trace_lookup.get(record_index, {})
        stance = dict(stance_lookup.get(record_index, {}))
        heuristic = _heuristic_record_stance(record)
        has_multi_source_context = (
            len(provenance.get("quoted_sources", []) or []) >= 2
            or len(provenance.get("attribution_sentences", []) or []) >= 2
            or len(provenance.get("source_entities", []) or []) >= 2
        )
        record_text = _record_text_blob(record).lower()
        if any(marker.lower() in record_text for marker in _STANCE_STRONG_DENY_MARKERS) and not has_multi_source_context:
            stance["stance"] = "deny"
            stance["reason"] = stance.get("reason") or "页面含明确反驳表述，且未呈现双方并列口径。"
        elif heuristic.get("stance") == "mixed":
            stance["stance"] = "mixed"
            stance["reason"] = heuristic.get("reason") or stance.get("reason", "")
        elif heuristic.get("stance") and stance.get("stance") in {None, "", "neutral", "irrelevant"}:
            stance["stance"] = heuristic["stance"]
            stance["reason"] = heuristic.get("reason") or stance.get("reason", "")
        elif stance.get("stance") == "mixed" and not has_multi_source_context:
            if any(marker.lower() in record_text for marker in _STANCE_DENY_MARKERS):
                stance["stance"] = "deny"
                stance["reason"] = stance.get("reason") or "页面主要呈现反驳口径，未见双方并列报道结构。"
        items.append(
            {
                "index": record_index,
                "url": record.get("url", ""),
                "source_name": trace.get("source_name") or profile.get("name", "未知来源"),
                "domain": trace.get("domain") or profile.get("domain", ""),
                "country": trace.get("country") or profile.get("country", "未知"),
                "tier": trace.get("tier") or profile.get("tier", "unknown"),
                "tier_label": trace.get("tier_label") or profile.get("tier_label", TIER_LABELS["unknown"]),
                "display_status": trace.get("display_status", "独立来源"),
                "relation": trace.get("relation", "independent"),
                "group_id": trace.get("group_id"),
                "group_description": trace.get("group_description", ""),
                "title": provenance.get("title", ""),
                "summary": record.get("summary", ""),
                "publish_time": provenance.get("publish_time", ""),
                "score": float(record.get("score", 0) or 0),
                "additional_fact_score": float(trace.get("additional_fact_score", 0.0) or 0.0),
                "additional_fact_count": int(trace.get("additional_fact_count", 0) or 0),
                "supplementary_facts": trace.get("supplementary_facts", []),
                "has_additional_reporting": bool(trace.get("has_additional_reporting")),
                "stance": stance.get("stance", "neutral"),
                "stance_label": _stance_label(stance.get("stance", "neutral")),
                "stance_reason": stance.get("reason", ""),
            }
        )
    relation_rank = {
        "independent": 0,
        "original": 1,
        "supplementary_repost": 2,
        "repost": 3,
    }
    items.sort(
        key=lambda item: (
            int(item.get("group_id") or 9999),
            relation_rank.get(str(item.get("relation", "")), 9),
            -float(item.get("additional_fact_score", 0.0) or 0.0),
            -float(item.get("score", 0.0) or 0.0),
            int(item.get("index") or 0),
        )
    )
    return items


def _stance_label(stance: str) -> str:
    return {
        "support": "✅支持",
        "deny": "❌否认",
        "mixed": "⚖️双向",
        "neutral": "➖中立",
        "irrelevant": "⬜无关",
    }.get(stance or "neutral", "➖中立")


def format_cross_verify_for_display(result: dict) -> str:
    """将交叉验证结果格式化为人类可读文本"""
    lines = []

    # 信源分布
    lines.append("📊 信源分布:")
    for tier_label, count in result.get("tier_distribution_readable", {}).items():
        lines.append(f"  {tier_label}: {count}条")

    # 独立来源
    total = result.get("total_source_count", 0)
    indep = result.get("independent_source_count", 0)
    lines.append(f"\n🔗 独立来源: {indep}/{total}")
    relevant = result.get("relevant_source_count", total)
    filtered = result.get("filtered_out_count", 0)
    lines.append(f"🎯 进入交叉验证的高相关证据: {relevant}条")
    if filtered:
        lines.append(f"🧹 混合语义筛选过滤低相关证据: {filtered}条")

    # 溯源去重
    groups = result.get("dedup_groups", [])
    if groups:
        lines.append("\n📂 来源分组:")
        for g in groups:
            members = g.get("member_indices", [])
            desc = g.get("description", "")
            group_type = g.get("group_type", "independent")
            label = "独立来源" if group_type == "independent" else "同源传播链"
            lines.append(f"  [{label}] 证据{members}: {desc}")
            for member in g.get("members", []):
                lines.append(
                    f"    - 证据[{member.get('index', '?')}] {member.get('display_status', '独立来源')} | {member.get('source_name', '未知来源')} | {member.get('url', '')}"
                )

    traceable_evidence = result.get("traceable_evidence", [])
    if traceable_evidence:
        lines.append("\n🔎 证据溯源明细:")
        for item in traceable_evidence:
            stance_label = item.get("stance_label") or _stance_label(item.get("stance", "neutral"))
            lines.append(
                f"  证据[{item.get('index', '?')}] {item.get('display_status', '独立来源')} | {stance_label} | {item.get('source_name', '未知来源')} | {item.get('country', '未知')} | {item.get('url', '')}"
            )
            if item.get("stance_reason"):
                lines.append(f"    立场理由: {item.get('stance_reason', '')}")

    source_diversity = result.get("source_diversity", {})
    if source_diversity.get("countries_readable"):
        lines.append("\n🌍 来源多样性:")
        lines.append(f"  国家分布: {_format_distribution(source_diversity.get('countries_readable', {}))}")
        lines.append(f"  阵营分布: {_format_distribution(source_diversity.get('geo_groups_readable', {}))}")
        if source_diversity.get("has_concentration_risk"):
            lines.append("  风险提示: 来源集中于相近国家/阵营，已降低交叉验证得分权重")

    numeric_analysis = result.get("numeric_analysis", {})
    if numeric_analysis.get("clusters"):
        lines.append("\n🔢 数字口径分析:")
        for cluster in numeric_analysis.get("clusters", [])[:4]:
            values = ", ".join(str(item.get("value", "")) for item in cluster.get("values", [])[:4])
            lines.append(
                f"  {cluster.get('context_label', cluster.get('context', '其他数字口径'))}: {values or '无'} ({'存在差异' if cluster.get('consistency') == 'mixed' else '基本一致'})"
            )
        if numeric_analysis.get("summary"):
            lines.append(f"  说明: {numeric_analysis.get('summary', '')}")

    # 立场分析
    stance = result.get("stance_analysis", {})
    if stance.get("stances"):
        lines.append("\n🎯 立场分析:")
        for s in stance["stances"]:
            label = _stance_label(s.get("stance", "neutral"))
            lines.append(f"  证据{s.get('index','?')}: {label} — {s.get('reason', '')}")

    # 矛盾检测
    if stance.get("has_contradiction"):
        lines.append(f"\n⚠️ 发现立场矛盾: {stance.get('contradiction_detail', '')}")

    # 综合得分
    lines.append(f"\n📈 交叉验证加权得分: {result.get('cross_verify_score', 5.0)}")
    if result.get("cross_verify_summary"):
        lines.append(f"💡 {result['cross_verify_summary']}")

    return "\n".join(lines)
