"""
主 Agent 管道。

本版以原生 tool-calling 作为真正的主 Agent：
- 大模型直接基于 MCP 工具 schema 自主调用工具
- Python 只负责执行工具、记录状态、流式输出和最终保存
"""

import ast
import asyncio
from collections import Counter
import json
import os
import re
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from app.config import PROJECT_ROOT, TEXT_MODEL, ZHIPU_API_KEY, ZHIPU_BASE_URL
from app.source_credibility import infer_source_metadata
main_agent_client = OpenAI(
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    timeout=120,
)


TOOL_EVENT_SOURCES = {
    "knowledge_base_lookup": "tool_a",
    "build_search_plan": "tool_b",
    "search_result_list": "tool_c1",
    "extract_relevant_segments": "tool_c2",
    "read_full_page": "tool_c3",
    "source_credibility_lookup": "tool_d",
    "cross_source_verify": "tool_e",
    "finalize_and_store": "tool_f",
    "save_result": "tool_save",
}
WRITE_ONLY_TOOLS = {"save_result", "finalize_and_store"}
MAX_AGENT_STEPS = 10
MAX_SEARCH_ROUNDS = 3
MAX_STALLED_SEARCH_ROUNDS = 2
MAX_CROSS_VERIFY_EVIDENCE_ITEMS = 12
MAX_COMPACT_TRACEABLE_EVIDENCE = 8

TOOL_SUMMARY_SPECS = {
    "knowledge_base_lookup": {
        "found": True,
        "can_determine": True,
        "reference_summary": 300,
        "reference_count": ("len", "reference_items"),
    },
    "build_search_plan": {
        "analysis_summary": 300,
        "core_question": 200,
        "verification_dimensions": 4,
        "focus_points": 4,
        "search_queries": 6,
        "risk_hypotheses": 4,
        "missing_information": 4,
    },
    "search_result_list": {
        "query": 180,
        "query_variants": 3,
        "result_count": True,
        "domain_distribution": True,
        "source_learning": True,
        "results": 5,
    },
    "extract_relevant_segments": {
        "matched_url_count": True,
        "search_terms": 8,
        "results": 5,
    },
    "read_full_page": {
        "url": 260,
        "title": 200,
        "content_length": True,
        "raw_content_excerpt": ("clip", "raw_content", 1200),
    },
    "source_credibility_lookup": {
        "profile_count": True,
        "tier_distribution": True,
        "country_distribution": True,
        "profiles": 6,
    },
    "cross_source_verify": {
        "independent_source_count": True,
        "total_source_count": True,
        "relevant_source_count": True,
        "cross_verify_score": True,
        "has_contradiction": True,
        "stance_analysis": True,
        "tier_distribution_readable": True,
        "numeric_analysis": True,
        "traceable_evidence": MAX_COMPACT_TRACEABLE_EVIDENCE,
        "filtered_out_samples": 3,
    },
    "finalize_and_store": {
        "classification": True,
        "reason": 300,
        "evidence_url": 260,
        "has_contradiction": True,
        "cross_verify_score": True,
    },
}

MAIN_AGENT_SYSTEM_PROMPT = """你是一个真正执行工具调用的主事实核查 Agent。你的目标是判断新闻是否可信，并在证据不足时自行继续取证；在证据已经足够或继续搜索收益很低时，及时停止并输出审慎裁决。

工作原则：
- 自主决定是否调用工具、调用哪个工具、调用几轮工具，不依赖固定链路。
- 优先使用最少但足够的工具，避免重复调用相同参数、重复读取同域名、重复扩大搜索范围。
- 搜索流程优先采用 C1→C2：先用 search_result_list 找候选网页，再把候选网页交给 extract_relevant_segments 提取命中片段。
- 只有当 C2 返回命中很少、片段与核心问题不相关、或明显缺少上下文时，才调用 read_full_page 获取网页全文。
- 不要跳过 C1 直接大范围调用 C3；C3 只用于补充读取少量高价值候选网页。
- 如果 use_db 为 false，不会提供 knowledge_base_lookup，不要假设数据库可用。
- 不要调用 save_result 或 finalize_and_store；最终保存由系统自动处理。
- 在搜索前、中、后都要主动分析当前材料的来源结构、立场分布、是否存在相互矛盾、还缺哪些关键信息，并据此调整下一步。
- 如果当前证据已经能支持稳健结论，就直接输出最终裁决 JSON；如果仍有关键不确定性，也要在停止时明确写进 reason，而不是无限搜索。
- 最终输出必须是单个 JSON 对象，不要输出 markdown 代码块，也不要添加额外解释。
- 现在时间是2026年3月。

何时继续搜索：
- 还没有抓到直接回应核心问题的证据。
- 现有证据主要来自单一来源、单一立场、单一转载链，缺少独立印证。
- 已发现明显立场冲突、数字冲突、时间线冲突，但还没有针对冲突点做过定向核查。
- 当前材料大多是摘要、二手转述或背景材料，缺少原始发布、官方说明、权威报道或正文上下文。

何时停止搜索并输出结果：
- 已有足够证据直接回答核心问题，例如已有高质量一手来源，或已有多个独立来源相互印证。
- 继续搜索已经明显进入低收益阶段，例如连续多轮都找不到新信息、命中内容高度重复、结果大多无关或只是在重复旧说法。
- 已经识别到矛盾，但经过针对性补充搜索后仍无法可靠消解，此时应停止搜索并给出“证据冲突/暂无法确定”的审慎结论。
- 当前剩余缺口并不影响主要裁决，只影响细节补充时，应停止搜索并直接输出。

搜索决策要求：
- 每一轮先判断当前最需要解决的是来源链路、核心事实、数字细节、时间地点、图像内容、传播立场，还是矛盾消解。
- 找到多方、多阵营对该数据的报道。比如新闻是讲伊朗的新闻，不止要找美国、以色列的这些敌对国家媒体，还要找伊朗的媒体，以及俄罗斯等其他中立媒体。或者这些不同阵营媒体报道的转载也行。
- 如果立场或矛盾已经暴露，优先做针对性搜索或交叉验证，不要继续泛搜。
- 如果已有候选网页足够多，优先从候选中提取片段或补读正文，而不是继续扩搜。
- 如果已经能回答问题，不要为了“凑更多来源”而机械继续搜索。

最终裁决 JSON 格式与字段说明：
{
    "claim_verdicts": [{
        "claim_id": 1,
        "claim_text": "...",
        "verdict_score": 5.0,
        "verdict_label": "...",
        "verdict_reason": "...",
        "key_evidence_url": "..."
    }],
    "classification": 5.0,
    "reason": "...",
    "evidence_url": "...",
    "publisher_conclusion": "...",
    "beneficiary_conclusion": "..."
}

字段解释：
- claim_verdicts: 对关键子声明逐条给出裁决列表；如果只有一个核心声明，也必须至少返回 1 条。
- claim_verdicts[].claim_id: 子声明编号，便于前端和后处理引用。
- claim_verdicts[].claim_text: 被核查的具体声明文本，应尽量聚焦可验证事实，而不是整段新闻原文。
- claim_verdicts[].verdict_score: 该子声明的可信度或成立度评分，范围 0 到 10，分数越高表示越可信或越接近被证实。
- claim_verdicts[].verdict_label: 对该子声明的简短标签，例如“基本属实”“存疑”“缺乏依据”“被辟谣”“信息冲突，暂无法确定”。
- claim_verdicts[].verdict_reason: 该子声明的主要依据，应概括证据、冲突、缺口和判断理由。
- claim_verdicts[].key_evidence_url: 支撑该子声明判断的最关键证据链接；没有时返回空字符串。
- classification: 对整条新闻的总体评分，范围 0 到 10，应与 claim_verdicts 的整体判断一致。
- reason: 对整条新闻的总体结论说明，需要说明为什么停搜、证据是否充分、是否存在冲突或未解问题。
- evidence_url: 整体判断最关键的证据链接；通常应与最重要的 claim_verdict 对应。
- publisher_conclusion: 对消息发布者或原始传播说法的结论，例如“有可靠依据”“缺少原始依据”“存在误导性表述”“来源不明”。
- beneficiary_conclusion: 对可能受益方、传播动机、受影响对象或立场受益结构的审慎总结；无法判断时可明确写“暂无法确定”。
"""

SUPPLEMENTAL_SEARCH_AGENT_PROMPT = """你是补充搜索智能体。你的任务是评估当前结构化证据是否已经足够支持进入最终分析。

要求：
- 关注证据是否直接回应核心问题，是否存在单一来源、同源转载、立场冲突、数字冲突或关键缺口。
- 如果证据不足，输出下一轮更聚焦的搜索项，而不是泛泛扩搜。
- 如果证据已经足够，明确指出可以停止搜索并进入验证分析。
- 仅输出 JSON，不要解释。

输出格式：
{
  "evidence_sufficient": true,
  "should_stop": true,
  "reason": "...",
  "updated_search_queries": ["..."],
  "priority_urls": ["..."],
  "should_cross_verify": true
}
"""

ANALYSIS_AGENT_PROMPT = """你是分析智能体。你的输入包括：原始声明、结构化证据、交叉验证结果，以及必要时补读的网页全文。

要求：
- 不要重新规划搜索，只做最终判断。
- 结合结构化证据、信源分层、矛盾情况和全文上下文，输出审慎结论。
- 如果证据仍不充分或存在未消解冲突，要明确写入 reason。
- 只输出单个 JSON 对象，不要输出 markdown。
- 现在时间是2026年3月。

输出格式必须符合：
{
    "claim_verdicts": [{
        "claim_id": 1,
        "claim_text": "...",
        "verdict_score": 5.0,
        "verdict_label": "...",
        "verdict_reason": "...",
        "key_evidence_url": "..."
    }],
    "classification": 5.0,
    "reason": "...",
    "evidence_url": "...",
    "publisher_conclusion": "...",
    "beneficiary_conclusion": "..."
}
"""

FOUR_AGENT_PLAN = {
    "goal": "判断新闻是否可信，并识别来源、核心事实与立场性叙事。",
    "agent_type": "four-agent-orchestrator",
    "stages": ["initial", "search", "validation"],
    "agents": [
        "initial_search_agent",
        "evidence_collection_agent",
        "supplemental_search_agent",
        "analysis_agent",
    ],
}


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("未找到 JSON 对象")
    return json.loads(text[start:end + 1])


def _parse_tool_output(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass

    return _extract_json_object(text)


def _clip(text: Any, limit: int = 5000) -> str:
    value = "" if text is None else str(text)
    return value[:limit]


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip())


def _safe_float(value: Any, default: float = 5.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _domain_from_url(url: str) -> str:
    match = re.match(r"https?://([^/]+)", url or "")
    if not match:
        return ""
    host = match.group(1).lower()
    return host[4:] if host.startswith("www.") else host


def _dedupe_text_keep_order(values: list[str], limit: int = 6, clip_limit: int = 160) -> list[str]:
    seen = set()
    items: list[str] = []
    for value in values or []:
        text = _normalize_query(_clip(value, clip_limit))
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _choose_granularity(current: str, incoming: str) -> str:
    rank = {"search_result": 1, "segment_match": 2, "full_page": 3}
    if rank.get(incoming, 0) >= rank.get(current, 0):
        return incoming
    return current or incoming


def _title_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()[:120]


def _merge_evidence_context(
    existing: dict[str, Any],
    tool_name: str,
    query_texts: list[str] | None = None,
    granularity: str = "search_result",
) -> dict[str, Any]:
    context = {
        "observation_count": int(existing.get("observation_count", 0)) + 1,
        "origin_tools": _dedupe_text_keep_order(list(existing.get("origin_tools", [])) + [tool_name], limit=4, clip_limit=40),
        "search_queries": _dedupe_text_keep_order(list(existing.get("search_queries", [])) + list(query_texts or []), limit=6, clip_limit=120),
        "evidence_granularity": _choose_granularity(existing.get("evidence_granularity", ""), granularity),
        "first_seen_tool": existing.get("first_seen_tool") or tool_name,
        "seen_titles": _dedupe_text_keep_order(existing.get("seen_titles", []), limit=4, clip_limit=200),
    }
    return context


def _source_type_from_domain(domain: str) -> str:
    lowered = (domain or "").lower()
    if any(token in lowered for token in ["gov", "mil", "mod.", "moj."]):
        return "官方"
    if any(token in lowered for token in ["reuters", "apnews", "bbc", "cnn", "nytimes", "xinhuanet"]):
        return "主流媒体"
    if any(token in lowered for token in ["defense", "military", "war", "csis", "think", "institute"]):
        return "专业媒体"
    return "其他"


def _tool_source(tool_name: str) -> str:
    return TOOL_EVENT_SOURCES.get(tool_name, tool_name)


def _coerce_text_list(value: Any, limit: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value[:limit]:
        text = _clip(item, 160).strip()
        if text and text not in items:
            items.append(text)
    return items


def _default_search_terms(state: dict) -> list[str]:
    analysis = state.get("analysis") or {}
    candidates = [
        analysis.get("core_question", ""),
        *analysis.get("focus_points", []),
        *analysis.get("verification_dimensions", []),
    ]
    normalized = []
    for item in candidates:
        text = _normalize_query(_clip(item, 120))
        if text and text not in normalized:
            normalized.append(text)
        if len(normalized) >= 6:
            break
    return normalized or [_normalize_query(_clip(state.get("news_text", ""), 120))]


def _default_search_query(state: dict) -> str:
    analysis = state.get("analysis") or {}
    used_queries = {
        _normalize_query(entry.get("arguments", {}).get("query", ""))
        for entry in state.get("history", [])
        if entry.get("tool_name") == "search_result_list"
    }
    for query in analysis.get("search_queries", []):
        normalized = _normalize_query(query)
        if normalized and normalized not in used_queries:
            return normalized
    return _normalize_query(analysis.get("core_question") or state.get("news_text", ""))


def _default_extract_urls(state: dict) -> list[str]:
    urls = []
    for search_round in reversed(state.get("search_results", [])):
        for item in search_round.get("results", []):
            url = item.get("url", "")
            if not url:
                continue
            evidence = state["evidence_by_url"].get(url, {})
            if evidence.get("matched_segments"):
                continue
            urls.append(url)
            if len(urls) >= 5:
                return _dedupe_keep_order(urls)
    return _dedupe_keep_order(urls)


def _collect_evidence_text(item: dict) -> str:
    parts = [
        item.get("title", ""),
        item.get("summary", ""),
        item.get("snippet", ""),
        item.get("reason", ""),
        " ".join(item.get("matched_segments", [])[:3]),
    ]
    return " ".join(part for part in parts if part).lower()


def _classify_evidence_stance(item: dict) -> str:
    text = _collect_evidence_text(item)
    if not text:
        return "neutral"

    deny_terms = [
        "辟谣", "不实", "谣言", "虚假", "夸大", "误导", "否认", "不存在", "未发生", "debunk", "false", "fake", "hoax", "misleading", "deny", "denied", "refute",
    ]
    support_terms = [
        "证实", "确认", "通报", "公布", "显示", "表明", "confirmed", "confirm", "verified", "shows", "indicates", "announced", "official said",
    ]

    deny_score = sum(1 for term in deny_terms if term in text)
    support_score = sum(1 for term in support_terms if term in text)
    if deny_score and support_score:
        return "mixed"
    if deny_score and deny_score >= support_score + 1:
        return "deny"
    if support_score and support_score >= deny_score + 1:
        return "support"
    return "neutral"


def _fallback_agent_evidence_catalog(state: dict) -> dict:
    evidence_items = sorted(
        state.get("evidence_by_url", {}).values(),
        key=lambda row: row.get("composite_score", 0),
        reverse=True,
    )[:10]
    evidence_notes = []
    signature_buckets: dict[str, list[str]] = {}
    for item in evidence_items:
        url = item.get("url", "")
        if not url:
            continue
        signature = _title_signature(item.get("title") or item.get("summary") or item.get("snippet", ""))
        if signature:
            signature_buckets.setdefault(signature, []).append(url)
        granularity = item.get("evidence_granularity", item.get("origin_tool", "search_result"))
        evidence_notes.append(
            {
                "url": url,
                "stance_hint": _classify_evidence_stance(item),
                "source_role": "background" if granularity == "search_result" else "primary",
                "originality_hint": "unknown",
                "same_source_group": signature[:48] if signature and len(signature_buckets.get(signature, [])) > 1 else "",
                "reason": "基于主智能体已浏览的标题、摘要和取证路径生成的保守先验。",
            }
        )

    group_hints = []
    group_index = 1
    for signature, urls in signature_buckets.items():
        normalized_urls = _dedupe_keep_order(urls)
        if len(normalized_urls) < 2:
            continue
        group_hints.append(
            {
                "group_id": f"fallback-{group_index}",
                "urls": normalized_urls[:4],
                "confidence": "medium",
                "reason": f"主智能体观察到标题/摘要签名接近: {signature[:60]}",
            }
        )
        group_index += 1

    return {
        "catalog_source": "fallback",
        "evidence_notes": evidence_notes,
        "group_hints": group_hints,
        "ambiguous_pairs": [],
        "summary": "未额外调用编目模型，使用主智能体已积累的证据上下文生成保守先验。",
    }


def _build_agent_evidence_catalog(state: dict) -> dict:
    evidence_items = sorted(
        state.get("evidence_by_url", {}).values(),
        key=lambda row: row.get("composite_score", 0),
        reverse=True,
    )[:10]
    if len(evidence_items) < 2:
        return _fallback_agent_evidence_catalog(state)

    payload = {
        "news_text": _clip(state.get("news_text", ""), 500),
        "core_question": _clip((state.get("analysis") or {}).get("core_question", ""), 200),
        "evidence_items": [
            {
                "url": item.get("url", ""),
                "title": _clip(item.get("title", ""), 200),
                "domain": item.get("domain", ""),
                "source_name": item.get("source_name", ""),
                "summary": _clip(item.get("summary", ""), 260),
                "matched_segments": item.get("matched_segments", [])[:3],
                "reason": _clip(item.get("reason", ""), 200),
                "origin_tools": item.get("origin_tools", [item.get("origin_tool", "")]),
                "search_queries": item.get("search_queries", [])[:4],
                "evidence_granularity": item.get("evidence_granularity", item.get("origin_tool", "search_result")),
                "composite_score": _safe_float(item.get("composite_score"), 5.0),
            }
            for item in evidence_items
            if item.get("url")
        ],
    }

    prompt = (
        "你是主事实核查 Agent 的证据编目助手。请基于已浏览过的证据卡，输出保守的同源分组先验与逐条证据备注。\n"
        "要求：\n"
        "- 只在明显同源时才给 group_hints；不要因为同一主题就强行归组。\n"
        "- evidence_notes 中对每条证据给出 stance_hint、source_role、originality_hint 和简短原因。\n"
        "- ambiguous_pairs 只列出确实值得交叉验证器重点比较的证据对。\n"
        "- 仅输出 JSON，对字段保持保守。\n\n"
        f"输入:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "输出格式:\n"
        "{\n"
        '  "evidence_notes": [{"url": "...", "stance_hint": "support|deny|mixed|neutral|irrelevant", "source_role": "primary|repost|analysis|background|unclear", "originality_hint": "original|repost|independent|unknown", "same_source_group": "", "reason": "..."}],\n'
        '  "group_hints": [{"group_id": "g1", "urls": ["...", "..."], "confidence": "high|medium", "reason": "..."}],\n'
        '  "ambiguous_pairs": [{"left_url": "...", "right_url": "...", "reason": "..."}],\n'
        '  "summary": "..."\n'
        "}"
    )
    try:
        response = main_agent_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1800,
        )
        content = response.choices[0].message.content or "{}"
        parsed = _parse_tool_output(content)
        if not isinstance(parsed, dict):
            raise ValueError("catalog json invalid")
        catalog = {
            "catalog_source": "agent_llm",
            "evidence_notes": parsed.get("evidence_notes", []) if isinstance(parsed.get("evidence_notes", []), list) else [],
            "group_hints": parsed.get("group_hints", []) if isinstance(parsed.get("group_hints", []), list) else [],
            "ambiguous_pairs": parsed.get("ambiguous_pairs", []) if isinstance(parsed.get("ambiguous_pairs", []), list) else [],
            "summary": _clip(parsed.get("summary", ""), 300),
        }
        if not catalog["evidence_notes"] and not catalog["group_hints"]:
            return _fallback_agent_evidence_catalog(state)
        return catalog
    except Exception:
        return _fallback_agent_evidence_catalog(state)


def _tool_result_stalled(tool_name: str, observation: dict) -> bool:
    if tool_name == "search_result_list":
        return int(observation.get("result_count") or 0) == 0
    if tool_name == "extract_relevant_segments":
        return int(observation.get("matched_url_count") or 0) == 0
    if tool_name == "read_full_page":
        return int(observation.get("content_length") or 0) < 200
    return False


def _count_trailing_stalled_rounds(state: dict) -> int:
    stalled = 0
    for entry in reversed(state.get("history", [])):
        tool_name = entry.get("tool_name")
        if tool_name not in {"search_result_list", "extract_relevant_segments", "read_full_page"}:
            continue
        if _tool_result_stalled(tool_name, entry.get("observation", {})):
            stalled += 1
            continue
        break
    return stalled


def _derive_internal_assessment(state: dict) -> dict:
    analysis = state.get("analysis") or {}
    cross_verify = state.get("cross_verify_result") or {}
    evidence_items = sorted(
        state.get("evidence_by_url", {}).values(),
        key=lambda item: item.get("composite_score", 0),
        reverse=True,
    )
    strong_evidence = [item for item in evidence_items if _safe_float(item.get("composite_score"), 0) >= 6.3]
    independent_domains = {item.get("domain") for item in strong_evidence if item.get("domain")}
    stance_rows = []
    stance_counter: Counter[str] = Counter()
    for item in evidence_items[:8]:
        stance = _classify_evidence_stance(item)
        stance_counter[stance] += 1
        stance_rows.append(
            {
                "url": item.get("url", ""),
                "source_type": item.get("source_type", "其他"),
                "stance": stance,
                "score": _safe_float(item.get("composite_score"), 5.0),
            }
        )

    dominant_stance = "neutral"
    if stance_counter:
        dominant_stance = max(["support", "deny", "mixed", "neutral"], key=lambda name: (stance_counter.get(name, 0), name == "neutral"))
    contradiction_risk = stance_counter.get("mixed", 0) > 0 or (stance_counter.get("support", 0) > 0 and stance_counter.get("deny", 0) > 0)
    stalled_rounds = _count_trailing_stalled_rounds(state)
    query_history = [
        _normalize_query(entry.get("arguments", {}).get("query", ""))
        for entry in state.get("history", [])
        if entry.get("tool_name") == "search_result_list"
    ]
    duplicate_query_count = len([query for query in query_history if query]) - len({query for query in query_history if query})
    enough_evidence = len(strong_evidence) >= 2 and len(independent_domains) >= 2
    resolved_by_cross_verify = bool(cross_verify) and int(cross_verify.get("relevant_source_count") or 0) >= 2 and not bool(cross_verify.get("has_contradiction"))
    search_saturated = stalled_rounds >= MAX_STALLED_SEARCH_ROUNDS or duplicate_query_count >= 2

    missing_information = _coerce_text_list(analysis.get("missing_information", []), limit=5)
    open_questions = _coerce_text_list(analysis.get("risk_hypotheses", []), limit=4)
    action = "continue_search"
    action_reason = "当前仍需补充直接证据。"
    if contradiction_risk and not cross_verify:
        action = "targeted_resolution"
        action_reason = "当前证据已出现支持与否认并存，优先定向消解矛盾。"
    elif enough_evidence or resolved_by_cross_verify:
        action = "finalize"
        action_reason = "当前证据已足以支撑稳健结论。"
    elif search_saturated:
        action = "finalize"
        action_reason = "继续搜索的新增收益已明显下降。"

    if contradiction_risk and not resolved_by_cross_verify and action == "finalize":
        action_reason = "虽然搜索已趋于饱和，但关键矛盾仍未完全消解，结论应保留不确定性。"

    return {
        "core_question": analysis.get("core_question", "") or _clip(state.get("news_text", ""), 120),
        "evidence_count": len(evidence_items),
        "strong_evidence_count": len(strong_evidence),
        "independent_domain_count": len(independent_domains),
        "dominant_stance": dominant_stance,
        "stance_distribution": dict(stance_counter),
        "contradiction_risk": contradiction_risk or bool(cross_verify.get("has_contradiction")),
        "cross_verify_available": bool(cross_verify),
        "cross_verify_score": _safe_float(cross_verify.get("cross_verify_score"), 5.0) if cross_verify else None,
        "stalled_search_rounds": stalled_rounds,
        "duplicate_query_count": duplicate_query_count,
        "missing_information": missing_information,
        "open_questions": open_questions,
        "recommended_action": action,
        "recommended_action_reason": action_reason,
        "stance_samples": stance_rows[:5],
    }


def _round_decision_message(state: dict, step_index: int) -> str:
    assessment = _derive_internal_assessment(state)
    return (
        f"当前是第 {step_index} 轮决策。下面是系统基于现有状态生成的内部态势评估，仅作为你规划下一步的参考，不是最终答案。\n"
        f"{json.dumps(assessment, ensure_ascii=False, indent=2)}\n\n"
        "决策要求：\n"
        "- 如果 recommended_action 为 finalize，且你已经能做出审慎结论，就直接输出最终 JSON。\n"
        "- 如果 recommended_action 为 targeted_resolution，优先做围绕矛盾点的定向搜索、片段提取或交叉验证，不要继续泛搜。\n"
        "- 如果已有候选网页足够多，优先使用 extract_relevant_segments 或 read_full_page 消化现有候选。\n"
        "- 如果 stalled_search_rounds 已较高或 duplicate_query_count 偏高，除非仍缺关键且可获取的一手证据，否则应停止搜索。\n"
        "- 输出最终 JSON 时，要在 reason 中说明证据充分性、停止原因和剩余不确定性。"
    )


def _summarize_value(value: Any, spec: Any) -> Any:
    if spec is True:
        return value
    if isinstance(spec, int):
        if isinstance(value, str):
            return _clip(value, spec)
        if isinstance(value, list):
            return value[:spec]
        return value
    if isinstance(spec, tuple):
        kind = spec[0]
        if kind == "len":
            return len(value or [])
        if kind == "clip":
            return _clip(value, spec[2])
    return value


def _compact_tool_result(tool_name: str, result: dict) -> dict:
    summary_spec = TOOL_SUMMARY_SPECS.get(tool_name)
    if summary_spec:
        compact_result = {}
        for key, spec in summary_spec.items():
            if tool_name == "cross_source_verify" and key == "traceable_evidence":
                compact_result[key] = _compact_traceable_evidence(
                    result.get("traceable_evidence", []),
                    limit=int(spec) if isinstance(spec, int) else MAX_COMPACT_TRACEABLE_EVIDENCE,
                )
                continue
            if isinstance(spec, tuple) and spec[0] == "clip":
                compact_result[key] = _summarize_value(result.get(spec[1], ""), spec)
                continue
            if isinstance(spec, tuple) and spec[0] == "len":
                compact_result[key] = _summarize_value(result.get(spec[1], []), spec)
                continue
            compact_result[key] = _summarize_value(result.get(key), spec)
        return compact_result
    return result


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


def _cross_verify_evidence_items(state: dict) -> list[dict]:
    agent_catalog = state.get("agent_evidence_catalog") or {}
    note_by_url = {
        note.get("url"): note
        for note in agent_catalog.get("evidence_notes", [])
        if isinstance(note, dict) and note.get("url")
    }
    evidence_items = sorted(
        state["evidence_by_url"].values(),
        key=lambda row: row.get("composite_score", 0),
        reverse=True,
    )
    selected_urls = set(_select_cross_verify_urls(evidence_items, note_by_url))
    normalized_items = []
    for item in evidence_items:
        if item.get("url", "") not in selected_urls:
            continue
        note = note_by_url.get(item.get("url", ""), {})
        normalized_items.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "domain": item.get("domain", ""),
                "summary": item.get("summary", ""),
                "snippet": item.get("snippet", item.get("summary", "")),
                "matched_segments": item.get("matched_segments", []),
                "extracted_numbers": item.get("extracted_numbers", []),
                "source_type": item.get("source_type", "其他"),
                "source_name": item.get("source_name", ""),
                "composite_score": item.get("composite_score", 5.0),
                "reason": item.get("reason", ""),
                "search_queries": item.get("search_queries", []),
                "origin_tools": item.get("origin_tools", [item.get("origin_tool", "")]),
                "evidence_granularity": item.get("evidence_granularity", item.get("origin_tool", "search_result")),
                "observation_count": item.get("observation_count", 1),
                "novelty_score": round(_evidence_novelty_score(item, note), 3),
                "agent_stance_hint": note.get("stance_hint", ""),
                "agent_source_role": note.get("source_role", ""),
                "agent_originality_hint": note.get("originality_hint", ""),
                "agent_same_source_group": note.get("same_source_group", ""),
                "agent_note": note.get("reason", ""),
            }
        )
    return normalized_items


def _same_source_bucket_counts(evidence_items: list[dict], note_by_url: dict[str, dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in evidence_items:
        bucket = _same_source_bucket_key(item, note_by_url.get(item.get("url", ""), {}))
        if bucket:
            counts[bucket] += 1
    return counts


def _same_source_bucket_key(item: dict[str, Any], note: dict[str, Any] | None = None) -> str:
    note = note or {}
    group = str(note.get("same_source_group", "") or item.get("agent_same_source_group", "")).strip()
    if group:
        return f"group:{group[:80]}"
    signature = _title_signature(item.get("title") or item.get("summary") or item.get("snippet", ""))
    return f"sig:{signature}" if signature else ""


def _evidence_novelty_score(item: dict[str, Any], note: dict[str, Any] | None = None) -> float:
    note = note or {}
    score = 0.0
    granularity = item.get("evidence_granularity", item.get("origin_tool", "search_result"))
    granularity_bonus = {"search_result": 0.2, "segment_match": 1.0, "full_page": 1.2}
    score += granularity_bonus.get(granularity, 0.4)
    score += min(len(item.get("matched_segments", []) or []), 4) * 0.35
    score += min(len(item.get("extracted_numbers", []) or []), 4) * 0.2
    score += min(int(item.get("observation_count", 1) or 1), 4) * 0.1
    if note.get("source_role") in {"primary", "analysis"}:
        score += 0.25
    if note.get("originality_hint") == "independent":
        score += 0.15
    text_lengths = [
        len(str(item.get("summary", "") or "")),
        len(str(item.get("snippet", "") or "")),
        sum(len(str(segment or "")) for segment in item.get("matched_segments", [])[:3]),
    ]
    score += min(max(text_lengths), 400) / 400
    return score


def _cross_verify_item_metadata(item: dict[str, Any], note: dict[str, Any] | None = None) -> dict[str, str]:
    note = note or {}
    existing = {
        key: value
        for key in ("country", "region", "geo_group")
        for value in [note.get(key) or item.get(key)]
        if str(value or "").strip()
    }
    text = " ".join(
        str(part or "")
        for part in [
            item.get("source_name", ""),
            item.get("title", ""),
            item.get("summary", ""),
            item.get("snippet", ""),
            item.get("reason", ""),
            note.get("reason", ""),
        ]
        if str(part or "").strip()
    )
    return infer_source_metadata(
        item.get("url") or item.get("domain", ""),
        name=str(item.get("source_name", "") or item.get("title", "")),
        text=text,
        existing=existing,
    )


def _cross_verify_candidate_rank(
    item: dict[str, Any],
    note: dict[str, Any] | None = None,
) -> tuple[float, float, int]:
    return (
        float(item.get("composite_score", 0) or 0),
        _evidence_novelty_score(item, note),
        int(item.get("observation_count", 1) or 1),
    )


def _is_same_source_supplement_candidate(
    item: dict[str, Any],
    note_by_url: dict[str, dict],
    bucket_counts: dict[str, int],
) -> bool:
    url = item.get("url", "")
    note = note_by_url.get(url, {})
    bucket = _same_source_bucket_key(item, note)
    if not bucket or bucket_counts.get(bucket, 0) < 2:
        return False
    return _evidence_novelty_score(item, note) >= 1.0


def _select_diversity_candidates(
    evidence_items: list[dict],
    note_by_url: dict[str, dict],
    selected_urls: list[str],
    limit: int,
    bucket_counts: dict[str, int],
) -> None:
    if len(selected_urls) >= limit:
        return

    covered_geo_groups: set[str] = set()
    covered_countries: set[str] = set()
    for url in selected_urls:
        item = next((row for row in evidence_items if row.get("url", "") == url), None)
        if not item:
            continue
        metadata = _cross_verify_item_metadata(item, note_by_url.get(url, {}))
        geo_group = str(metadata.get("geo_group", "") or "")
        country = str(metadata.get("country", "") or "")
        if geo_group and geo_group != "unknown":
            covered_geo_groups.add(geo_group)
        if country and country != "未知":
            covered_countries.add(country)

    target = min(limit, len(selected_urls) + min(3, max(0, limit - len(selected_urls))))
    while len(selected_urls) < target:
        best_url = ""
        best_rank: tuple[int, int, int, float, float, int] | None = None
        for item in evidence_items:
            url = item.get("url", "")
            if not url or url in selected_urls:
                continue
            note = note_by_url.get(url, {})
            metadata = _cross_verify_item_metadata(item, note)
            geo_group = str(metadata.get("geo_group", "") or "unknown")
            country = str(metadata.get("country", "") or "未知")
            adds_geo_group = int(geo_group != "unknown" and geo_group not in covered_geo_groups)
            adds_country = int(country != "未知" and country not in covered_countries)
            supplement = int(_is_same_source_supplement_candidate(item, note_by_url, bucket_counts))
            score, novelty, observations = _cross_verify_candidate_rank(item, note)
            if adds_geo_group == 0 and adds_country == 0 and supplement == 0:
                continue
            rank = (
                adds_geo_group,
                adds_country,
                supplement,
                score >= 6.0,
                score,
                int(novelty * 1000) + observations,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_url = url

        if not best_url:
            break

        selected_urls.append(best_url)
        chosen_item = next((row for row in evidence_items if row.get("url", "") == best_url), None)
        chosen_metadata = _cross_verify_item_metadata(chosen_item or {}, note_by_url.get(best_url, {}))
        geo_group = str(chosen_metadata.get("geo_group", "") or "unknown")
        country = str(chosen_metadata.get("country", "") or "未知")
        if geo_group != "unknown":
            covered_geo_groups.add(geo_group)
        if country != "未知":
            covered_countries.add(country)


def _select_cross_verify_urls(evidence_items: list[dict], note_by_url: dict[str, dict]) -> list[str]:
    if not evidence_items:
        return []

    limit = min(MAX_CROSS_VERIFY_EVIDENCE_ITEMS, len(evidence_items))
    base_limit = min(8, limit)
    selected_urls: list[str] = []
    ranked_items = sorted(
        evidence_items,
        key=lambda row: _cross_verify_candidate_rank(row, note_by_url.get(row.get("url", ""), {})),
        reverse=True,
    )

    for item in ranked_items[:base_limit]:
        url = item.get("url", "")
        if url and url not in selected_urls:
            selected_urls.append(url)

    bucket_counts = _same_source_bucket_counts(evidence_items, note_by_url)
    _select_diversity_candidates(ranked_items, note_by_url, selected_urls, limit, bucket_counts)
    supplement_candidates: list[tuple[float, float, str]] = []
    for item in ranked_items:
        url = item.get("url", "")
        if not url or url in selected_urls:
            continue
        if not _is_same_source_supplement_candidate(item, note_by_url, bucket_counts):
            continue
        novelty = _evidence_novelty_score(item, note_by_url.get(url, {}))
        supplement_candidates.append((novelty, float(item.get("composite_score", 0) or 0), url))

    supplement_candidates.sort(reverse=True)
    supplement_quota = min(max(2, limit - base_limit), len(supplement_candidates))
    for _, _, url in supplement_candidates[:supplement_quota]:
        if url not in selected_urls and len(selected_urls) < limit:
            selected_urls.append(url)

    for item in ranked_items:
        url = item.get("url", "")
        if not url or url in selected_urls:
            continue
        selected_urls.append(url)
        if len(selected_urls) >= limit:
            break

    return selected_urls[:limit]


def _compact_traceable_evidence(items: list[dict], limit: int = MAX_COMPACT_TRACEABLE_EVIDENCE) -> list[dict]:
    if not isinstance(items, list) or limit <= 0:
        return []

    group_order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        group_key = str(item.get("group_id") or item.get("url") or item.get("index") or "")
        if not group_key:
            continue
        if group_key not in groups:
            group_order.append(group_key)
            groups[group_key] = []
        groups[group_key].append(item)

    selected: list[dict] = []
    seen_keys: set[str] = set()

    def _append(item: dict[str, Any]) -> None:
        dedupe_key = str(item.get("url") or item.get("index") or "")
        if not dedupe_key or dedupe_key in seen_keys or len(selected) >= limit:
            return
        seen_keys.add(dedupe_key)
        selected.append(item)

    def _member_rank(item: dict[str, Any]) -> tuple[int, float, float, int]:
        relation = str(item.get("relation", "") or "independent")
        relation_rank = {
            "independent": 0,
            "original": 1,
            "supplementary_repost": 2,
            "repost": 3,
        }.get(relation, 9)
        return (
            relation_rank,
            -float(item.get("additional_fact_score", 0.0) or 0.0),
            -float(item.get("score", 0.0) or 0.0),
            int(item.get("index") or 0),
        )

    for group_key in group_order:
        ordered_members = sorted(groups[group_key], key=_member_rank)
        representative = next(
            (item for item in ordered_members if str(item.get("relation", "")) in {"independent", "original"}),
            ordered_members[0] if ordered_members else None,
        )
        if representative:
            _append(representative)

    for group_key in group_order:
        ordered_members = sorted(
            (
                item for item in groups[group_key]
                if bool(item.get("has_additional_reporting")) or float(item.get("additional_fact_score", 0.0) or 0.0) > 0.45
            ),
            key=lambda row: (
                float(row.get("additional_fact_score", 0.0) or 0.0),
                float(row.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for item in ordered_members:
            _append(item)

    remaining = sorted(
        [item for members in groups.values() for item in members],
        key=lambda row: (
            float(row.get("score", 0.0) or 0.0),
            float(row.get("additional_fact_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    for item in remaining:
        _append(item)

    return selected[:limit]


def _prepare_tool_arguments(state: dict, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(arguments)
    if tool_name == "knowledge_base_lookup":
        prepared.setdefault("news_text", state.get("news_text", ""))
    elif tool_name == "build_search_plan":
        prepared.setdefault("news_text", state.get("news_text", ""))
        if state.get("img_base64"):
            prepared.setdefault("img_base64", state.get("img_base64"))
        if state.get("knowledge_base"):
            prepared.setdefault("knowledge_references", state["knowledge_base"].get("reference_items", []))
    elif tool_name == "search_result_list":
        prepared.setdefault("query", _default_search_query(state))
        prepared.setdefault("exclude_urls", list(state.get("evidence_by_url", {}).keys())[:8])
    elif tool_name == "extract_relevant_segments":
        prepared.setdefault("urls", _default_extract_urls(state))
        prepared.setdefault("search_terms", _default_search_terms(state))
    elif tool_name == "read_full_page":
        if not prepared.get("url"):
            candidate_urls = _default_extract_urls(state)
            if candidate_urls:
                prepared["url"] = candidate_urls[0]
    elif tool_name == "source_credibility_lookup":
        prepared.setdefault("urls", _default_extract_urls(state) or _top_evidence_urls(state, limit=5))
    if tool_name == "cross_source_verify":
        prepared.setdefault("claim_text", state.get("news_text", ""))
        if not state.get("agent_evidence_catalog"):
            state["agent_evidence_catalog"] = _build_agent_evidence_catalog(state)
        evidence_items = _coerce_cross_verify_items(prepared.get("evidence_items"))
        if not evidence_items:
            evidence_items = _cross_verify_evidence_items(state)
        prepared["evidence_items"] = evidence_items
        prepared["evidence_list"] = _coerce_cross_verify_list(prepared.get("evidence_list"))
        prepared["web_list"] = _coerce_cross_verify_list(prepared.get("web_list"))
        prepared["scores"] = _coerce_cross_verify_list(prepared.get("scores"))
        prepared["reasons"] = _coerce_cross_verify_list(prepared.get("reasons"))
        if not prepared["evidence_list"]:
            prepared["evidence_list"] = [item.get("summary", "") for item in evidence_items]
        if not prepared["web_list"]:
            prepared["web_list"] = [item.get("url", "") for item in evidence_items]
        if not prepared["scores"]:
            prepared["scores"] = [item.get("composite_score", 5.0) for item in evidence_items]
        if not prepared["reasons"]:
            prepared["reasons"] = [item.get("reason", "") for item in evidence_items]
        prepared.setdefault("agent_evidence_catalog", state.get("agent_evidence_catalog", {}))
    return prepared


def _append_evidence_from_search_results(state: dict, result: dict) -> None:
    for item in result.get("results", [])[:6]:
        url = item.get("url", "")
        if not url:
            continue
        existing = state["evidence_by_url"].get(url, {})
        source_profile = item.get("source_profile", {})
        domain = item.get("domain") or _domain_from_url(url)
        summary = item.get("snippet", "")
        evidence = {
            "title": item.get("title", existing.get("title", "")),
            "url": url,
            "summary": _clip(existing.get("summary", "") or summary, 320),
            "snippet": _clip(existing.get("snippet", "") or summary, 320),
            "source_type": existing.get("source_type") or _source_type_from_domain(domain),
            "composite_score": max(existing.get("composite_score", 0), 4.8),
            "reason": existing.get("reason", "") or "来自搜索结果摘要，尚未读取正文。",
            "domain": domain,
            "source_name": existing.get("source_name") or source_profile.get("name", domain),
            "matched_segments": existing.get("matched_segments", []),
            "extracted_numbers": existing.get("extracted_numbers", []),
            "origin_tool": "search_result_list",
        }
        evidence.update(_merge_evidence_context(existing, "search_result_list", query_texts=[result.get("query", "")], granularity="search_result"))
        evidence["seen_titles"] = _dedupe_text_keep_order(list(existing.get("seen_titles", [])) + [item.get("title", existing.get("title", ""))], limit=4, clip_limit=200)
        state["evidence_by_url"][url] = evidence


def _append_evidence_from_segments(state: dict, result: dict) -> None:
    for item in result.get("results", [])[:6]:
        url = item.get("url", "")
        if not url:
            continue
        segments = item.get("matched_segments", [])
        summary = "；".join(segments[:2])
        existing = state["evidence_by_url"].get(url, {})
        domain = _domain_from_url(url)
        merged = {
            "title": item.get("title", existing.get("title", "")),
            "url": url,
            "summary": _clip(summary or existing.get("summary", ""), 420),
            "snippet": existing.get("snippet", existing.get("summary", "")),
            "source_type": existing.get("source_type") or _source_type_from_domain(domain),
            "composite_score": max(existing.get("composite_score", 0), 6.4),
            "reason": "网页正文中命中了搜索词。",
            "domain": domain,
            "source_name": existing.get("source_name", domain),
            "matched_segments": segments[:4],
            "extracted_numbers": item.get("extracted_numbers", existing.get("extracted_numbers", [])),
            "origin_tool": "extract_relevant_segments",
        }
        merged.update(_merge_evidence_context(existing, "extract_relevant_segments", query_texts=result.get("search_terms", []), granularity="segment_match"))
        merged["seen_titles"] = _dedupe_text_keep_order(list(existing.get("seen_titles", [])) + [item.get("title", existing.get("title", ""))], limit=4, clip_limit=200)
        state["evidence_by_url"][url] = merged


def _is_valid_full_page_result(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return False
    raw_content = str(result.get("raw_content", "") or "").strip()
    title = str(result.get("title", "") or "").strip()
    if not raw_content:
        return False
    if int(result.get("content_length") or len(raw_content)) < 200:
        return False
    return bool(title or raw_content)


def _append_evidence_from_full_page(state: dict, result: dict) -> None:
    url = result.get("url", "")
    if not url:
        return
    if not _is_valid_full_page_result(result):
        return
    excerpt = _clip(result.get("raw_content", ""), 420)
    existing = state["evidence_by_url"].get(url, {})
    domain = _domain_from_url(url)
    merged = {
        "title": result.get("title", existing.get("title", "")),
        "url": url,
        "summary": excerpt or existing.get("summary", ""),
        "snippet": existing.get("snippet", existing.get("summary", "")),
        "source_type": existing.get("source_type") or _source_type_from_domain(domain),
        "composite_score": max(existing.get("composite_score", 0), 6.8),
        "reason": "已读取网页全文并补充上下文。",
        "domain": domain,
        "source_name": existing.get("source_name", domain),
        "matched_segments": existing.get("matched_segments", []),
        "extracted_numbers": existing.get("extracted_numbers", []),
        "origin_tool": "read_full_page",
    }
    merged.update(_merge_evidence_context(existing, "read_full_page", query_texts=[], granularity="full_page"))
    merged["seen_titles"] = _dedupe_text_keep_order(list(existing.get("seen_titles", [])) + [result.get("title", existing.get("title", ""))], limit=4, clip_limit=200)
    state["evidence_by_url"][url] = merged


def _update_state_from_tool(state: dict, tool_name: str, result: dict) -> None:
    state["tool_results"][tool_name] = result
    if tool_name == "knowledge_base_lookup":
        state["knowledge_base"] = result
    elif tool_name == "build_search_plan":
        state["analysis"] = result
    elif tool_name == "search_result_list":
        state["search_results"].append(result)
        _append_evidence_from_search_results(state, result)
    elif tool_name == "source_credibility_lookup":
        _append_evidence_from_source_lookup(state, result)
    elif tool_name == "extract_relevant_segments":
        state["segment_results"].append(result)
        _append_evidence_from_segments(state, result)
    elif tool_name == "read_full_page":
        state["full_pages"].append(result)
        _append_evidence_from_full_page(state, result)
    elif tool_name == "cross_source_verify":
        state["cross_verify_result"] = result
    elif tool_name == "finalize_and_store":
        state["finalize_and_store_result"] = result


def _compact_state(state: dict) -> dict:
    evidence_items = list(state["evidence_by_url"].values())
    return {
        "use_db": state["use_db"],
        "has_image": bool(state["img_base64"]),
        "history_count": len(state["history"]),
        "knowledge_base": _compact_tool_result("knowledge_base_lookup", state.get("knowledge_base", {})) if state.get("knowledge_base") else {},
        "analysis": _compact_tool_result("build_search_plan", state.get("analysis", {})) if state.get("analysis") else {},
        "search_rounds": [_compact_tool_result("search_result_list", item) for item in state["search_results"][-2:]],
        "segment_rounds": [_compact_tool_result("extract_relevant_segments", item) for item in state["segment_results"][-2:]],
        "full_pages": [_compact_tool_result("read_full_page", item) for item in state["full_pages"][-2:]],
        "evidence_candidates": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "summary": _clip(item.get("summary", ""), 220),
                "source_type": item.get("source_type", "其他"),
                "composite_score": item.get("composite_score", 5.0),
                "evidence_granularity": item.get("evidence_granularity", item.get("origin_tool", "search_result")),
                "reason": _clip(item.get("reason", ""), 120),
            }
            for item in sorted(evidence_items, key=lambda row: row.get("composite_score", 0), reverse=True)[:8]
        ],
        "agent_evidence_catalog": {
            "catalog_source": (state.get("agent_evidence_catalog") or {}).get("catalog_source", ""),
            "group_hints": (state.get("agent_evidence_catalog") or {}).get("group_hints", [])[:4],
            "summary": (state.get("agent_evidence_catalog") or {}).get("summary", ""),
        } if state.get("agent_evidence_catalog") else {},
        "cross_verify_result": _compact_tool_result("cross_source_verify", state.get("cross_verify_result", {})) if state.get("cross_verify_result") else {},
        "internal_assessment": _derive_internal_assessment(state),
        "recent_history": state["history"][-5:],
    }


def _build_runtime_tool_schemas(tool_defs: list[dict], use_db: bool) -> tuple[list[dict], list[str]]:
    allowed_defs = []
    for tool_def in tool_defs:
        tool_name = tool_def["name"]
        if tool_name in WRITE_ONLY_TOOLS:
            continue
        if not use_db and tool_name == "knowledge_base_lookup":
            continue
        allowed_defs.append(tool_def)

    schemas = []
    for tool_def in allowed_defs:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool_def["name"],
                    "description": tool_def.get("description", ""),
                    "parameters": tool_def.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return schemas, [tool_def["name"] for tool_def in allowed_defs]


def _assistant_message_payload(message: Any) -> dict:
    payload = {"role": "assistant"}
    if getattr(message, "content", None):
        payload["content"] = message.content
    tool_calls = []
    for tool_call in getattr(message, "tool_calls", []) or []:
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _initial_agent_messages(state: dict) -> list[dict[str, Any]]:
    task_payload = {
        "news_text": state["news_text"],
        "has_image": bool(state["img_base64"]),
        "use_db": state["use_db"],
        "image_path": state["image_path"],
    }
    user_prompt = (
        "请对下面的新闻进行事实核查。你可以自主使用工具完成取证、检索、交叉验证，并在证据足够时直接输出最终裁决 JSON。\n\n"
        f"任务输入:\n{json.dumps(task_payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": MAIN_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _finalize_from_state(state: dict) -> dict:
    assessment = _derive_internal_assessment(state)
    evidence_items = sorted(state["evidence_by_url"].values(), key=lambda item: item.get("composite_score", 0), reverse=True)
    weighted = 5.0
    if evidence_items:
        weighted = sum(_safe_float(item.get("composite_score"), 5.0) for item in evidence_items[:5]) / min(len(evidence_items), 5)
    if state.get("cross_verify_result"):
        weighted = (weighted + _safe_float(state["cross_verify_result"].get("cross_verify_score"), 5.0)) / 2
    weighted = max(0.0, min(10.0, round(weighted, 1)))
    evidence_url = evidence_items[0].get("url", "") if evidence_items else ""
    core_question = (state.get("analysis") or {}).get("core_question", "综合核查") or "综合核查"
    reason_parts = []
    if (state.get("analysis") or {}).get("analysis_summary"):
        reason_parts.append(state["analysis"]["analysis_summary"])
    if evidence_items:
        reason_parts.append(f"已累计 {len(evidence_items)} 条候选证据。")
    if state.get("cross_verify_result"):
        reason_parts.append(f"交叉验证得分为 {_safe_float(state['cross_verify_result'].get('cross_verify_score'), 5.0):.1f}。")
    if assessment.get("contradiction_risk"):
        reason_parts.append("现有材料存在一定立场或表述冲突，结论已按审慎原则收束。")
    if assessment.get("recommended_action_reason"):
        reason_parts.append(assessment["recommended_action_reason"])

    publisher_conclusion = "来源链路仍需补充"
    if evidence_items:
        if assessment.get("contradiction_risk"):
            publisher_conclusion = "发布说法与现有证据之间存在冲突或表述落差"
        elif weighted >= 6.5:
            publisher_conclusion = "发布说法目前有较强证据支撑"
        elif weighted <= 4.0:
            publisher_conclusion = "发布说法缺少可靠依据或存在明显问题"
        else:
            publisher_conclusion = "发布说法已有部分依据，但仍存在待核实缺口"

    beneficiary_conclusion = "暂无法确定或不适用"
    if assessment.get("dominant_stance") == "support":
        beneficiary_conclusion = "传播结构更偏向强化原说法，受益方仍需结合外部背景判断"
    elif assessment.get("dominant_stance") == "deny":
        beneficiary_conclusion = "传播结构更偏向质疑或纠偏原说法，直接受益方暂不明确"
    if assessment.get("contradiction_risk"):
        beneficiary_conclusion = "不同立场叙事同时存在，潜在受益方或传播动机暂无法可靠锁定"

    return {
        "claim_verdicts": [
            {
                "claim_id": 1,
                "claim_text": core_question,
                "verdict_score": weighted,
                "verdict_label": "主 Agent 综合判断",
                "verdict_reason": " ".join(reason_parts)[:500] if reason_parts else "主 Agent 已完成综合判断。",
                "key_evidence_url": evidence_url,
            }
        ],
        "classification": weighted,
        "reason": " ".join(reason_parts)[:800] if reason_parts else "主 Agent 已基于当前证据完成综合判断。",
        "evidence_url": evidence_url,
        "publisher_conclusion": publisher_conclusion,
        "beneficiary_conclusion": beneficiary_conclusion,
    }


def _normalize_final_result(final_result: dict, state: dict) -> dict:
    fallback = _finalize_from_state(state)
    normalized = dict(fallback)
    normalized.update({key: value for key, value in final_result.items() if value is not None})
    normalized["classification"] = max(0.0, min(10.0, _safe_float(normalized.get("classification"), fallback["classification"])))
    if not isinstance(normalized.get("claim_verdicts"), list) or not normalized["claim_verdicts"]:
        normalized["claim_verdicts"] = fallback["claim_verdicts"]
    normalized.setdefault("reason", fallback["reason"])
    normalized.setdefault("evidence_url", fallback["evidence_url"])
    normalized.setdefault("publisher_conclusion", fallback["publisher_conclusion"])
    normalized.setdefault("beneficiary_conclusion", fallback["beneficiary_conclusion"])
    return normalized


def _source_type_from_profile(profile: dict[str, Any]) -> str:
    tier = str(profile.get("tier", "") or "").lower()
    mapping = {
        "official": "官方",
        "mainstream": "主流媒体",
        "professional": "专业媒体",
        "portal": "门户平台",
        "self_media": "自媒体",
    }
    return mapping.get(tier, "其他")


def _append_evidence_from_source_lookup(state: dict, result: dict) -> None:
    for profile in result.get("profiles", [])[:10]:
        url = profile.get("url", "")
        if not url:
            continue
        existing = state["evidence_by_url"].get(url, {"url": url})
        merged = {
            "title": existing.get("title", ""),
            "url": url,
            "summary": existing.get("summary", ""),
            "snippet": existing.get("snippet", ""),
            "source_type": _source_type_from_profile(profile),
            "composite_score": max(existing.get("composite_score", 0), float(profile.get("credibility_score", 3) or 3) / 2),
            "reason": existing.get("reason", "") or str(profile.get("reason", "") or "已比对信源知识库。"),
            "domain": profile.get("domain", existing.get("domain", "")),
            "source_name": profile.get("name", existing.get("source_name", "")),
            "matched_segments": existing.get("matched_segments", []),
            "extracted_numbers": existing.get("extracted_numbers", []),
            "origin_tool": existing.get("origin_tool", "source_credibility_lookup"),
            "source_profile": profile,
        }
        merged.update(existing)
        merged["source_type"] = _source_type_from_profile(profile)
        merged["domain"] = profile.get("domain", merged.get("domain", ""))
        merged["source_name"] = profile.get("name", merged.get("source_name", ""))
        merged["source_profile"] = profile
        state["evidence_by_url"][url] = merged
        state.setdefault("source_profiles_by_url", {})[url] = profile


def _knowledge_base_direct_result(state: dict) -> dict | None:
    knowledge_base = state.get("knowledge_base") or {}
    if not knowledge_base.get("found") or not knowledge_base.get("can_determine"):
        return None
    references = knowledge_base.get("reference_items", []) or []
    if not references:
        return None
    top_reference = references[0]
    classification = max(0.0, min(10.0, _safe_float(top_reference.get("is_real"), 5.0)))
    reason = _clip(top_reference.get("reason", "知识库命中高相似历史案例，可直接给出参考结论。"), 800)
    claim_text = _clip(top_reference.get("news_text") or state.get("news_text", ""), 200)
    evidence_url = top_reference.get("evidence_url", "")
    label = "知识库高相似结果"
    return {
        "claim_verdicts": [
            {
                "claim_id": 1,
                "claim_text": claim_text,
                "verdict_score": classification,
                "verdict_label": label,
                "verdict_reason": reason,
                "key_evidence_url": evidence_url,
            }
        ],
        "classification": classification,
        "reason": reason,
        "evidence_url": evidence_url,
        "publisher_conclusion": "知识库中已有高相似参考记录",
        "beneficiary_conclusion": "基于历史案例直接收束，传播动机暂不单独展开",
    }


def _top_evidence_urls(state: dict, limit: int = 3) -> list[str]:
    ranked = sorted(
        state.get("evidence_by_url", {}).values(),
        key=lambda item: (
            _safe_float(item.get("composite_score"), 0),
            len(item.get("matched_segments", []) or []),
            item.get("evidence_granularity", item.get("origin_tool", "")) == "full_page",
        ),
        reverse=True,
    )
    urls = []
    for item in ranked:
        url = item.get("url", "")
        if url and url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _llm_json_call(system_prompt: str, payload: dict[str, Any], max_tokens: int = 1800) -> dict[str, Any]:
    response = main_agent_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or "{}"
    parsed = _parse_tool_output(content)
    return parsed if isinstance(parsed, dict) else {}


def _fallback_supplemental_decision(state: dict) -> dict[str, Any]:
    assessment = _derive_internal_assessment(state)
    action = assessment.get("recommended_action", "continue_search")
    should_stop = action == "finalize"
    return {
        "evidence_sufficient": should_stop,
        "should_stop": should_stop,
        "reason": assessment.get("recommended_action_reason", "继续按照当前证据状态推进。"),
        "updated_search_queries": [] if should_stop else [_default_search_query(state)],
        "priority_urls": _top_evidence_urls(state, limit=2) if assessment.get("contradiction_risk") else [],
        "should_cross_verify": bool(state.get("evidence_by_url")),
    }


def _run_supplemental_search_agent(state: dict, round_index: int) -> dict[str, Any]:
    payload = {
        "round_index": round_index,
        "news_text": _clip(state.get("news_text", ""), 600),
        "analysis": _compact_tool_result("build_search_plan", state.get("analysis", {})) if state.get("analysis") else {},
        "evidence_candidates": _compact_state(state).get("evidence_candidates", []),
        "recent_search_rounds": _compact_state(state).get("search_rounds", []),
        "recent_segment_rounds": _compact_state(state).get("segment_rounds", []),
        "internal_assessment": _derive_internal_assessment(state),
    }
    try:
        decision = _llm_json_call(SUPPLEMENTAL_SEARCH_AGENT_PROMPT, payload, max_tokens=1200)
    except Exception:
        decision = _fallback_supplemental_decision(state)

    fallback = _fallback_supplemental_decision(state)
    queries = [
        _normalize_query(item)
        for item in decision.get("updated_search_queries", [])
        if _normalize_query(item)
    ]
    priority_urls = [url for url in decision.get("priority_urls", []) if isinstance(url, str) and url.strip()]
    return {
        "evidence_sufficient": bool(decision.get("evidence_sufficient", fallback["evidence_sufficient"])),
        "should_stop": bool(decision.get("should_stop", fallback["should_stop"])),
        "reason": _clip(decision.get("reason", fallback["reason"]), 300),
        "updated_search_queries": queries[:4] or fallback["updated_search_queries"],
        "priority_urls": _dedupe_keep_order(priority_urls or fallback["priority_urls"])[:3],
        "should_cross_verify": bool(decision.get("should_cross_verify", fallback["should_cross_verify"])),
    }


def _run_analysis_agent(state: dict) -> dict[str, Any]:
    payload = {
        "news_text": _clip(state.get("news_text", ""), 800),
        "analysis": _compact_tool_result("build_search_plan", state.get("analysis", {})) if state.get("analysis") else {},
        "knowledge_base": _compact_tool_result("knowledge_base_lookup", state.get("knowledge_base", {})) if state.get("knowledge_base") else {},
        "cross_verify_result": _compact_tool_result("cross_source_verify", state.get("cross_verify_result", {})) if state.get("cross_verify_result") else {},
        "evidence_candidates": _compact_state(state).get("evidence_candidates", []),
        "full_pages": [_compact_tool_result("read_full_page", item) for item in state.get("full_pages", [])[-3:]],
        "internal_assessment": _derive_internal_assessment(state),
    }
    try:
        return _llm_json_call(ANALYSIS_AGENT_PROMPT, payload, max_tokens=2200)
    except Exception:
        return _finalize_from_state(state)


async def _call_mcp_tool_once(tool_name: str, arguments: dict[str, Any]) -> dict:
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = PROJECT_ROOT if not python_path else PROJECT_ROOT + os.pathsep + python_path
    result = None
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_tools_server"],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

    if result is None:
        raise RuntimeError(f"MCP 工具调用失败: {tool_name}")

    text_output = "\n".join(item.text for item in result.content if getattr(item, "type", "") == "text")
    if not text_output.strip():
        return {}
    return _parse_tool_output(text_output)


async def _list_mcp_tools_once() -> list[dict[str, Any]]:
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = PROJECT_ROOT if not python_path else PROJECT_ROOT + os.pathsep + python_path
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_tools_server"],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

    tools: list[dict[str, Any]] = []
    for tool in result.tools:
        tools.append(
            {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "input_schema": getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}},
            }
        )
    return tools


class MCPToolInvoker:
    def list_tools(self) -> list[dict[str, Any]]:
        return asyncio.run(_list_mcp_tools_once())

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict:
        return asyncio.run(_call_mcp_tool_once(tool_name, arguments))


class MainFactCheckAgent:
    def __init__(self, tools: MCPToolInvoker):
        self.tools = tools

    def _tool(self, tool_name: str, arguments: dict[str, Any]) -> dict:
        return self.tools.call_tool(tool_name, arguments)

    def _available_tools(self, use_db: bool) -> tuple[list[dict], list[str]]:
        tool_defs = self.tools.list_tools()
        return _build_runtime_tool_schemas(tool_defs, use_db)

    def run_stream(
        self,
        news_text: str,
        img_base64: str | None = None,
        image_path: str | None = None,
        use_db: bool = True,
    ):
        state = {
            "news_text": news_text,
            "img_base64": img_base64,
            "image_path": image_path,
            "use_db": use_db,
            "knowledge_base": None,
            "analysis": None,
            "search_results": [],
            "segment_results": [],
            "full_pages": [],
            "cross_verify_result": None,
            "agent_evidence_catalog": None,
            "finalize_and_store_result": None,
            "tool_results": {},
            "evidence_by_url": {},
            "source_profiles_by_url": {},
            "history": [],
        }
        _, available_tools = self._available_tools(use_db)
        plan = {**FOUR_AGENT_PLAN, "available_tools": available_tools, "max_search_rounds": MAX_SEARCH_ROUNDS}
        tool_step = 0

        def _already_read(url: str) -> bool:
            return any(page.get("url", "") == url for page in state.get("full_pages", []))

        def _record_history(tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
            nonlocal tool_step
            tool_step += 1
            state["history"].append(
                {
                    "step": tool_step,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "observation": _compact_tool_result(tool_name, result),
                }
            )

        def _invoke_tool(tool_name: str, arguments: dict[str, Any]):
            prepared = _prepare_tool_arguments(state, tool_name, arguments)
            tool_source = _tool_source(tool_name)
            yield _evt(
                "tool_input",
                tool_source,
                f"{tool_source} 输入",
                detail=json.dumps(prepared, ensure_ascii=False, indent=2),
            )
            result = self._tool(tool_name, prepared)
            _update_state_from_tool(state, tool_name, result)
            _record_history(tool_name, prepared, result)
            yield _evt(
                "tool_output",
                tool_source,
                f"{tool_source} 输出",
                detail=json.dumps(result, ensure_ascii=False, indent=2),
                data=result,
            )
            return result

        yield _evt(
            "thinking",
            "main_agent",
            "四智能体编排已启动，将按初始阶段、搜索阶段和验证阶段顺序推进。",
            detail=json.dumps(plan, ensure_ascii=False, indent=2),
        )

        final_result = None

        if use_db and "knowledge_base_lookup" in available_tools:
            yield _evt(
                "thinking",
                "initial_search_agent",
                "初始阶段先执行新闻知识库比对，判断是否存在可直接复用的高相似结论。",
            )
            yield_result = yield from _invoke_tool("knowledge_base_lookup", {"news_text": news_text})
            if yield_result is not None:
                final_result = _knowledge_base_direct_result(state)

        if final_result is None:
            yield _evt(
                "thinking",
                "initial_search_agent",
                "初步搜索智能体正在构建首轮搜索计划。",
                detail=json.dumps({"news_text": _clip(news_text, 400)}, ensure_ascii=False, indent=2),
            )
            if "build_search_plan" in available_tools:
                yield from _invoke_tool("build_search_plan", {})
            elif not state.get("analysis"):
                state["analysis"] = {
                    "analysis_summary": "未提供搜索规划工具，已回退到默认检索建议。",
                    "core_question": _clip(news_text, 120),
                    "verification_dimensions": ["来源链路", "核心事实", "关键细节"],
                    "focus_points": ["原始发布", "直接证据", "是否存在冲突"],
                    "search_queries": [_default_search_query(state)],
                    "risk_hypotheses": [],
                    "img_description": "",
                    "missing_information": [],
                }

            current_queries = _dedupe_keep_order((state.get("analysis") or {}).get("search_queries", []))[:4] or [_default_search_query(state)]
            for round_index in range(1, MAX_SEARCH_ROUNDS + 1):
                used_queries = {
                    _normalize_query(entry.get("arguments", {}).get("query", ""))
                    for entry in state.get("history", [])
                    if entry.get("tool_name") == "search_result_list"
                }
                round_queries = [query for query in current_queries if _normalize_query(query) not in used_queries]
                if not round_queries:
                    fallback_query = _default_search_query(state)
                    round_queries = [fallback_query] if fallback_query else []
                if not round_queries or "search_result_list" not in available_tools:
                    break

                yield _evt(
                    "thinking",
                    "evidence_collection_agent",
                    f"收集证据智能体开始第 {round_index} 轮取证。",
                    detail=json.dumps({"queries": round_queries[:3], "state": _compact_state(state)}, ensure_ascii=False, indent=2),
                )

                for query in round_queries[:3]:
                    search_result = yield from _invoke_tool("search_result_list", {"query": query})
                    urls = [item.get("url", "") for item in search_result.get("results", []) if item.get("url")][:5]
                    if urls and "source_credibility_lookup" in available_tools:
                        yield from _invoke_tool("source_credibility_lookup", {"urls": urls})
                    if urls and "extract_relevant_segments" in available_tools:
                        search_terms = _dedupe_keep_order([query, *_default_search_terms(state)])[:6]
                        segment_result = yield from _invoke_tool("extract_relevant_segments", {"urls": urls, "search_terms": search_terms})
                        if int(segment_result.get("matched_url_count") or 0) == 0 and "read_full_page" in available_tools and urls:
                            yield from _invoke_tool("read_full_page", {"url": urls[0]})

                supplement_decision = _run_supplemental_search_agent(state, round_index)
                yield _evt(
                    "thinking",
                    "supplemental_search_agent",
                    f"补充搜索智能体完成第 {round_index} 轮证据评估。",
                    detail=json.dumps(supplement_decision, ensure_ascii=False, indent=2),
                )

                if not supplement_decision.get("evidence_sufficient") and "read_full_page" in available_tools:
                    for url in supplement_decision.get("priority_urls", [])[:2]:
                        if url and not _already_read(url):
                            yield from _invoke_tool("read_full_page", {"url": url})

                if supplement_decision.get("should_stop") or supplement_decision.get("evidence_sufficient"):
                    break
                current_queries = supplement_decision.get("updated_search_queries", []) or [_default_search_query(state)]

            if state.get("evidence_by_url") and "cross_source_verify" in available_tools:
                yield _evt(
                    "thinking",
                    "main_agent",
                    "搜索阶段结束，进入验证阶段的结构化证据交叉验证。",
                    detail=json.dumps(_compact_state(state), ensure_ascii=False, indent=2),
                )
                yield from _invoke_tool("cross_source_verify", {})

            yield _evt(
                "thinking",
                "analysis_agent",
                "分析智能体正在补读关键网页并生成最终判断。",
                detail=json.dumps({"top_urls": _top_evidence_urls(state, limit=2)}, ensure_ascii=False, indent=2),
            )
            if "read_full_page" in available_tools:
                for url in _top_evidence_urls(state, limit=2):
                    if url and not _already_read(url):
                        yield from _invoke_tool("read_full_page", {"url": url})

            final_result = _normalize_final_result(_run_analysis_agent(state), state)
            yield _evt(
                "thinking",
                "analysis_agent",
                "分析智能体已完成最终裁决。",
                detail=json.dumps(final_result, ensure_ascii=False, indent=2),
            )

        if final_result is None:
            if state.get("finalize_and_store_result"):
                final_result = state["finalize_and_store_result"]
            else:
                final_result = _finalize_from_state(state)

        final_result = _normalize_final_result(final_result, state)
        final_result["has_contradiction"] = state["cross_verify_result"].get("has_contradiction", False) if state.get("cross_verify_result") else False
        final_result["cross_verify_score"] = state["cross_verify_result"].get("cross_verify_score", 5.0) if state.get("cross_verify_result") else 5.0

        yield _evt(
            "thinking",
            "main_agent",
            "主 Agent 已完成最终裁决，系统将自动保存结果。",
            detail=json.dumps(final_result, ensure_ascii=False, indent=2),
        )
        save_args = {
            "news_text": news_text,
            "image_path": image_path,
            "classification": final_result.get("classification"),
            "reason": final_result.get("reason", ""),
            "evidence_url": final_result.get("evidence_url", ""),
        }
        yield _evt("tool_input", "tool_save", "保存工具输入", detail=json.dumps(save_args, ensure_ascii=False, indent=2))
        save_result = self._tool("save_result", save_args)
        yield _evt("tool_output", "tool_save", "保存工具输出", detail=json.dumps(save_result, ensure_ascii=False, indent=2), data=save_result)

        final = {
            "classification": final_result.get("classification", 5),
            "reason": final_result.get("reason", ""),
            "evidence_url": final_result.get("evidence_url", ""),
            "claim_verdicts": final_result.get("claim_verdicts", []),
            "has_contradiction": final_result.get("has_contradiction", False),
            "cross_verify_score": final_result.get("cross_verify_score", 5.0),
            "publisher_conclusion": final_result.get("publisher_conclusion", "未知"),
            "beneficiary_conclusion": final_result.get("beneficiary_conclusion", "未知"),
            "plan": plan,
        }
        yield _evt("final", "main_agent", "最终判断结果", data=final)
        yield _evt(
            "db_status",
            "sub_agent_2",
            "数据库写入成功",
            data={"db_updated": save_result.get("db_updated", False), "new_id": save_result.get("new_id")},
        )


def fact_check_pipeline_stream(
    news_text: str,
    img_base64: str | None = None,
    image_path: str | None = None,
    use_db: bool = True,
):
    agent = MainFactCheckAgent(MCPToolInvoker())
    yield from agent.run_stream(news_text, img_base64, image_path, use_db=use_db)


def _evt(event_type: str, source: str, message: str, detail: str = "", data: dict | None = None) -> dict:
    evt = {"type": event_type, "source": source, "message": message}
    if detail:
        evt["detail"] = detail
    if data is not None:
        evt["data"] = data
    return evt


def fact_check_pipeline(
    news_text: str,
    img_base64: str | None = None,
    image_path: str | None = None,
    use_db: bool = True,
) -> dict:
    result = {"steps": {}, "final_result": None, "db_updated": False, "new_id": None}
    for evt in fact_check_pipeline_stream(news_text, img_base64, image_path, use_db=use_db):
        event_type = evt["type"]
        if event_type == "tool_output":
            step_key = evt["source"].replace("tool_", "")
            result["steps"][step_key] = evt.get("data", {})
        elif event_type == "final":
            result["final_result"] = evt.get("data", {})
        elif event_type == "db_status":
            data = evt.get("data", {})
            result["db_updated"] = data.get("db_updated", False)
            result["new_id"] = data.get("new_id")
    return result
