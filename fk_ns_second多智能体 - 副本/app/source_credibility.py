"""
中文媒体信源信誉度知识库

将常见中文新闻域名按权威性分为5个层级：
    - official    (10): 政府/官方机构网站
    - mainstream  (8) : 主流权威媒体
    - professional(6) : 专业垂直媒体
    - portal      (4) : 门户/聚合平台（内容质量参差）
    - self_media  (2) : 自媒体/个人平台（可信度低）

除内置信源表外，还会把工具 E 中 LLM 补录的未知来源分类持久化到本地知识库。
"""

import json
import os
import re
from typing import Any
from urllib.parse import urlparse

from app.config import SOURCE_CREDIBILITY_KB_PATH

# ════════════════════════════════════════════════════
#  信源信誉度数据库（域名 → 信息）
# ════════════════════════════════════════════════════

SOURCE_DB = {
    # ──── 官方/政府机构 (tier=official, score=10) ────
    "gov.cn":           {"name": "中国政府网",       "tier": "official",     "score": 10},
    "piyao.org.cn":     {"name": "中国互联网联合辟谣平台", "tier": "official", "score": 10},
    "12377.cn":         {"name": "中央网信办举报中心", "tier": "official",     "score": 10},
    "moj.gov.cn":       {"name": "司法部",           "tier": "official",     "score": 10},
    "mfa.gov.cn":       {"name": "外交部",           "tier": "official",     "score": 10},
    "nhc.gov.cn":       {"name": "国家卫健委",       "tier": "official",     "score": 10},
    "samr.gov.cn":      {"name": "国家市场监管总局",  "tier": "official",     "score": 10},
    "cac.gov.cn":       {"name": "国家互联网信息办",  "tier": "official",     "score": 10},
    "stats.gov.cn":     {"name": "国家统计局",       "tier": "official",     "score": 10},
    "customs.gov.cn":   {"name": "海关总署",         "tier": "official",     "score": 10},
    "who.int":          {"name": "世界卫生组织",     "tier": "official",     "score": 10},

    # ──── 主流权威媒体 (tier=mainstream, score=8) ────
    "xinhuanet.com":    {"name": "新华网",           "tier": "mainstream",   "score": 8},
    "news.cn":          {"name": "新华网",           "tier": "mainstream",   "score": 8},
    "people.com.cn":    {"name": "人民网",           "tier": "mainstream",   "score": 8},
    "cctv.com":         {"name": "央视网",           "tier": "mainstream",   "score": 8},
    "chinadaily.com.cn":{"name": "中国日报",         "tier": "mainstream",   "score": 8},
    "gmw.cn":           {"name": "光明网",           "tier": "mainstream",   "score": 8},
    "cnr.cn":           {"name": "央广网",           "tier": "mainstream",   "score": 8},
    "ce.cn":            {"name": "中国经济网",       "tier": "mainstream",   "score": 8},
    "youth.cn":         {"name": "中国青年网",       "tier": "mainstream",   "score": 8},
    "thepaper.cn":      {"name": "澎湃新闻",        "tier": "mainstream",   "score": 8},
    "bjnews.com.cn":    {"name": "新京报",           "tier": "mainstream",   "score": 8},
    "caixin.com":       {"name": "财新网",           "tier": "mainstream",   "score": 8},
    "reuters.com":      {"name": "路透社",           "tier": "mainstream",   "score": 8},
    "apnews.com":       {"name": "美联社",           "tier": "mainstream",   "score": 8},
    "bbc.com":          {"name": "BBC",              "tier": "mainstream",   "score": 8},
    "bbc.co.uk":        {"name": "BBC",              "tier": "mainstream",   "score": 8},
    "nytimes.com":      {"name": "纽约时报",         "tier": "mainstream",   "score": 8},
    "tasnimnews.com":   {"name": "Tasnim News",      "tier": "mainstream",   "score": 8},
    "farsnews.ir":      {"name": "Fars News",        "tier": "mainstream",   "score": 8},
    "irna.ir":          {"name": "IRNA",             "tier": "mainstream",   "score": 8},
    "presstv.ir":       {"name": "Press TV",         "tier": "mainstream",   "score": 8},
    "tehrantimes.com":  {"name": "Tehran Times",     "tier": "mainstream",   "score": 8},
    "mehrnews.com":     {"name": "Mehr News",        "tier": "mainstream",   "score": 8},
    "tass.com":         {"name": "TASS",             "tier": "mainstream",   "score": 8},
    "ria.ru":           {"name": "RIA Novosti",      "tier": "mainstream",   "score": 8},
    "rt.com":           {"name": "RT",               "tier": "mainstream",   "score": 8},
    "sputnikglobe.com": {"name": "Sputnik",          "tier": "mainstream",   "score": 8},
    "sputniknews.com":  {"name": "Sputnik",          "tier": "mainstream",   "score": 8},
    "sputniknews.cn":   {"name": "俄罗斯卫星通讯社", "tier": "mainstream",   "score": 8},

    # ──── 专业/垂直媒体 (tier=professional, score=6) ────
    "infzm.com":        {"name": "南方周末",         "tier": "professional", "score": 6},
    "jiemian.com":      {"name": "界面新闻",         "tier": "professional", "score": 6},
    "36kr.com":         {"name": "36氪",             "tier": "professional", "score": 6},
    "yicai.com":        {"name": "第一财经",         "tier": "professional", "score": 6},
    "stcn.com":         {"name": "证券时报",         "tier": "professional", "score": 6},
    "cls.cn":           {"name": "财联社",           "tier": "professional", "score": 6},
    "guancha.cn":       {"name": "观察者网",         "tier": "professional", "score": 6},
    "huanqiu.com":      {"name": "环球网",           "tier": "professional", "score": 6},
    "ifeng.com":        {"name": "凤凰网",           "tier": "professional", "score": 6},
    "zhihu.com":        {"name": "知乎",             "tier": "professional", "score": 6},
    "wikipedia.org":    {"name": "维基百科",         "tier": "professional", "score": 6},
    "baike.baidu.com":  {"name": "百度百科",         "tier": "professional", "score": 6},

    # ──── 门户/聚合平台 (tier=portal, score=4) ────
    "sina.com.cn":      {"name": "新浪",             "tier": "portal",       "score": 4},
    "sina.cn":          {"name": "新浪",             "tier": "portal",       "score": 4},
    "sohu.com":         {"name": "搜狐",             "tier": "portal",       "score": 4},
    "163.com":          {"name": "网易",             "tier": "portal",       "score": 4},
    "qq.com":           {"name": "腾讯网",           "tier": "portal",       "score": 4},
    "toutiao.com":      {"name": "今日头条",         "tier": "portal",       "score": 4},
    "baidu.com":        {"name": "百度",             "tier": "portal",       "score": 4},
    "bing.com":         {"name": "必应",             "tier": "portal",       "score": 4},
    "douyin.com":       {"name": "抖音",             "tier": "portal",       "score": 4},
    "bilibili.com":     {"name": "哔哩哔哩",         "tier": "portal",       "score": 4},

    # ──── 自媒体/低可信度 (tier=self_media, score=2) ────
    "weibo.com":        {"name": "微博",             "tier": "self_media",   "score": 2},
    "weixin.qq.com":    {"name": "微信公众号",       "tier": "self_media",   "score": 2},
    "mp.weixin.qq.com": {"name": "微信公众号",       "tier": "self_media",   "score": 2},
    "kuaishou.com":     {"name": "快手",             "tier": "self_media",   "score": 2},
    "xiaohongshu.com":  {"name": "小红书",           "tier": "self_media",   "score": 2},
    "tieba.baidu.com":  {"name": "百度贴吧",         "tier": "self_media",   "score": 2},
    "baijiahao.baidu.com": {"name": "百家号",        "tier": "self_media",   "score": 2},
}

# 信源层级中文标签
TIER_LABELS = {
    "official":     "🏛️ 官方权威",
    "mainstream":   "📰 主流媒体",
    "professional": "📋 专业媒体",
    "portal":       "🌐 门户平台",
    "self_media":   "📱 自媒体",
    "unknown":      "❓ 未知来源",
}

TIER_SCORES = {
    "official": 10,
    "mainstream": 8,
    "professional": 6,
    "portal": 4,
    "self_media": 2,
    "unknown": 3,
}

SOURCE_METADATA_HINTS = {
    "gov.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "piyao.org.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "12377.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "moj.gov.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "mfa.gov.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "nhc.gov.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "samr.gov.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "cac.gov.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "stats.gov.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "customs.gov.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "who.int": {"country": "国际组织", "region": "国际", "geo_group": "international_org"},
    "xinhuanet.com": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "news.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "people.com.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "cctv.com": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "chinadaily.com.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "gmw.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "cnr.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "ce.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "youth.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "thepaper.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "bjnews.com.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "caixin.com": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "huanqiu.com": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "ifeng.com": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "sina.com.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "sina.cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "sohu.com": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "163.com": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "qq.com": {"country": "中国", "region": "东亚", "geo_group": "china"},
    "reuters.com": {"country": "英国", "region": "欧洲", "geo_group": "international_wire"},
    "apnews.com": {"country": "美国", "region": "北美", "geo_group": "us_western"},
    "bbc.com": {"country": "英国", "region": "欧洲", "geo_group": "us_western"},
    "bbc.co.uk": {"country": "英国", "region": "欧洲", "geo_group": "us_western"},
    "nytimes.com": {"country": "美国", "region": "北美", "geo_group": "us_western"},
    "algemeiner.com": {"country": "美国", "region": "北美", "geo_group": "us_israel"},
    "mideastjournal.org": {"country": "美国", "region": "北美", "geo_group": "us_israel"},
    "criticalthreats.org": {"country": "美国", "region": "北美", "geo_group": "us_israel"},
    "jinsa.org": {"country": "美国", "region": "北美", "geo_group": "us_israel"},
    "israelhayom.com": {"country": "以色列", "region": "中东", "geo_group": "us_israel"},
    "timesofisrael.com": {"country": "以色列", "region": "中东", "geo_group": "us_israel"},
    "jpost.com": {"country": "以色列", "region": "中东", "geo_group": "us_israel"},
    "haaretz.com": {"country": "以色列", "region": "中东", "geo_group": "us_israel"},
    "tasnimnews.com": {"country": "伊朗", "region": "中东", "geo_group": "iran"},
    "farsnews.ir": {"country": "伊朗", "region": "中东", "geo_group": "iran"},
    "irna.ir": {"country": "伊朗", "region": "中东", "geo_group": "iran"},
    "presstv.ir": {"country": "伊朗", "region": "中东", "geo_group": "iran"},
    "tehrantimes.com": {"country": "伊朗", "region": "中东", "geo_group": "iran"},
    "mehrnews.com": {"country": "伊朗", "region": "中东", "geo_group": "iran"},
    "tass.com": {"country": "俄罗斯", "region": "欧洲", "geo_group": "russia"},
    "ria.ru": {"country": "俄罗斯", "region": "欧洲", "geo_group": "russia"},
    "rt.com": {"country": "俄罗斯", "region": "欧洲", "geo_group": "russia"},
    "sputnikglobe.com": {"country": "俄罗斯", "region": "欧洲", "geo_group": "russia"},
    "sputniknews.com": {"country": "俄罗斯", "region": "欧洲", "geo_group": "russia"},
    "sputniknews.cn": {"country": "俄罗斯", "region": "欧洲", "geo_group": "russia"},
}

COUNTRY_CODE_HINTS = {
    ".cn": {"country": "中国", "region": "东亚", "geo_group": "china"},
    ".il": {"country": "以色列", "region": "中东", "geo_group": "us_israel"},
    ".ir": {"country": "伊朗", "region": "中东", "geo_group": "iran"},
    ".ru": {"country": "俄罗斯", "region": "欧洲", "geo_group": "russia"},
    ".ua": {"country": "乌克兰", "region": "欧洲", "geo_group": "ukraine"},
    ".jp": {"country": "日本", "region": "东亚", "geo_group": "japan"},
    ".kr": {"country": "韩国", "region": "东亚", "geo_group": "korea"},
    ".uk": {"country": "英国", "region": "欧洲", "geo_group": "us_western"},
    ".tw": {"country": "中国台湾", "region": "东亚", "geo_group": "taiwan"},
}

COUNTRY_GROUP_HINTS = {
    "中国": {"region": "东亚", "geo_group": "china"},
    "中国台湾": {"region": "东亚", "geo_group": "taiwan"},
    "美国": {"region": "北美", "geo_group": "us_western"},
    "英国": {"region": "欧洲", "geo_group": "us_western"},
    "以色列": {"region": "中东", "geo_group": "us_israel"},
    "伊朗": {"region": "中东", "geo_group": "iran"},
    "俄罗斯": {"region": "欧洲", "geo_group": "russia"},
    "乌克兰": {"region": "欧洲", "geo_group": "ukraine"},
    "日本": {"region": "东亚", "geo_group": "japan"},
    "韩国": {"region": "东亚", "geo_group": "korea"},
    "德国": {"region": "欧洲", "geo_group": "europe"},
    "法国": {"region": "欧洲", "geo_group": "europe"},
    "卡塔尔": {"region": "中东", "geo_group": "middle_east"},
    "国际组织": {"region": "国际", "geo_group": "international_org"},
}

GEO_GROUP_LABELS = {
    "china": "中国来源圈",
    "taiwan": "台湾来源圈",
    "us_israel": "美以来源圈",
    "us_western": "英美西方来源圈",
    "international_wire": "国际通讯社",
    "international_org": "国际组织",
    "iran": "伊朗来源圈",
    "russia": "俄罗斯来源圈",
    "ukraine": "乌克兰来源圈",
    "japan": "日本来源圈",
    "korea": "韩国来源圈",
    "europe": "欧洲来源圈",
    "middle_east": "中东来源圈",
    "unknown": "未知来源圈",
}

_CUSTOM_SOURCE_DB_CACHE: dict[str, dict] | None = None


def _default_custom_payload() -> dict[str, Any]:
    return {
        "metadata": {"count": 0, "observation_count": 0},
        "entries": {},
        "observations": {},
    }


def _trim_unique_strings(values: list[str], limit: int = 5, item_limit: int = 200) -> list[str]:
    trimmed: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in trimmed:
            continue
        trimmed.append(normalized[:item_limit])
        if len(trimmed) >= limit:
            break
    return trimmed


def _extract_domain(url: str) -> str:
    """从 URL 中提取域名"""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = parsed.hostname or ""
        return host.lower().strip()
    except Exception:
        return ""


def _normalize_domain(host: str) -> str:
    return (host or "").lower().strip().lstrip("www.")


def _match_source_metadata_hint(domain: str) -> dict[str, str] | None:
    normalized = _normalize_domain(domain)
    if not normalized:
        return None
    if normalized in SOURCE_METADATA_HINTS:
        return dict(SOURCE_METADATA_HINTS[normalized])
    parts = normalized.split(".")
    for index in range(1, len(parts)):
        parent = ".".join(parts[index:])
        if parent in SOURCE_METADATA_HINTS:
            return dict(SOURCE_METADATA_HINTS[parent])
    return None


def infer_source_metadata(
    url_or_domain: str = "",
    name: str = "",
    text: str = "",
    existing: dict[str, Any] | None = None,
) -> dict[str, str]:
    existing = existing or {}
    domain = _normalize_domain(_extract_domain(url_or_domain) or url_or_domain or existing.get("domain", ""))

    explicit_country = str(existing.get("country", "")).strip()
    explicit_region = str(existing.get("region", "")).strip()
    explicit_geo_group = str(existing.get("geo_group", "")).strip()
    if explicit_country:
        derived = COUNTRY_GROUP_HINTS.get(explicit_country, {})
        return {
            "country": explicit_country,
            "region": explicit_region or derived.get("region", "未知"),
            "geo_group": explicit_geo_group or derived.get("geo_group", "unknown"),
            "country_confidence": str(existing.get("country_confidence", "explicit") or "explicit"),
        }

    matched_hint = _match_source_metadata_hint(domain)
    if matched_hint:
        return {
            **matched_hint,
            "country_confidence": "high",
        }

    for suffix, hint in COUNTRY_CODE_HINTS.items():
        if domain.endswith(suffix):
            return {
                **hint,
                "country_confidence": "medium",
            }

    combined_text = " ".join(part for part in [name, text, domain] if part).lower()
    heuristics = [
        (["israel", "以色列"], {"country": "以色列", "region": "中东", "geo_group": "us_israel"}),
        (["u.s.", "united states", "american", "美国"], {"country": "美国", "region": "北美", "geo_group": "us_western"}),
        (["china", "中国"], {"country": "中国", "region": "东亚", "geo_group": "china"}),
        (["iran", "伊朗"], {"country": "伊朗", "region": "中东", "geo_group": "iran"}),
        (["ukraine", "乌克兰"], {"country": "乌克兰", "region": "欧洲", "geo_group": "ukraine"}),
        (["russia", "俄罗斯"], {"country": "俄罗斯", "region": "欧洲", "geo_group": "russia"}),
        (["japan", "日本"], {"country": "日本", "region": "东亚", "geo_group": "japan"}),
        (["korea", "韩国"], {"country": "韩国", "region": "东亚", "geo_group": "korea"}),
        (["united nations", "联合国", "who", "world health organization"], {"country": "国际组织", "region": "国际", "geo_group": "international_org"}),
    ]
    for tokens, metadata in heuristics:
        if any(token in combined_text for token in tokens):
            return {
                **metadata,
                "country_confidence": "low",
            }

    return {
        "country": "未知",
        "region": "未知",
        "geo_group": "unknown",
        "country_confidence": "low",
    }


def get_representative_domains(
    *,
    geo_group: str = "",
    country: str = "",
    tiers: list[str] | None = None,
    limit: int = 6,
) -> list[str]:
    """Return representative domains for one geo group/country, ranked by source tier."""
    normalized_geo_group = str(geo_group or "").strip()
    normalized_country = str(country or "").strip()
    allowed_tiers = {tier for tier in (tiers or []) if tier in TIER_SCORES}

    candidates: list[tuple[int, str]] = []
    for domain, info in SOURCE_DB.items():
        normalized_info = _normalize_source_entry(domain, info)
        if normalized_geo_group and normalized_info.get("geo_group") != normalized_geo_group:
            continue
        if normalized_country and normalized_info.get("country") != normalized_country:
            continue
        if allowed_tiers and normalized_info.get("tier") not in allowed_tiers:
            continue
        candidates.append((normalized_info.get("score", 0), domain))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [domain for _, domain in candidates[: max(1, limit)]]


def _normalize_source_entry(domain: str, value: dict[str, Any]) -> dict[str, Any]:
    entry = dict(value or {})
    entry.update(infer_source_metadata(domain, name=str(entry.get("name", "")), text=str(entry.get("reason", "")), existing=entry))
    return entry


def _normalize_observation_entry(domain: str, value: dict[str, Any]) -> dict[str, Any]:
    observation = dict(value or {})
    example_title = " ".join(str(item) for item in observation.get("example_titles", [])[:2])
    observation.update(
        infer_source_metadata(
            domain,
            name=str(observation.get("name", "")),
            text=" ".join([example_title, str(observation.get("last_analysis_context", ""))]).strip(),
            existing=observation,
        )
    )
    return observation


def _load_custom_source_payload() -> dict[str, Any]:
    if not os.path.exists(SOURCE_CREDIBILITY_KB_PATH):
        return _default_custom_payload()

    try:
        with open(SOURCE_CREDIBILITY_KB_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return _default_custom_payload()

    if not isinstance(payload, dict):
        return _default_custom_payload()

    normalized_entries = {}
    payload_changed = False
    for domain, value in (payload.get("entries", {}) or {}).items():
        normalized_domain = _normalize_domain(domain)
        if normalized_domain and isinstance(value, dict):
            normalized_value = _normalize_source_entry(normalized_domain, value)
            normalized_entries[normalized_domain] = normalized_value
            if normalized_domain != domain or normalized_value != value:
                payload_changed = True

    normalized_observations = {}
    for domain, value in (payload.get("observations", {}) or {}).items():
        normalized_domain = _normalize_domain(domain)
        if normalized_domain and isinstance(value, dict):
            normalized_value = _normalize_observation_entry(normalized_domain, value)
            normalized_observations[normalized_domain] = normalized_value
            if normalized_domain != domain or normalized_value != value:
                payload_changed = True

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    metadata.setdefault("count", len(normalized_entries))
    metadata.setdefault("observation_count", len(normalized_observations))

    normalized_payload = {
        "metadata": metadata,
        "entries": normalized_entries,
        "observations": normalized_observations,
    }
    if payload_changed:
        _save_custom_source_payload(normalized_payload)
    return normalized_payload


def _load_custom_source_db() -> dict[str, dict]:
    global _CUSTOM_SOURCE_DB_CACHE
    if _CUSTOM_SOURCE_DB_CACHE is not None:
        return _CUSTOM_SOURCE_DB_CACHE

    payload = _load_custom_source_payload()
    _CUSTOM_SOURCE_DB_CACHE = payload.get("entries", {})
    return _CUSTOM_SOURCE_DB_CACHE


def _save_custom_source_payload(payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(SOURCE_CREDIBILITY_KB_PATH), exist_ok=True)
    entries = payload.get("entries", {}) if isinstance(payload.get("entries"), dict) else {}
    observations = payload.get("observations", {}) if isinstance(payload.get("observations"), dict) else {}
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    metadata["count"] = len(entries)
    metadata["observation_count"] = len(observations)
    payload = {
        "metadata": metadata,
        "entries": dict(sorted(entries.items())),
        "observations": dict(sorted(observations.items())),
    }
    with open(SOURCE_CREDIBILITY_KB_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _save_custom_source_db(
    entries: dict[str, dict],
    observations: dict[str, dict] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    global _CUSTOM_SOURCE_DB_CACHE
    normalized_entries = {
        _normalize_domain(domain): _normalize_source_entry(_normalize_domain(domain), value)
        for domain, value in entries.items()
        if _normalize_domain(domain) and isinstance(value, dict)
    }
    normalized_observations = {
        _normalize_domain(domain): _normalize_observation_entry(_normalize_domain(domain), value)
        for domain, value in (observations or {}).items()
        if _normalize_domain(domain) and isinstance(value, dict)
    }
    payload = _default_custom_payload()
    payload["entries"] = normalized_entries
    payload["observations"] = normalized_observations
    payload["metadata"] = metadata or {}
    _save_custom_source_payload(payload)
    _CUSTOM_SOURCE_DB_CACHE = normalized_entries


def save_source_classification(
    url: str,
    name: str,
    tier: str,
    reason: str = "",
    source: str = "llm",
    country: str = "",
) -> bool:
    """将补录的信源分类写入本地知识库，后续可直接复用。"""
    if tier not in TIER_SCORES or tier == "unknown":
        return False

    domain = _normalize_domain(_extract_domain(url) or url)
    if not domain:
        return False

    payload = _load_custom_source_payload()
    entries = payload.get("entries", {})
    entries[domain] = _normalize_source_entry(domain, {
        "name": (name or domain)[:120],
        "tier": tier,
        "score": TIER_SCORES[tier],
        "reason": str(reason or "")[:200],
        "source": source,
        "country": str(country or "").strip()[:40],
    })
    _save_custom_source_db(
        entries,
        observations=payload.get("observations", {}),
        metadata=payload.get("metadata", {}),
    )
    return True


def observe_source_candidates(
    results: list[dict],
    query: str = "",
    analysis_context: str = "",
    source: str = "search",
) -> dict:
    """记录搜索过程中遇到的域名观察信息，用于后续稳健地补充信源知识库。"""
    payload = _load_custom_source_payload()
    observations = payload.get("observations", {})
    entries = payload.get("entries", {})

    observed_domains: list[str] = []
    new_domains: list[str] = []
    updated_count = 0
    for item in results or []:
        url = item.get("url", "") if isinstance(item, dict) else ""
        domain = _normalize_domain(_extract_domain(url) or url)
        if not domain:
            continue
        observed_domains.append(domain)
        observation = observations.get(domain, {})
        if not observation:
            new_domains.append(domain)
        observation["domain"] = domain
        observation["first_seen_query"] = observation.get("first_seen_query") or str(query or "")[:200]
        observation["last_seen_query"] = str(query or "")[:200]
        observation["last_analysis_context"] = str(analysis_context or "")[:400]
        observation["source"] = source
        observation["observed_count"] = int(observation.get("observed_count", 0)) + 1
        observation["example_urls"] = _trim_unique_strings(
            list(observation.get("example_urls", [])) + [url],
            limit=5,
            item_limit=300,
        )
        observation["example_titles"] = _trim_unique_strings(
            list(observation.get("example_titles", [])) + [item.get("title", "")],
            limit=5,
            item_limit=200,
        )
        observation["recent_queries"] = _trim_unique_strings(
            list(observation.get("recent_queries", [])) + [query],
            limit=5,
            item_limit=200,
        )
        if domain in entries:
            observation["resolved_tier"] = entries[domain].get("tier", "unknown")
            observation["country"] = entries[domain].get("country", observation.get("country", "未知"))
        observations[domain] = observation
        updated_count += 1

    _save_custom_source_db(entries, observations=observations, metadata=payload.get("metadata", {}))
    return {
        "observed_domain_count": len(set(observed_domains)),
        "new_domain_count": len(set(new_domains)),
        "updated_count": updated_count,
        "domains": _trim_unique_strings(observed_domains, limit=12, item_limit=120),
    }


def _match_domain(host: str) -> dict | None:
    """按域名从知识库中查找，支持子域名匹配"""
    if not host:
        return None

    custom_db = _load_custom_source_db()

    def choose_entry(custom_entry: dict[str, Any] | None, builtin_entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if custom_entry and custom_entry.get("source") in {"manual", "manual_override"}:
            return custom_entry
        if builtin_entry:
            return builtin_entry
        return custom_entry

    # 精确匹配
    if host in custom_db or host in SOURCE_DB:
        return choose_entry(custom_db.get(host), SOURCE_DB.get(host))

    # 去掉 www. 后再试
    clean = _normalize_domain(host)
    if clean in custom_db or clean in SOURCE_DB:
        return choose_entry(custom_db.get(clean), SOURCE_DB.get(clean))

    # 子域名匹配（如 news.sina.com.cn → sina.com.cn）
    parts = clean.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[i:])
        if parent in custom_db or parent in SOURCE_DB:
            return choose_entry(custom_db.get(parent), SOURCE_DB.get(parent))

    return None


def get_source_credibility(url: str) -> dict:
    """
    获取某个 URL 的信源信誉度信息。

    返回:
    {
        "url": "...",
        "domain": "xinhuanet.com",
        "name": "新华网",
        "tier": "mainstream",
        "tier_label": "📰 主流媒体",
        "credibility_score": 8
    }
    """
    domain = _extract_domain(url)
    info = _match_domain(domain)

    if info:
        normalized_info = _normalize_source_entry(domain, info)
        return {
            "url": url,
            "domain": domain,
            "name": normalized_info["name"],
            "tier": normalized_info["tier"],
            "tier_label": TIER_LABELS.get(normalized_info["tier"], "❓ 未知来源"),
            "credibility_score": normalized_info["score"],
            "knowledge_base_source": normalized_info.get("source", "builtin"),
            "knowledge_base_reason": normalized_info.get("reason", ""),
            "country": normalized_info.get("country", "未知"),
            "region": normalized_info.get("region", "未知"),
            "geo_group": normalized_info.get("geo_group", "unknown"),
            "country_confidence": normalized_info.get("country_confidence", "low"),
        }
    else:
        inferred = infer_source_metadata(url, name=domain)
        return {
            "url": url,
            "domain": domain,
            "name": domain or "未知",
            "tier": "unknown",
            "tier_label": TIER_LABELS["unknown"],
            "credibility_score": TIER_SCORES["unknown"],
            "country": inferred.get("country", "未知"),
            "region": inferred.get("region", "未知"),
            "geo_group": inferred.get("geo_group", "unknown"),
            "country_confidence": inferred.get("country_confidence", "low"),
        }


def batch_evaluate_sources(urls: list[str]) -> list[dict]:
    """批量评估多个 URL 的信源信誉度"""
    return [get_source_credibility(u) for u in urls]


def update_source_classification(
    domain: str,
    name: str,
    tier: str,
    reason: str = "",
    country: str = "",
) -> dict | None:
    """更新或覆盖某个域名的信源分类，持久化到本地知识库。"""
    normalized_domain = _normalize_domain(_extract_domain(domain) or domain)
    normalized_tier = str(tier or "unknown").strip().lower()
    if not normalized_domain or normalized_tier not in TIER_SCORES:
        return None

    payload = _load_custom_source_payload()
    entries = payload.get("entries", {})
    existing_entry = dict(entries.get(normalized_domain, {}))
    builtin_entry = dict(SOURCE_DB.get(normalized_domain, {}))
    base_entry = existing_entry or builtin_entry

    source_tag = str(existing_entry.get("source") or "").strip()
    if not source_tag:
        source_tag = "manual_override" if normalized_domain in SOURCE_DB else "manual"

    normalized_country = str(country or base_entry.get("country", "")).strip()[:40]
    entries[normalized_domain] = _normalize_source_entry(
        normalized_domain,
        {
            "name": str(name or base_entry.get("name") or normalized_domain).strip()[:120],
            "tier": normalized_tier,
            "score": TIER_SCORES[normalized_tier],
            "reason": str(reason or base_entry.get("reason") or "").strip()[:200],
            "source": source_tag,
            "country": normalized_country,
            "region": str(base_entry.get("region", "")).strip()[:40],
            "geo_group": str(base_entry.get("geo_group", "")).strip()[:40],
            "country_confidence": "explicit" if normalized_country else str(base_entry.get("country_confidence", "low") or "low"),
        },
    )
    _save_custom_source_db(
        entries,
        observations=payload.get("observations", {}),
        metadata=payload.get("metadata", {}),
    )
    return get_source_credibility(f"https://{normalized_domain}")


def _build_source_list_entry(domain: str, info: dict[str, Any], default_source: str) -> dict[str, Any]:
    normalized_info = _normalize_source_entry(domain, info)
    knowledge_base_source = normalized_info.get("source", default_source) or default_source
    return {
        "domain": domain,
        "name": normalized_info.get("name", domain),
        "tier": normalized_info.get("tier", "unknown"),
        "tier_label": TIER_LABELS.get(normalized_info.get("tier", "unknown"), TIER_LABELS["unknown"]),
        "credibility_score": normalized_info.get("score", TIER_SCORES["unknown"]),
        "knowledge_base_source": knowledge_base_source,
        "knowledge_base_reason": normalized_info.get("reason", ""),
        "country": normalized_info.get("country", "未知"),
        "region": normalized_info.get("region", "未知"),
        "geo_group": normalized_info.get("geo_group", "unknown"),
    }


def list_source_classifications(
    query: str = "",
    tier: str = "",
    scope: str = "all",
    country: str = "",
) -> dict:
    """列出信源分类知识库，支持搜索与按层级筛选。"""
    normalized_query = (query or "").strip().lower()
    normalized_tier = (tier or "").strip().lower()
    normalized_scope = (scope or "all").strip().lower()
    normalized_country = (country or "").strip()
    if normalized_scope not in {"all", "builtin", "custom"}:
        normalized_scope = "all"

    merged_entries: dict[str, dict[str, Any]] = {}
    if normalized_scope in {"all", "builtin"}:
        for domain, info in SOURCE_DB.items():
            merged_entries[domain] = _build_source_list_entry(domain, info, default_source="builtin")

    if normalized_scope in {"all", "custom"}:
        for domain, info in _load_custom_source_db().items():
            merged_entries[domain] = _build_source_list_entry(domain, info, default_source="custom")

    filtered_entries = []
    for entry in merged_entries.values():
        if normalized_tier and entry.get("tier") != normalized_tier:
            continue
        if normalized_country and entry.get("country", "未知") != normalized_country:
            continue
        if normalized_query:
            haystack = " ".join(
                [
                    entry.get("domain", ""),
                    entry.get("name", ""),
                    entry.get("knowledge_base_reason", ""),
                    entry.get("tier_label", ""),
                    entry.get("knowledge_base_source", ""),
                    entry.get("country", ""),
                ]
            ).lower()
            if normalized_query not in haystack:
                continue
        filtered_entries.append(entry)

    filtered_entries.sort(
        key=lambda item: (
            0 if item.get("knowledge_base_source") == "builtin" else 1,
            item.get("tier", "unknown"),
            item.get("domain", ""),
        )
    )

    tier_counts = {tier_key: 0 for tier_key in TIER_LABELS}
    source_counts = {"builtin": 0, "custom": 0}
    country_counts: dict[str, int] = {}
    for entry in filtered_entries:
        tier_counts[entry.get("tier", "unknown")] = tier_counts.get(entry.get("tier", "unknown"), 0) + 1
        source_key = "builtin" if entry.get("knowledge_base_source") == "builtin" else "custom"
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        country_key = str(entry.get("country") or "未知")
        country_counts[country_key] = country_counts.get(country_key, 0) + 1

    country_counts_readable = {
        key: value for key, value in sorted(country_counts.items(), key=lambda item: (-item[1], item[0]))
    }

    return {
        "items": filtered_entries,
        "total": len(filtered_entries),
        "query": query,
        "tier": normalized_tier,
        "scope": normalized_scope,
        "country": normalized_country,
        "tier_counts": tier_counts,
        "tier_counts_readable": {
            TIER_LABELS.get(key, key): value for key, value in tier_counts.items() if value
        },
        "country_counts": country_counts,
        "country_counts_readable": country_counts_readable,
        "available_countries": list(country_counts_readable.keys()),
        "source_counts": source_counts,
    }
