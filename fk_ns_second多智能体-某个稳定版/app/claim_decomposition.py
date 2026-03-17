"""
分层验证框架子 Agent。

这里不再把新闻机械拆成很多细碎声明，而是先抽出更适合事实核查与论文展示的
四层验证框架：权威信源、专业分析、对立口径、时间事件关联。
"""

import re


def _clip(text: str, limit: int = 160) -> str:
    return (text or "").strip()[:limit]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _detect_plan_language(news_text: str) -> str:
    return "zh" if _contains_cjk(news_text) else "en"


def _clean_topic_base(news_text: str, limit: int = 90) -> str:
    text = _normalize(news_text)
    text = re.sub(r'^[\[【\("“”\']+|[\]】\)"“”\']+$', "", text)
    text = re.split(r"[。！？!?\n]", text)[0].strip() or text
    return _clip(text, limit)


def _extract_key_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text or "")[:3]


def _build_zh_dimensions(topic_base: str, key_numbers: list[str], has_image: bool) -> list[dict]:
    number_hint = f" {' '.join(key_numbers)}" if key_numbers else ""
    context_hint = " 图片来源 上下文" if has_image else " 时间线 背景"
    return [
        {
            "dimension_id": "layer-1",
            "dimension_type": "authority",
            "title": "权威信源核查",
            "objective": "先确认最早公开来源、原始发布者和主要传播链。",
            "analysis_focus": "优先判断消息是不是只在单一口径里循环转述，再看主流媒体和官方渠道是否独立提到同一事实。",
            "execution_query": f"{topic_base} 原始来源 发布者",
            "recommended_queries": [
                f'"{topic_base}" site:reuters.com OR site:bbc.com OR site:aljazeera.com',
                f'"{topic_base}" site:xinhuanet.com OR site:caixin.com OR site:thepaper.cn',
            ],
            "judgement_points": [
                "是否能定位到最早公开来源，而不是只看到二手转述。",
                "主流媒体和官方渠道是否独立描述了同一核心事实。",
            ],
            "priority": "high",
        },
        {
            "dimension_id": "layer-2",
            "dimension_type": "specialist",
            "title": "军事专业机构分析",
            "objective": "核查核心事实、关键数字和技术口径是否符合专业机构评估。",
            "analysis_focus": "当新闻涉及军力、库存、武器性能或战果时，优先看智库、研究机构和专业报告，而不是泛媒体评论。",
            "execution_query": f"{topic_base}{number_hint} 专业分析 报告",
            "recommended_queries": [
                f'"{topic_base}" site:csis.org OR site:sipri.org OR site:rand.org',
                f'"{topic_base}" filetype:pdf site:rand.org OR site:fas.org OR site:iiss.org',
            ],
            "judgement_points": [
                "不同机构的数字口径是否一致，是否区分库存、发射器、战备数量等概念。",
                "结论对应的是哪一个时间点，是否还能代表当前状态。",
            ],
            "priority": "high",
        },
        {
            "dimension_id": "layer-3",
            "dimension_type": "contrast",
            "title": "对立立场信源对比",
            "objective": "对比当事方、对立方与第三方信源的叙述差异。",
            "analysis_focus": "这一步不是机械平衡双方观点，而是识别政治宣传、威慑叙事和选择性披露带来的偏差。",
            "execution_query": f"{topic_base} 官方回应 对比",
            "recommended_queries": [
                f'"{topic_base}" site:irna.ir OR site:presstv.ir',
                f'"{topic_base}" site:state.gov OR site:idf.il OR site:defense.gov',
            ],
            "judgement_points": [
                "当事方与对立方分别强调了什么，又回避了什么。",
                "第三方权威来源是否支持其中某一方的关键说法。",
            ],
            "priority": "medium",
        },
        {
            "dimension_id": "layer-4",
            "dimension_type": "timeline",
            "title": "时间与事件关联性",
            "objective": "核查该说法是否与近期事件、打击行动或旧闻翻炒有关。",
            "analysis_focus": "很多争议新闻不是事实完全相反，而是把旧数据、旧照片或阶段性损失说成当前全局结论。",
            "execution_query": f"{topic_base}{context_hint}",
            "recommended_queries": [
                f'"{topic_base}" 近期 之后 打击 影响',
                f'"{topic_base}" 旧闻 翻炒 上下文 时间线',
            ],
            "judgement_points": [
                "新闻引用的数据和素材对应的是不是当前事件周期。",
                "是否存在把局部损失、旧闻素材或单次袭击结果扩大成整体结论的情况。",
            ],
            "priority": "medium",
        },
    ]


def _build_en_dimensions(topic_base: str, key_numbers: list[str], has_image: bool) -> list[dict]:
    number_hint = f" {' '.join(key_numbers)}" if key_numbers else ""
    context_hint = " image context" if has_image else " timeline background"
    return [
        {
            "dimension_id": "layer-1",
            "dimension_type": "authority",
            "title": "Authoritative Source Check",
            "objective": "Identify the earliest public source, original publisher, and main circulation chain.",
            "analysis_focus": "Establish whether the story comes from one repeated narrative or from independently reported primary and mainstream sources.",
            "execution_query": f"{topic_base} original source publisher",
            "recommended_queries": [
                f'"{topic_base}" site:reuters.com OR site:bbc.com OR site:apnews.com',
                f'"{topic_base}" site:aljazeera.com OR site:nytimes.com OR site:wsj.com',
            ],
            "judgement_points": [
                "Can the earliest public source be identified rather than only later repetition?",
                "Do mainstream or official sources independently describe the same core fact?",
            ],
            "priority": "high",
        },
        {
            "dimension_id": "layer-2",
            "dimension_type": "specialist",
            "title": "Military Specialist Assessment",
            "objective": "Check whether the core fact, numbers, and technical framing match specialist assessments.",
            "analysis_focus": "For military capability claims, think tanks and research institutes usually define the statistical scope more clearly than general reporting.",
            "execution_query": f"{topic_base}{number_hint} missile inventory analysis",
            "recommended_queries": [
                f'"{topic_base}" site:csis.org OR site:sipri.org OR site:rand.org',
                f'"{topic_base}" filetype:pdf site:rand.org OR site:fas.org OR site:iiss.org',
            ],
            "judgement_points": [
                "Do different institutions measure the same category, such as total inventory versus ready launchers?",
                "What is the timestamp of the estimate, and does it still fit the current event window?",
            ],
            "priority": "high",
        },
        {
            "dimension_id": "layer-3",
            "dimension_type": "contrast",
            "title": "Rival Narrative Comparison",
            "objective": "Compare statements from the involved side, the opposing side, and neutral third-party sources.",
            "analysis_focus": "The goal is not symmetry for its own sake but to identify messaging incentives, omissions, and selective framing.",
            "execution_query": f"{topic_base} official response comparison",
            "recommended_queries": [
                f'"{topic_base}" site:irna.ir OR site:presstv.ir',
                f'"{topic_base}" site:state.gov OR site:idf.il OR site:defense.gov',
            ],
            "judgement_points": [
                "What does each side emphasize or leave out?",
                "Which disputed claim is supported by neutral or specialist reporting?",
            ],
            "priority": "medium",
        },
        {
            "dimension_id": "layer-4",
            "dimension_type": "timeline",
            "title": "Time and Event Linkage",
            "objective": "Check whether the claim is tied to a recent strike, an older report, or recycled context.",
            "analysis_focus": "Many contested stories misuse older numbers, images, or local losses and reframe them as a current overall conclusion.",
            "execution_query": f"{topic_base}{context_hint}",
            "recommended_queries": [
                f'"{topic_base}" after strike current capability',
                f'"{topic_base}" old report recycled context timeline',
            ],
            "judgement_points": [
                "Does the cited material belong to the same event cycle as the current claim?",
                "Is a local or temporary loss being reframed as a full strategic conclusion?",
            ],
            "priority": "medium",
        },
    ]


def build_layered_verification_framework(
    news_text: str,
    suspicious_section: str = "",
    img_description: str = "",
) -> dict:
    plan_language = _detect_plan_language(news_text)
    topic_base = _clean_topic_base(suspicious_section or news_text)
    key_numbers = _extract_key_numbers(news_text)
    has_image = bool(_normalize(img_description))

    if plan_language == "zh":
        verification_dimensions = _build_zh_dimensions(topic_base, key_numbers, has_image)
        framework_summary = "采用分层验证而不是机械拆句：先查权威信源，再查专业机构，再对比对立口径，最后核对时间线和事件背景。"
        reasoning_style = "全中文分层计划，便于直接写入论文中的方法设计与案例分析。"
    else:
        verification_dimensions = _build_en_dimensions(topic_base, key_numbers, has_image)
        framework_summary = "Use layered verification instead of mechanical claim splitting: authoritative sources first, specialist assessment second, rival narratives third, and timeline linkage last."
        reasoning_style = "English-only layered plans keep the method description clean for paper writing and reproducible search reporting."

    return {
        "plan_language": plan_language,
        "topic_base": topic_base,
        "framework_summary": framework_summary,
        "reasoning_style": reasoning_style,
        "verification_dimensions": verification_dimensions,
    }


def format_framework_for_display(framework_result: dict) -> str:
    lines = [framework_result.get("framework_summary", "").strip()]
    for item in framework_result.get("verification_dimensions", []):
        lines.append(f"\n[{item.get('dimension_id', '?')}] {item.get('title', '')}")
        lines.append(f"目标: {item.get('objective', '')}")
        lines.append(f"分析重点: {item.get('analysis_focus', '')}")
        queries = item.get("recommended_queries", [])
        if queries:
            lines.append("推荐搜索语句:")
            for query in queries:
                lines.append(f"- {query}")
    return "\n".join(line for line in lines if line)


def decompose_claims(
    news_text: str,
    suspicious_section: str = "",
    img_description: str = "",
) -> dict:
    """兼容旧接口，但每一项代表一个核查维度。"""
    framework = build_layered_verification_framework(news_text, suspicious_section, img_description)
    claims = []
    for index, item in enumerate(framework.get("verification_dimensions", []), start=1):
        claims.append(
            {
                "claim_id": index,
                "claim_text": item.get("objective", ""),
                "claim_type": item.get("dimension_type", "factual"),
                "importance": item.get("priority", "medium"),
                "search_query": item.get("recommended_queries", [item.get("execution_query", "")])[0],
                "is_suspicious": index <= 2,
            }
        )

    return {
        "total_claims": len(claims),
        "claims": claims,
        "decomposition_summary": framework.get("framework_summary", ""),
        "plan_language": framework.get("plan_language", "zh"),
    }


def format_claims_for_display(claims_result: dict) -> str:
    lines = [f"共整理出 {claims_result.get('total_claims', 0)} 个核查维度:\n"]
    for item in claims_result.get("claims", []):
        lines.append(
            f"[{item.get('claim_id', '?')}] {item.get('claim_text', '')}\n"
            f"    维度类型: {item.get('claim_type', '')}\n"
            f"    推荐搜索语句: {item.get('search_query', '')}"
        )
    if claims_result.get("decomposition_summary"):
        lines.append(f"\n方法说明: {claims_result['decomposition_summary']}")
    return "\n".join(lines)