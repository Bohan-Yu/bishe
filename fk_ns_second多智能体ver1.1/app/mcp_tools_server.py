"""
MCP 工具服务。

将事实核查能力封装为多工具集合，供主 Agent 自主编排调用。
"""

from fastmcp import FastMCP

from app.cross_source_verification import cross_source_verify as run_cross_source_verify
from app.tools import (
    tool_a_knowledge_base_lookup,
    tool_c1_search_results,
    tool_c2_extract_relevant_segments,
    tool_c3_read_full_page,
    tool_d_source_credibility_lookup,
    tool_save_result,
)


mcp = FastMCP("fake-news-fact-check")


@mcp.tool
def knowledge_base_lookup(news_text: str, img_base64: str | None = None) -> dict:
    """工具 A：比对知识库，提取类似声明的判别信息作为参考。

    输入:
    - news_text: 用户提交的待核查新闻文本。
    - img_base64: 可选，图片的 base64 内容。

    输出:
    - found: 是否命中知识库候选。
    - can_determine: 是否可直接用知识库结果给出高相似参考结论。
    - reference_items: 相似历史记录列表。
    - reference_summary: 对知识库命中的简要总结。
    """
    return tool_a_knowledge_base_lookup(news_text, img_base64)


# @mcp.tool
# def build_search_plan(
#     news_text: str,
#     img_base64: str | None = None,
#     knowledge_references: list[dict] | None = None,
#     framework_result: dict | None = None,
# ) -> dict:
#     """工具 B：输出轻量级初始分析与检索建议。"""
#     return tool_b_search_plan(news_text, img_base64, knowledge_references)


@mcp.tool
def search_result_list(
    queries: list[str] | None = None,
    max_results: int = 6,
    exclude_urls: list[str] | None = None,
) -> dict:
    """工具 C1：一轮内执行多个搜索词并返回聚合结果列表。

    输入:
    - queries: 本轮要执行的多个搜索句子。
    - max_results: 每个搜索关键词最多返回多少条候选结果。
    - exclude_urls: 需要排除的已处理链接列表。

    输出:
    - queries: 实际执行的查询句子。
    - query_summaries: 每个查询词的结果统计。
    - results: 搜索结果数组，每条含 title、snippet、url、domain。
    - result_count: 返回结果数量。
    - domain_distribution: 域名分布统计。
    - source_learning: 搜索轮次中的来源观察信息。
    """
    return tool_c1_search_results(queries=queries, max_results=max_results, exclude_urls=exclude_urls)


@mcp.tool
def extract_relevant_segments(
    urls: list[str],
    keyword_pairs: list[list[str]] | None = None,
    search_terms: list[str] | None = None,
) -> dict:
    """工具 C2：提取网页正文并返回命中关键词对任一词的句子列表。

    输入:
    - urls: 需要抽取正文的网页链接列表。
    - keyword_pairs: 可选，按关键词对进行句子匹配。
    - search_terms: 可选，按搜索词列表进行句子匹配。

    输出:
    - urls: 实际处理的链接列表。
    - results: 每个网页的命中片段、标题、数字提取结果。
    - matched_url_count: 命中网页数量。
    - sentence_list: 聚合后的句子列表。
    - error: 抽取失败时的错误信息。
    """
    return tool_c2_extract_relevant_segments(urls=urls, keyword_pairs=keyword_pairs, search_terms=search_terms)


@mcp.tool
def read_full_page(url: str) -> dict:
    """工具 C3：返回目标网址的网页全文。

    输入:
    - url: 需要读取全文的网页链接。

    输出:
    - url: 实际读取的链接。
    - title: 网页标题。
    - raw_content: 网页正文全文。
    - content_length: 正文长度。
    - error 或 warning: 读取失败或降级抓取时的说明。
    """
    return tool_c3_read_full_page(url=url)


@mcp.tool
def source_credibility_lookup(urls: list[str]) -> dict:
    """工具 D：比对信源知识库，返回候选网址的信誉画像。

    输入:
    - urls: 候选来源链接列表。

    输出:
    - profiles: 每个链接的信源画像。
    - profile_count: 命中画像数量。
    - tier_distribution: 层级分布。
    - country_distribution: 国家分布。
    """
    return tool_d_source_credibility_lookup(urls=urls)


@mcp.tool
def cross_source_verify(
    claim_text: str,
    evidence_list: list[str],
    web_list: list[str],
    scores: list,
    reasons: list[str],
    evidence_items: list[dict] | None = None,
    agent_evidence_catalog: dict | None = None,
) -> dict:
    """工具 E：对多来源证据进行信源分层、去重和立场交叉验证。

    输入:
    - claim_text: 待验证的核心声明。
    - evidence_list: 证据摘要列表。
    - web_list: 证据链接列表。
    - scores: 证据原始分数列表。
    - reasons: 证据原始理由列表。
    - evidence_items: 可选，结构化证据项列表。
    - agent_evidence_catalog: 可选，主智能体生成的同源先验与证据备注。

    输出:
    - independent_source_count: 独立来源数量。
    - total_source_count: 总来源数量。
    - relevant_source_count: 高相关证据数量。
    - tier_distribution_readable: 可读层级分布。
    - stance_analysis: 立场分析结果。
    - numeric_analysis: 数字口径分析结果。
    - traceable_evidence: 可追踪证据明细。
    - cross_verify_score: 交叉验证加权得分。
    - has_contradiction: 是否存在矛盾。
    """
    return run_cross_source_verify(
        claim_text,
        evidence_list,
        web_list,
        scores,
        reasons,
        evidence_items=evidence_items,
        agent_evidence_catalog=agent_evidence_catalog,
    )


@mcp.tool
def save_result(
    news_text: str,
    image_path: str | None,
    classification: float | int | None,
    reason: str,
    evidence_url: str = "",
) -> dict:
    """保存核查结果，不参与最终裁决。

    输入:
    - news_text: 新闻文本。
    - image_path: 图片路径。
    - classification: 最终评分。
    - reason: 结论说明。
    - evidence_url: 关键证据链接。

    输出:
    - db_updated: 是否成功写入数据库。
    - new_id: 新记录 ID。
    - vector_store_warning: 向量索引更新失败时的提示。
    """
    return tool_save_result(
        news_text=news_text,
        image_path=image_path,
        classification=classification,
        reason=reason,
        evidence_url=evidence_url,
    )


if __name__ == "__main__":
    mcp.run()