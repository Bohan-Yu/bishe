"""
MCP 工具服务。

将事实核查能力封装为多工具集合，供主 Agent 自主编排调用。
"""

from fastmcp import FastMCP

from app.cross_source_verification import cross_source_verify as run_cross_source_verify
from app.tools import (
    tool_a_knowledge_base_lookup,
    tool_b_search_plan,
    tool_c1_search_results,
    tool_c2_extract_relevant_segments,
    tool_c3_read_full_page,
    tool_d_source_credibility_lookup,
    tool_f_final_verify_and_save,
    tool_save_result,
)


mcp = FastMCP("fake-news-fact-check")


@mcp.tool
def knowledge_base_lookup(news_text: str, img_base64: str | None = None) -> dict:
    """工具 A：比对知识库，提取类似声明的判别信息作为参考。"""
    return tool_a_knowledge_base_lookup(news_text, img_base64)


@mcp.tool
def build_search_plan(
    news_text: str,
    img_base64: str | None = None,
    knowledge_references: list[dict] | None = None,
    framework_result: dict | None = None,
) -> dict:
    """工具 B：输出轻量级初始分析与检索建议。"""
    return tool_b_search_plan(news_text, img_base64, knowledge_references)


@mcp.tool
def search_result_list(
    query: str,
    max_results: int = 6,
    exclude_urls: list[str] | None = None,
) -> dict:
    """工具 C1：只返回搜索引擎结果列表。"""
    return tool_c1_search_results(query=query, max_results=max_results, exclude_urls=exclude_urls)


@mcp.tool
def extract_relevant_segments(
    urls: list[str],
    keyword_pairs: list[list[str]] | None = None,
    search_terms: list[str] | None = None,
) -> dict:
    """工具 C2：提取网页正文并返回命中关键词对任一词的句子列表。"""
    return tool_c2_extract_relevant_segments(urls=urls, keyword_pairs=keyword_pairs, search_terms=search_terms)


@mcp.tool
def read_full_page(url: str) -> dict:
    """工具 C3：返回目标网址的网页全文。"""
    return tool_c3_read_full_page(url=url)


@mcp.tool
def source_credibility_lookup(urls: list[str]) -> dict:
    """工具 D：比对信源知识库，返回候选网址的信誉画像。"""
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
    """工具 E：对多来源证据进行信源分层、去重和立场交叉验证。"""
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
def finalize_and_store(
    news_text: str,
    image_path: str | None,
    img_description: str,
    analysis_result: dict,
    iteration_logs: list[dict],
    all_evidence_items: list[dict],
    cross_verify_result: dict | None = None,
) -> dict:
    """兼容旧入口：最终保存结果。"""
    return tool_f_final_verify_and_save(
        news_text=news_text,
        image_path=image_path,
        img_description=img_description,
        analysis_result=analysis_result,
        iteration_logs=iteration_logs,
        all_evidence_items=all_evidence_items,
        cross_verify_result=cross_verify_result,
    )


@mcp.tool
def save_result(
    news_text: str,
    image_path: str | None,
    classification: float | int | None,
    reason: str,
    evidence_url: str = "",
) -> dict:
    """保存核查结果，不参与最终裁决。"""
    return tool_save_result(
        news_text=news_text,
        image_path=image_path,
        classification=classification,
        reason=reason,
        evidence_url=evidence_url,
    )


if __name__ == "__main__":
    mcp.run()