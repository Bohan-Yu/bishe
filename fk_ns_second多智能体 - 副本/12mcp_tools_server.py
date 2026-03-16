from fastmcp import FastMCP


mcp = FastMCP("fact-check-demo")


@mcp.tool
def db_check(news_text: str) -> dict:
    """Check whether the input looks like a news claim and whether it already exists."""
    text = (news_text or "").strip()
    return {
        "is_news": len(text) >= 8,
        "found": "新华社" in text,
        "matched_reason": "命中演示库" if "新华社" in text else ""
    }


@mcp.tool
def claim_split(news_text: str) -> dict:
    """Split the news text into a few verifiable claims."""
    claims = [
        {"id": 1, "text": "某地发布了新政策", "importance": "high"},
        {"id": 2, "text": "该政策已影响大量市民", "importance": "medium"},
    ]
    return {"claims": claims if news_text else []}


@mcp.tool
def search_evidence(claim_text: str) -> dict:
    """Return mock evidence items and a preliminary support score for one claim."""
    if "新政策" in claim_text:
        return {
            "evidence": [
                {"title": "政府公告", "source": "official", "stance": "support"},
                {"title": "主流媒体解读", "source": "mainstream", "stance": "support"},
            ],
            "score": 8.2,
        }
    return {
        "evidence": [
            {"title": "自媒体转述", "source": "self_media", "stance": "support"},
            {"title": "门户转载", "source": "portal", "stance": "uncertain"},
        ],
        "score": 5.8,
    }


@mcp.tool
def cross_source_verify(claim_text: str, evidence: list[dict]) -> dict:
    """Compute a simple cross-source consistency score from evidence items."""
    support_count = sum(1 for item in evidence if item.get("stance") == "support")
    official_count = sum(1 for item in evidence if item.get("source") == "official")
    contradiction = any(item.get("stance") == "deny" for item in evidence)
    cross_score = 8.5 if official_count and not contradiction else 6.0
    if support_count <= 1:
        cross_score -= 1.0

    return {
        "independent_source_count": len(evidence),
        "has_contradiction": contradiction,
        "cross_score": max(0.0, round(cross_score, 2)),
        "summary": f"共发现 {len(evidence)} 个来源，其中官方来源 {official_count} 个",
    }


@mcp.tool
def final_verify(news_text: str, claim_results: list[dict], skepticism: dict | None = None) -> dict:
    """Aggregate claim-level results into a final verdict."""
    if not claim_results:
        return {"classification": 5.0, "label": "信息不足", "reason": "没有可用声明结果"}

    avg_score = sum(item.get("score", 5.0) for item in claim_results) / len(claim_results)
    risk_penalty = float((skepticism or {}).get("risk_penalty", 0.0))
    final_score = max(0.0, min(10.0, avg_score - risk_penalty))
    label = "基本真实" if avg_score >= 6.5 else "存疑"
    return {
        "classification": round(final_score, 2),
        "label": label,
        "reason": f"根据 {len(claim_results)} 条声明的综合结果给出判断",
        "risk_penalty": risk_penalty,
    }


if __name__ == "__main__":
    mcp.run()