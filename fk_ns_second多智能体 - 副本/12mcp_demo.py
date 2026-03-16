import asyncio
import ast
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI


load_dotenv()


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("未找到 JSON 对象")
    return json.loads(text[start : end + 1])


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


class MCPToolInvoker:
    def __init__(self, mcp_session: ClientSession):
        self.mcp_session = mcp_session

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        result = await self.mcp_session.call_tool(tool_name, arguments=arguments)
        text_output = "\n".join(item.text for item in result.content if item.type == "text")
        return _parse_tool_output(text_output)


class IntakeAgent:
    def __init__(self, tools: MCPToolInvoker):
        self.tools = tools

    async def run(self, news_text: str) -> dict:
        return await self.tools.call_tool("db_check", {"news_text": news_text})


class ClaimDecompositionAgent:
    def __init__(self, tools: MCPToolInvoker):
        self.tools = tools

    async def run(self, news_text: str) -> dict:
        return await self.tools.call_tool("claim_split", {"news_text": news_text})


class EvidenceAgent:
    def __init__(self, tools: MCPToolInvoker):
        self.tools = tools

    async def run(self, claims: list[dict]) -> list[dict]:
        claim_results = []
        for claim in claims:
            search_result = await self.tools.call_tool("search_evidence", {"claim_text": claim["text"]})
            cross_result = await self.tools.call_tool(
                "cross_source_verify",
                {"claim_text": claim["text"], "evidence": search_result["evidence"]},
            )
            claim_results.append(
                {
                    "claim": claim,
                    "evidence": search_result["evidence"],
                    "score": round((search_result["score"] + cross_result["cross_score"]) / 2, 2),
                    "cross_check": cross_result,
                }
            )
        return claim_results


class SkepticAgent:
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm_client = llm_client

    async def run(self, news_text: str, claim_results: list[dict]) -> dict:
        prompt = (
            "你是质疑型子agent。请从证据薄弱、来源单一、措辞夸张三个角度挑刺。"
            " 只返回 JSON，格式为 {\"risks\": [str], \"risk_penalty\": number, \"summary\": str}。"
            f" 新闻：{news_text}。claim_results：{json.dumps(claim_results, ensure_ascii=False)}"
        )
        try:
            response = await self.llm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL2", "glm-4.5-air"),
                messages=[{"role": "user", "content": prompt}],
                timeout=20,
            )
            content = response.choices[0].message.content or ""
            result = _extract_json_object(content)
        except Exception:
            result = {
                "risks": ["部分声明仍缺少更多独立来源"],
                "risk_penalty": 0.6,
                "summary": "默认质疑：证据独立性有限",
            }
        return result


class JudgeAgent:
    def __init__(self, tools: MCPToolInvoker):
        self.tools = tools

    async def run(self, news_text: str, claim_results: list[dict], skepticism: dict) -> dict:
        result = await self.tools.call_tool(
            "final_verify",
            {"news_text": news_text, "claim_results": claim_results, "skepticism": skepticism},
        )
        return result


class StrongMainAgent:
    def __init__(self, llm_client: AsyncOpenAI, tool_invoker: MCPToolInvoker):
        self.llm_client = llm_client
        self.intake_agent = IntakeAgent(tool_invoker)
        self.claim_agent = ClaimDecompositionAgent(tool_invoker)
        self.evidence_agent = EvidenceAgent(tool_invoker)
        self.skeptic_agent = SkepticAgent(llm_client)
        self.judge_agent = JudgeAgent(tool_invoker)

    async def _make_plan(self, news_text: str) -> dict:
        prompt = (
            "你是强主agent，负责为事实核查任务制定计划。"
            " 只返回 JSON，格式为 {\"goal\": str, \"steps\": [str], \"why\": str}。"
            f" 待核查新闻：{news_text}"
        )
        try:
            response = await self.llm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL2", "glm-4.5-air"),
                messages=[{"role": "user", "content": prompt}],
                timeout=20,
            )
            content = response.choices[0].message.content or ""
            return _extract_json_object(content)
        except Exception:
            return {
                "goal": "判断新闻是否可信",
                "steps": ["入口筛查", "声明拆解", "证据验证", "质疑审查", "最终裁决"],
                "why": "先确认任务有效，再逐层验证并加入反方意见",
            }

    async def run(self, news_text: str) -> dict:
        plan = await self._make_plan(news_text)
        db_result = await self.intake_agent.run(news_text)
        if not db_result.get("is_news", True):
            return {"plan": plan, "classification": None, "reason": "输入不是新闻声明"}
        if db_result.get("found"):
            return {
                "plan": plan,
                "classification": 8.5,
                "reason": db_result.get("matched_reason", "数据库已有结果"),
            }

        split_result = await self.claim_agent.run(news_text)
        claims = split_result.get("claims", [])
        claim_results = await self.evidence_agent.run(claims)
        skepticism = await self.skeptic_agent.run(news_text, claim_results)
        final_result = await self.judge_agent.run(news_text, claim_results, skepticism)
        final_result["plan"] = plan
        final_result["subagents"] = {
            "intake": db_result,
            "claim_decomposition": split_result,
            "skeptic": skepticism,
        }
        final_result["claim_results"] = claim_results
        return final_result


async def main() -> None:
    server_params = StdioServerParameters(
        command="python",
        args=["12mcp_tools_server.py"],
        env=os.environ.copy(),
    )
    llm_client = AsyncOpenAI(
        api_key=os.getenv("SF_API_KEY"),
        base_url=os.getenv("SF_BASE_URL"),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_defs = []
            for tool in (await session.list_tools()).tools:
                tool_defs.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )

            tool_invoker = MCPToolInvoker(session)
            agent = StrongMainAgent(llm_client, tool_invoker)
            result = await agent.run("网传某地突然出台新规，已经影响大量市民生活")
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())