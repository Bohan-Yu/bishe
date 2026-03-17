import json
import unittest
from unittest.mock import patch

from app.agent import MainFactCheckAgent, _derive_internal_assessment, _prepare_tool_arguments, _safe_parse_tool_output


class FakeTools:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [
            {
                "name": "knowledge_base_lookup",
                "description": "数据库参考",
                "input_schema": {"type": "object", "properties": {"news_text": {"type": "string"}}},
            },
            {
                "name": "search_result_list",
                "description": "搜索结果列表",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "queries": {"type": "array"}}},
            },
            {
                "name": "source_credibility_lookup",
                "description": "信源知识库",
                "input_schema": {"type": "object", "properties": {"urls": {"type": "array"}}},
            },
            {
                "name": "extract_relevant_segments",
                "description": "片段抽取",
                "input_schema": {"type": "object", "properties": {"urls": {"type": "array"}, "search_terms": {"type": "array"}}},
            },
            {
                "name": "cross_source_verify",
                "description": "交叉验证",
                "input_schema": {"type": "object", "properties": {"claim_text": {"type": "string"}}},
            },
            {
                "name": "read_full_page",
                "description": "全文读取",
                "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}},
            },
            {
                "name": "save_result",
                "description": "保存结果",
                "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}},
            },
        ]

    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        if tool_name == "search_result_list":
            queries = arguments.get("queries") or ([arguments.get("query")] if arguments.get("query") else [])
            return {
                "query": queries[0] if queries else "",
                "queries": queries,
                "result_count": 1,
                "query_variants": queries,
                "query_summaries": [{"query": item, "result_count": 1, "domains": {"example.com": 1}, "error": ""} for item in queries],
                "domain_distribution": {"example.com": 1},
                "source_learning": {"observed_domain_count": 1, "saved_domain_count": 0},
                "results": [
                    {
                        "title": "Example Fact Check",
                        "snippet": "权威来源指出该说法不成立。",
                        "url": "https://example.com/fact-check",
                        "domain": "example.com",
                        "matched_queries": queries[:1],
                    }
                ],
            }
        if tool_name == "source_credibility_lookup":
            return {
                "urls": arguments.get("urls", []),
                "profile_count": 1,
                "tier_distribution": {"mainstream": 1},
                "country_distribution": {"英国": 1},
                "profiles": [
                    {
                        "url": "https://example.com/fact-check",
                        "domain": "example.com",
                        "name": "Example",
                        "tier": "mainstream",
                        "tier_label": "📰 主流媒体",
                        "credibility_score": 8,
                        "country": "英国",
                        "region": "欧洲",
                        "geo_group": "international_wire",
                        "reason": "主流媒体",
                    }
                ],
            }
        if tool_name == "extract_relevant_segments":
            return {
                "urls": arguments.get("urls", []),
                "search_terms": arguments.get("search_terms", []),
                "matched_url_count": 1,
                "results": [
                    {
                        "url": "https://example.com/fact-check",
                        "title": "Example Fact Check",
                        "matched_segments": ["该说法不成立，权威来源已辟谣。"],
                        "matched_segment_count": 1,
                        "extracted_numbers": [],
                    }
                ],
            }
        if tool_name == "cross_source_verify":
            return {
                "independent_source_count": 1,
                "total_source_count": 1,
                "relevant_source_count": 1,
                "cross_verify_score": 2.2,
                "has_contradiction": False,
                "traceable_evidence": [
                    {
                        "url": "https://example.com/fact-check",
                        "source_name": "Example",
                        "stance_label": "❌反驳",
                    }
                ],
            }
        if tool_name == "read_full_page":
            return {
                "url": arguments.get("url", ""),
                "title": "Example Fact Check",
                "raw_content": "完整正文说明该说法不成立。",
                "content_length": 14,
            }
        if tool_name == "save_result":
            return {"db_updated": True, "new_id": 42}
        raise AssertionError(f"unexpected tool call: {tool_name}")


class NativeToolCallingAgentTests(unittest.TestCase):
    def test_safe_parse_tool_output_falls_back_for_plain_text(self):
        parsed = _safe_parse_tool_output("这不是 JSON，只是一段说明文字", default={"status": "fallback"})
        self.assertEqual(parsed["status"], "fallback")
        self.assertIn("这不是 JSON", parsed["raw_text"])

    def test_extract_segments_arguments_coerce_stringified_lists(self):
        state = {
            "news_text": "测试新闻",
            "img_base64": None,
            "image_path": None,
            "use_db": False,
            "knowledge_base": None,
            "analysis": {
                "core_question": "核心问题",
                "focus_points": ["焦点一"],
                "verification_dimensions": ["维度一"],
            },
            "search_results": [],
            "segment_results": [],
            "full_pages": [],
            "cross_verify_result": None,
            "agent_evidence_catalog": None,
            "finalize_and_store_result": None,
            "tool_results": {},
            "evidence_by_url": {},
            "history": [],
        }

        prepared = _prepare_tool_arguments(
            state,
            "extract_relevant_segments",
            {
                "urls": ["https://example.com/a"],
                "search_terms": '["100", "operational", "launchers"]',
                "keyword_pairs": '[["Iran", "IDF"], ["missile", "launcher"]]',
            },
        )

        self.assertEqual(prepared["search_terms"], ["100", "operational", "launchers"])
        self.assertEqual(prepared["keyword_pairs"], [["Iran", "IDF"], ["missile", "launcher"]])

    def test_four_agent_pipeline_uses_runtime_tools_and_auto_saves(self):
        fake_tools = FakeTools()
        agent = MainFactCheckAgent(fake_tools)

        initial_plan = {
            "analysis_summary": "需要核查核心说法。",
            "core_question": "测试新闻是否属实",
            "verification_dimensions": ["来源链路", "核心事实"],
            "focus_points": ["原始发布", "直接证据"],
            "search_queries": ["测试新闻 事实核查"],
            "risk_hypotheses": ["可能为误导性转述"],
            "img_description": "",
            "missing_information": ["权威来源"],
        }
        supplemental_result = {
            "evidence_sufficient": True,
            "should_stop": True,
            "reason": "已有直接回应核心问题的证据，可以进入最终分析。",
            "updated_search_queries": [],
            "priority_urls": ["https://example.com/fact-check"],
            "should_cross_verify": True,
        }
        analysis_result = {
            "claim_verdicts": [
                {
                    "claim_id": 1,
                    "claim_text": "核心说法",
                    "verdict_score": 2.2,
                    "verdict_label": "分析智能体综合判断",
                    "verdict_reason": "已检索到权威反驳来源。",
                    "key_evidence_url": "https://example.com/fact-check",
                }
            ],
            "classification": 2.2,
            "reason": "已检索到权威反驳来源。",
            "evidence_url": "https://example.com/fact-check",
            "publisher_conclusion": "缺少可靠原始发布依据",
            "beneficiary_conclusion": "存在误导传播风险",
        }

        def fake_role_agent(self, state, agent_name, system_prompt, payload, tool_defs, allowed_tool_names, invoke_tool, max_steps=6, max_tokens=1800):
            if agent_name == "evidence_collection_agent":
                search_result = yield from invoke_tool("search_result_list", {"queries": payload["search_queries"]}, agent_name=agent_name)
                urls = [item["url"] for item in search_result.get("results", [])]
                yield from invoke_tool("source_credibility_lookup", {"urls": urls}, agent_name=agent_name)
                yield from invoke_tool(
                    "extract_relevant_segments",
                    {"urls": urls, "search_terms": payload["search_queries"]},
                    agent_name=agent_name,
                )
                return {
                    "round_summary": "已完成首轮取证。",
                    "coverage_assessment": "已有直接反驳证据。",
                    "structured_evidence_ready": True,
                    "top_evidence_urls": urls,
                    "structured_evidence_items": [
                        {
                            "url": "https://example.com/fact-check",
                            "title": "Example Fact Check",
                            "summary": "权威来源指出该说法不成立。",
                            "stance": "deny",
                            "source_name": "Example",
                            "source_type": "主流媒体",
                            "domain": "example.com",
                            "evidence_granularity": "segment_match",
                            "reason": "已有权威反驳。",
                        }
                    ],
                    "key_findings": ["权威来源指出该说法不成立。"],
                    "remaining_gaps": [],
                    "recommended_next_queries": [],
                }
            if agent_name == "analysis_agent":
                yield from invoke_tool("read_full_page", {"url": "https://example.com/fact-check"}, agent_name=agent_name)
                return analysis_result
            raise AssertionError(f"unexpected agent: {agent_name}")

        with patch("app.agent._run_initial_search_agent", return_value=initial_plan) as mocked_initial, \
             patch("app.agent._run_supplemental_search_agent", return_value=supplemental_result) as mocked_supplemental, \
             patch.object(MainFactCheckAgent, "_run_tool_calling_agent", new=fake_role_agent):
            events = list(agent.run_stream("测试新闻", use_db=False))

        self.assertEqual(mocked_initial.call_count, 1)
        self.assertEqual(mocked_supplemental.call_count, 1)

        executed_tools = [name for name, _ in fake_tools.calls]
        self.assertEqual(
            executed_tools,
            [
                "search_result_list",
                "source_credibility_lookup",
                "extract_relevant_segments",
                "cross_source_verify",
                "read_full_page",
                "save_result",
            ],
        )

        final_event = next(evt for evt in events if evt["type"] == "final")
        self.assertEqual(final_event["data"]["classification"], 2.2)
        self.assertEqual(final_event["data"]["evidence_url"], "https://example.com/fact-check")
        self.assertEqual(final_event["data"]["plan"]["agent_type"], "four-llm-agents")
        self.assertEqual(final_event["data"]["plan"]["analysis_agent"]["tools"], ["read_full_page"])

        analysis_handoff_index = next(
            index for index, evt in enumerate(events)
            if evt["type"] == "thinking" and evt["source"] == "analysis_agent" and "先做交叉验证筛证" in evt["message"]
        )
        cross_verify_output_index = next(
            index for index, evt in enumerate(events)
            if evt["type"] == "tool_output" and evt["source"] == "tool_e"
        )
        self.assertLess(analysis_handoff_index, cross_verify_output_index)

        db_event = next(evt for evt in events if evt["type"] == "db_status")
        self.assertTrue(db_event["data"]["db_updated"])
        self.assertEqual(db_event["data"]["new_id"], 42)

    def test_analysis_agent_plain_text_response_falls_back_instead_of_crashing(self):
        fake_tools = FakeTools()
        agent = MainFactCheckAgent(fake_tools)

        initial_plan = {
            "analysis_summary": "需要核查核心说法。",
            "core_question": "测试新闻是否属实",
            "verification_dimensions": ["来源链路", "核心事实"],
            "focus_points": ["原始发布", "直接证据"],
            "search_queries": ["测试新闻 事实核查"],
            "risk_hypotheses": ["可能为误导性转述"],
            "img_description": "",
            "missing_information": ["权威来源"],
        }
        supplemental_result = {
            "evidence_sufficient": True,
            "should_stop": True,
            "reason": "已有直接回应核心问题的证据，可以进入最终分析。",
            "updated_search_queries": [],
            "priority_urls": ["https://example.com/fact-check"],
            "should_cross_verify": True,
        }

        def fake_role_agent(self, state, agent_name, system_prompt, payload, tool_defs, allowed_tool_names, invoke_tool, max_steps=6, max_tokens=1800):
            if agent_name == "evidence_collection_agent":
                search_result = yield from invoke_tool("search_result_list", {"queries": payload["search_queries"]}, agent_name=agent_name)
                urls = [item["url"] for item in search_result.get("results", [])]
                yield from invoke_tool("source_credibility_lookup", {"urls": urls}, agent_name=agent_name)
                yield from invoke_tool(
                    "extract_relevant_segments",
                    {"urls": urls, "search_terms": payload["search_queries"]},
                    agent_name=agent_name,
                )
                return {"round_summary": "已完成首轮取证。", "structured_evidence_ready": True}
            if agent_name == "analysis_agent":
                yield from invoke_tool("read_full_page", {"url": "https://example.com/fact-check"}, agent_name=agent_name)
                return {"raw_text": "我暂时无法输出严格 JSON，但倾向认为该说法不成立。"}
            raise AssertionError(f"unexpected agent: {agent_name}")

        with patch("app.agent._run_initial_search_agent", return_value=initial_plan), \
             patch("app.agent._run_supplemental_search_agent", return_value=supplemental_result), \
             patch.object(MainFactCheckAgent, "_run_tool_calling_agent", new=fake_role_agent):
            events = list(agent.run_stream("测试新闻", use_db=False))

        final_event = next(evt for evt in events if evt["type"] == "final")
        self.assertIsInstance(final_event["data"]["classification"], float)
        self.assertTrue(final_event["data"]["reason"])

    def test_internal_assessment_and_default_search_arguments_are_state_driven(self):
        state = {
            "news_text": "某消息称某地已经正式发布新规。",
            "img_base64": None,
            "image_path": None,
            "use_db": False,
            "knowledge_base": None,
            "analysis": {
                "core_question": "某地是否正式发布新规",
                "search_queries": ["某地 新规 官方 通报", "某地 新规 辟谣"],
                "focus_points": ["原始发布", "官方说明"],
                "verification_dimensions": ["来源链路", "传播立场"],
                "risk_hypotheses": ["存在误导性转述"],
                "missing_information": ["官方公告"],
            },
            "search_results": [],
            "segment_results": [],
            "full_pages": [],
            "cross_verify_result": None,
            "finalize_and_store_result": None,
            "tool_results": {},
            "evidence_by_url": {
                "https://a.example.com/post": {
                    "title": "帖子称消息属实",
                    "url": "https://a.example.com/post",
                    "summary": "消息已经证实，官方已经确认。",
                    "snippet": "官方已经确认。",
                    "source_type": "其他",
                    "composite_score": 6.6,
                    "reason": "摘要显示该说法成立。",
                    "domain": "a.example.com",
                    "source_name": "a.example.com",
                    "matched_segments": ["该消息已经确认。"],
                    "extracted_numbers": [],
                    "origin_tool": "extract_relevant_segments",
                },
                "https://b.example.com/post": {
                    "title": "另一个页面称消息不实",
                    "url": "https://b.example.com/post",
                    "summary": "该说法为谣言，官方否认。",
                    "snippet": "官方否认。",
                    "source_type": "其他",
                    "composite_score": 6.5,
                    "reason": "正文指出这是不实信息。",
                    "domain": "b.example.com",
                    "source_name": "b.example.com",
                    "matched_segments": ["官方已辟谣。"],
                    "extracted_numbers": [],
                    "origin_tool": "extract_relevant_segments",
                },
            },
            "history": [
                {
                    "step": 1,
                    "tool_name": "search_result_list",
                    "arguments": {"query": "某地 新规 官方 通报"},
                    "observation": {"result_count": 0},
                }
            ],
        }

        assessment = _derive_internal_assessment(state)
        self.assertEqual(assessment["recommended_action"], "targeted_resolution")
        self.assertTrue(assessment["contradiction_risk"])

        search_args = _prepare_tool_arguments(state, "search_result_list", {})
        self.assertEqual(search_args["query"], "某地 新规 辟谣")
        self.assertEqual(search_args["queries"], ["某地 新规 辟谣"])
        self.assertIn("https://a.example.com/post", search_args["exclude_urls"])

    def test_cross_verify_arguments_include_agent_catalog_and_evidence_metadata(self):
        state = {
            "news_text": "某消息称某地已经正式发布新规。",
            "img_base64": None,
            "image_path": None,
            "use_db": False,
            "knowledge_base": None,
            "analysis": {"core_question": "某地是否正式发布新规"},
            "search_results": [],
            "segment_results": [],
            "full_pages": [],
            "cross_verify_result": None,
            "agent_evidence_catalog": None,
            "finalize_and_store_result": None,
            "tool_results": {},
            "evidence_by_url": {
                "https://a.example.com/post": {
                    "title": "帖子称消息属实",
                    "url": "https://a.example.com/post",
                    "summary": "消息已经证实，官方已经确认。",
                    "snippet": "官方已经确认。",
                    "source_type": "其他",
                    "composite_score": 6.6,
                    "reason": "摘要显示该说法成立。",
                    "domain": "a.example.com",
                    "source_name": "a.example.com",
                    "matched_segments": ["该消息已经确认。"],
                    "extracted_numbers": [],
                    "origin_tool": "extract_relevant_segments",
                    "origin_tools": ["search_result_list", "extract_relevant_segments"],
                    "search_queries": ["某地 新规 官方 通报"],
                    "evidence_granularity": "segment_match",
                    "observation_count": 2,
                },
            },
            "history": [],
        }

        catalog = {
            "catalog_source": "agent_llm",
            "evidence_notes": [
                {
                    "url": "https://a.example.com/post",
                    "stance_hint": "support",
                    "source_role": "primary",
                    "originality_hint": "independent",
                    "same_source_group": "",
                    "reason": "主智能体认为该页直接回应核心问题。",
                }
            ],
            "group_hints": [],
            "ambiguous_pairs": [],
            "summary": "主智能体已完成证据编目。",
        }

        with patch("app.agent._build_agent_evidence_catalog", return_value=catalog):
            prepared = _prepare_tool_arguments(state, "cross_source_verify", {})

        self.assertEqual(prepared["agent_evidence_catalog"]["catalog_source"], "agent_llm")
        self.assertEqual(prepared["evidence_items"][0]["agent_stance_hint"], "support")
        self.assertEqual(prepared["evidence_items"][0]["evidence_granularity"], "segment_match")
        self.assertEqual(prepared["evidence_items"][0]["observation_count"], 2)

    def test_cross_verify_arguments_rebuild_invalid_model_payloads_from_state(self):
        state = {
            "news_text": "伊朗目前仅剩约100个可正常使用的导弹发射器。",
            "img_base64": None,
            "image_path": None,
            "use_db": False,
            "knowledge_base": None,
            "analysis": {"core_question": "伊朗导弹发射器剩余数量是否属实"},
            "search_results": [],
            "segment_results": [],
            "full_pages": [],
            "cross_verify_result": None,
            "agent_evidence_catalog": {
                "catalog_source": "agent_llm",
                "evidence_notes": [],
                "group_hints": [],
                "ambiguous_pairs": [],
                "summary": "",
            },
            "finalize_and_store_result": None,
            "tool_results": {},
            "evidence_by_url": {
                "https://www.i24news.tv/en/news/example": {
                    "title": "i24NEWS report",
                    "url": "https://www.i24news.tv/en/news/example",
                    "summary": "报道称伊朗只剩约100个可正常使用的导弹发射器。",
                    "snippet": "只剩约100个可正常使用的导弹发射器。",
                    "source_type": "其他",
                    "composite_score": 8.4,
                    "reason": "正文直接讨论剩余发射器数量。",
                    "domain": "i24news.tv",
                    "source_name": "i24news",
                    "matched_segments": ["只剩约100个可正常使用的导弹发射器。"],
                    "extracted_numbers": ["100"],
                    "origin_tool": "extract_relevant_segments",
                    "origin_tools": ["search_result_list", "extract_relevant_segments"],
                    "search_queries": ["伊朗 导弹发射器 100 i24news"],
                    "evidence_granularity": "segment_match",
                    "observation_count": 2,
                },
            },
            "history": [],
        }

        prepared = _prepare_tool_arguments(
            state,
            "cross_source_verify",
            {
                "evidence_list": {"i24news": {"credibility": 8}},
                "web_list": {"i24news": "https://www.i24news.tv/en/news/example"},
                "scores": {"i24news": 8.4},
                "reasons": {"i24news": "正文直接讨论剩余发射器数量。"},
            },
        )

        self.assertIsInstance(prepared["evidence_items"], list)
        self.assertEqual(prepared["evidence_items"][0]["url"], "https://www.i24news.tv/en/news/example")
        self.assertEqual(prepared["evidence_list"], ["报道称伊朗只剩约100个可正常使用的导弹发射器。"])
        self.assertEqual(prepared["web_list"], ["https://www.i24news.tv/en/news/example"])
        self.assertEqual(prepared["scores"], [8.4])
        self.assertEqual(prepared["reasons"], ["正文直接讨论剩余发射器数量。"])


if __name__ == "__main__":
    unittest.main()
