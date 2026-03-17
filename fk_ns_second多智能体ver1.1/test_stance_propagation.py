"""
测试 stance_reason 和 source_name 字段在管道中的传播。
"""
import json
import sys
import unittest

sys.path.insert(0, ".")


class TestMergeEvidenceAgentStructuredItems(unittest.TestCase):
    """测试 _merge_evidence_agent_structured_items 将证据智能体的 stance_reason/source_name 回写到 state"""

    def test_merge_updates_existing_evidence(self):
        from app.agent import _merge_evidence_agent_structured_items

        state = {
            "evidence_by_url": {
                "https://example.com/article1": {
                    "url": "https://example.com/article1",
                    "title": "Test Article",
                    "summary": "Some content",
                    "source_name": "example.com",
                    "agent_stance_hint": "",
                },
            }
        }
        evidence_result = {
            "structured_evidence_items": [
                {
                    "url": "https://example.com/article1",
                    "stance": "support",
                    "stance_reason": "文章明确引用IDF评估称伊朗仅剩三分之一的发射器",
                    "source_name": "Israel Defense Forces评估",
                },
            ]
        }
        _merge_evidence_agent_structured_items(state, evidence_result)

        evidence = state["evidence_by_url"]["https://example.com/article1"]
        self.assertEqual(evidence["stance_reason"], "文章明确引用IDF评估称伊朗仅剩三分之一的发射器")
        self.assertEqual(evidence["source_name"], "Israel Defense Forces评估")
        self.assertEqual(evidence["agent_stance_hint"], "support")

    def test_merge_does_not_create_new_entries(self):
        from app.agent import _merge_evidence_agent_structured_items

        state = {"evidence_by_url": {}}
        evidence_result = {
            "structured_evidence_items": [
                {
                    "url": "https://nonexistent.com/article",
                    "stance_reason": "test",
                    "source_name": "test",
                },
            ]
        }
        _merge_evidence_agent_structured_items(state, evidence_result)
        self.assertEqual(len(state["evidence_by_url"]), 0)

    def test_merge_ignores_invalid_stance(self):
        from app.agent import _merge_evidence_agent_structured_items

        state = {
            "evidence_by_url": {
                "https://example.com/a": {
                    "url": "https://example.com/a",
                    "agent_stance_hint": "neutral",
                },
            }
        }
        evidence_result = {
            "structured_evidence_items": [
                {"url": "https://example.com/a", "stance": "invalid_value"},
            ]
        }
        _merge_evidence_agent_structured_items(state, evidence_result)
        self.assertEqual(state["evidence_by_url"]["https://example.com/a"]["agent_stance_hint"], "neutral")


class TestBuildStructuredEvidenceItems(unittest.TestCase):
    """测试 _build_structured_evidence_items 包含 stance_reason 字段"""

    def test_includes_stance_reason(self):
        from app.agent import _build_structured_evidence_items

        state = {
            "evidence_by_url": {
                "https://example.com/article": {
                    "url": "https://example.com/article",
                    "title": "Test",
                    "summary": "Content",
                    "source_name": "Test Source",
                    "source_type": "主流媒体",
                    "domain": "example.com",
                    "composite_score": 7.0,
                    "evidence_granularity": "segment_match",
                    "reason": "关键证据",
                    "stance_reason": "引用了IDF官方评估数据",
                    "matched_segments": [],
                    "search_queries": [],
                },
            },
            "source_profiles_by_url": {},
        }
        items = _build_structured_evidence_items(state)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["stance_reason"], "引用了IDF官方评估数据")
        self.assertEqual(items[0]["source_name"], "Test Source")


class TestCrossVerifyEvidenceItemsIncludeStanceReason(unittest.TestCase):
    """测试 _cross_verify_evidence_items 包含 stance_reason"""

    def test_stance_reason_propagates(self):
        from app.agent import _cross_verify_evidence_items

        state = {
            "agent_evidence_catalog": {"evidence_notes": []},
            "evidence_by_url": {
                "https://example.com/a": {
                    "url": "https://example.com/a",
                    "title": "Title A",
                    "domain": "example.com",
                    "summary": "Summary A",
                    "snippet": "Snippet A",
                    "source_type": "主流",
                    "source_name": "Reuters",
                    "composite_score": 8.0,
                    "reason": "Reason A",
                    "stance_reason": "直接引用Reuters消息源",
                    "matched_segments": [],
                    "extracted_numbers": [],
                    "search_queries": [],
                    "origin_tools": ["search_result_list"],
                    "evidence_granularity": "search_result",
                    "observation_count": 1,
                },
            },
        }
        items = _cross_verify_evidence_items(state)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["stance_reason"], "直接引用Reuters消息源")


class TestBuildSimpleStanceAnalysis(unittest.TestCase):
    """测试 _build_simple_stance_analysis 使用 record 的 stance_reason"""

    def test_uses_record_stance_reason(self):
        from app.cross_source_verification import _build_simple_stance_analysis

        records = [
            {
                "index": 1,
                "agent_stance_hint": "support",
                "stance_reason": "报道明确提到70%的发射能力已丧失",
                "source_name_hint": "i24news",
            },
            {
                "index": 2,
                "agent_stance_hint": "neutral",
                "stance_reason": "仅提到伊朗拥有数百枚导弹",
                "source_name_hint": "Mideast Journal",
            },
        ]
        result = _build_simple_stance_analysis("test claim", records)
        stances = result["stances"]
        self.assertEqual(stances[0]["reason"], "报道明确提到70%的发射能力已丧失")
        self.assertEqual(stances[0]["source_description"], "i24news")
        self.assertEqual(stances[1]["reason"], "仅提到伊朗拥有数百枚导弹")
        self.assertEqual(stances[1]["source_description"], "Mideast Journal")


class TestBuildSimpleTraceableEvidence(unittest.TestCase):
    """测试 _build_simple_traceable_evidence 的 stance_reason 回退逻辑"""

    def test_uses_fallback_stance_reason(self):
        from app.cross_source_verification import _build_simple_traceable_evidence

        records = [
            {
                "index": 1,
                "url": "https://example.com/a",
                "title": "Test",
                "summary": "Content",
                "score": 7.0,
                "stance_reason": "来自证据智能体的立场原因",
                "source_name_hint": "Test Source",
                "source_profile": {
                    "name": "Example",
                    "domain": "example.com",
                    "country": "美国",
                    "tier": "mainstream",
                    "tier_label": "📰 主流媒体",
                },
                "provenance": {},
            },
        ]
        # stance_analysis 没有 reason，应回退到 record 的 stance_reason
        stance_analysis = {
            "stances": [
                {"index": 1, "stance": "support", "reason": "", "source_description": ""},
            ],
        }
        items = _build_simple_traceable_evidence(records, stance_analysis)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["stance_reason"], "来自证据智能体的立场原因")
        # source_description 应回退到 source_name_hint
        self.assertEqual(items[0]["source_description"], "Test Source")

    def test_analysis_agent_reason_takes_priority(self):
        from app.cross_source_verification import _build_simple_traceable_evidence

        records = [
            {
                "index": 1,
                "url": "https://example.com/a",
                "title": "Test",
                "summary": "Content",
                "score": 7.0,
                "stance_reason": "证据智能体的立场原因",
                "source_name_hint": "Some Source",
                "source_profile": {
                    "name": "Example",
                    "domain": "example.com",
                    "country": "美国",
                    "tier": "mainstream",
                    "tier_label": "📰 主流媒体",
                },
                "provenance": {},
            },
        ]
        stance_analysis = {
            "stances": [
                {
                    "index": 1,
                    "stance": "support",
                    "reason": "分析智能体的更详细立场原因",
                    "source_description": "详细来源描述",
                },
            ],
        }
        items = _build_simple_traceable_evidence(records, stance_analysis)
        self.assertEqual(items[0]["stance_reason"], "分析智能体的更详细立场原因")
        self.assertEqual(items[0]["source_description"], "详细来源描述")


class TestMergeEvidenceAnnotationsToCrossVerify(unittest.TestCase):
    """测试分析智能体的 evidence_annotations 能合并回 traceable_evidence"""

    def test_merge_annotations(self):
        from app.agent import _merge_evidence_annotations_to_cross_verify

        state = {
            "cross_verify_result": {
                "traceable_evidence": [
                    {
                        "index": 1,
                        "url": "https://example.com/a",
                        "stance": "neutral",
                        "stance_label": "➖中立",
                        "stance_reason": "",
                        "source_description": "",
                    },
                ],
                "stance_analysis": {
                    "stances": [
                        {"index": 1, "stance": "neutral", "reason": "", "source_description": ""},
                    ],
                },
            }
        }
        final_result = {
            "evidence_annotations": [
                {
                    "url": "https://example.com/a",
                    "stance": "support",
                    "stance_reason": "分析确认该报道引用IDF评估",
                    "source_description": "i24news引用以色列国防军评估",
                },
            ]
        }
        _merge_evidence_annotations_to_cross_verify(state, final_result)

        trace = state["cross_verify_result"]["traceable_evidence"][0]
        self.assertEqual(trace["stance"], "support")
        self.assertEqual(trace["stance_label"], "✅支持")
        self.assertEqual(trace["stance_reason"], "分析确认该报道引用IDF评估")
        self.assertEqual(trace["source_description"], "i24news引用以色列国防军评估")


if __name__ == "__main__":
    unittest.main()
