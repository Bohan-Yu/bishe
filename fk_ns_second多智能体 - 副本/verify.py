from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
evidence_list=["根据AP新闻的事实核查，辉瑞疫苗第132页确实提到临床试验参与者应在最后一剂研究干预后至少28天内采取措施避免怀孕或使他人怀孕，但这与声明中声称的'基因操纵导致出生缺陷'的说法不同","Snopes事实核查网站明确指出，声称'辉瑞疫苗第132页警告因基因操纵相关出生缺陷而避免无保护性行为'的说法是错误的","非洲事实核查组织确认，辉瑞公司并未警告COVID-19疫苗会导致'出生缺陷'，声明中的说法是虚假的","Full Fact组织指出，原始疫苗试验参与者确实被要求在第二剂后28天内避免无保护性行为，但这只是临床试验的标准预防措施，并非因为疫苗会导致出生缺陷","AP新闻进一步澄清，这一要求是临床试验协议的一部分，旨在确保在疫苗对孕妇的安全性得到充分研究之前采取预防措施，而不是因为疫苗会导致出生缺陷"]
web_list=["https://apnews.com/article/fact-checking-583715031774","https://www.snopes.com/fact-check/covid-vaccine-unprotected-sex/","https://africacheck.org/fact-checks/meta-programme-fact-checks/no-pfizer-hasnt-warned-covid-vaccine-carries-risk-births","https://fullfact.org/online/pfizer-covid-vaccine-sex-rules/","https://apnews.com/article/fact-checking-583715031774"]
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import Tool
from langchain.agents import create_agent
import json
import bs4
from dotenv import load_dotenv
load_dotenv()
# 直接使用网页加载器


# 简化后的提示词
claim_text="""
Pfizer vaccine "page 132" warns not to have unprotected sex for 28 days after the second dose of the COVID-19 vaccine because "genetic manipulation" may cause birth defects.
"""
evidence_list=["根据AP新闻的事实核查，辉瑞疫苗第132页确实提到临床试验参与者应在最后一剂研究干预后至少28天内采取措施避免怀孕或使他人怀孕，但这与声明中声称的'基因操纵导致出生缺陷'的说法不同","Snopes事实核查网站明确指出，声称'辉瑞疫苗第132页警告因基因操纵相关出生缺陷而避免无保护性行为'的说法是错误的","非洲事实核查组织确认，辉瑞公司并未警告COVID-19疫苗会导致'出生缺陷'，声明中的说法是虚假的","Full Fact组织指出，原始疫苗试验参与者确实被要求在第二剂后28天内避免无保护性行为，但这只是临床试验的标准预防措施，并非因为疫苗会导致出生缺陷","AP新闻进一步澄清，这一要求是临床试验协议的一部分，旨在确保在疫苗对孕妇的安全性得到充分研究之前采取预防措施，而不是因为疫苗会导致出生缺陷"]
score_list=[9, 8, 7, 8, 9]
reason_list=["AP新闻的事实核查直接针对声明中的核心内容进行澄清，提供了权威的信息来源和具体的证据支持，分数较高","Snopes作为知名的事实核查网站，其分析和结论具有较高的可信度，分数较高","非洲事实核查组织的确认增加了对声明真实性的质疑，分数较高","Full Fact组织的分析提供了对声明中预防措施背景的深入理解，分数较高","AP新闻的进一步澄清强化了对声明中错误信息的驳斥，分数较高"]
img_description="一个人戴着蓝色手套拿着一个疫苗瓶"
explanation_prompt = PromptTemplate(
    template="""
你是一个新闻分类专家。请根据以下信息来对新闻真假进行分类，并给出验证过程。
网页网址列表: {web_list}
原声明文本:{claim_text}
原声明图片描述:{img_description}
证据列表: {evidence_list}
证据权威性相关性分数列表: {score_list}
证据分数原因列表: {reason_list}

请执行：
1. 选择最合适的证据网页网址，要求该证据能够直接反驳声明中的核心错误信息，并且具有较高的权威性和相关性。
2. 给出真假得分。区间0-10，分数越高表示越真。大于等于5表示更倾向于真，低于5表示更倾向于假。
3. 给出验证原因，要求结合声明文本、图片描述、证据内容和权威性分数进行综合分析，给出详细的分析过程和最终结论。
请只输出JSON格式，格式如下：
{{
    "evidence_web_id": 最合适的证据编号,
    "classification": 真假得分,
    "reason": 原因,
}}
    """,
    input_variables=["claim_text","img_description","evidence_list","score_list","reason_list"]
)

prompt_text = explanation_prompt.format(claim_text=claim_text, img_description=img_description, evidence_list=evidence_list, score_list=score_list,reason_list=reason_list)
agent = create_agent(
    model="deepseek:deepseek-chat",
#     tools=[load_webpage_content],
#     system_prompt=research_instructions
)
for event in agent.stream(
    {"messages":[{"role": "user", 
    "content": prompt_text}]
    },
    stream_mode="values"
):
    event["messages"][-1].pretty_print()

# 最后输出编号对应的网址