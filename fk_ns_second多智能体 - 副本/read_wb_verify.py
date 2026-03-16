import requests
from openai import OpenAI
import json

web_list = [
    "https://africacheck.org/fact-checks/meta-programme-fact-checks/no-pfizer-hasnt-warned-covid-vaccine-carries-risk-births",
    "https://fullfact.org/online/pfizer-covid-vaccine-sex-rules/",
    "https://apnews.com/article/fact-checking-583715031774"
]

# 1. 使用 Jina Reader API 读取网页内容
import requests

def jina_search(query: str, max_results: int = 5) -> list[dict]:
    """使用 Jina Search API 搜索中文内容"""
    resp = requests.get(
        f"https://s.jina.ai/{query}",
        headers={"Accept": "application/json", "X-Retain-Images": "none"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("data", [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "")[:300],
        })
    return results
print(jina_search("北京小客车指标2023年4月将过期"))
# # 2. 配置大模型 API (此处以 DeepSeek 为例，Kimi 或 Qwen 同理)
# # 需要执行: pip install openai
# client = OpenAI(
#     api_key="sk-668d96dae1e24ec788ab0bb2859b5a5c",  # 替换为你的真实 API Key
#     base_url="https://api.deepseek.com" # DeepSeek 的官方接口地址
# )

# def evaluate_sources(urls):
#     contents = []
#     for url in urls:
#         text = load_webpage_content_jina(url)
#         # 为了防止 token 超出，可以截取前 4000 个字符
#         contents.append(f"网址: {url}\n内容片段: {text[:4000]}")
    
#     combined_content = "\n\n---\n\n".join(contents)
    
#     prompt = f"""
#     你是一个来源可信度判断专家。请阅读以下从网页中提取的内容，综合评估每个来源的可信度。
#     评估标准包括：来源的域名权威性（如是否为知名新闻机构或事实核查组织）、文章的客观程度等。
#     打分区间为0-10，分越高表示越权威。
    
#     网页内容：
#     {combined_content}
    
#     请严格按照以下 JSON 格式输出，不要输出其他废话：
#     {{
#         "evaluations": [
#             {{"url": "...", "score": 8, "reason": "..."}}
#         ]
#     }}
#     """

#     print("正在请求大模型进行权威性分析...")
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "你是一个严格的数据分析师，仅输出合法的 JSON 代码。"},
#             {"role": "user", "content": prompt}
#         ],
#         response_format={"type": "json_object"} # 强制要求大模型返回 JSON
#     )
    
#     return json.loads(response.choices[0].message.content)

# # 运行评估
# result = evaluate_sources(web_list)
# print(json.dumps(result, indent=4, ensure_ascii=False))