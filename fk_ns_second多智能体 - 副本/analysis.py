import base64
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# 清除可能干扰的环境变量
os.environ.pop("OPENAI_API_KEY", None)

# 读取本地图片并转换为base64
img_path = r"E:\finefake\FineFake\Image\snope\10432.jpeg"
with open(img_path, "rb") as img_file:
    img_base = base64.b64encode(img_file.read()).decode("utf-8")

# 创建LangChain的ChatOpenAI实例（配置为智谱API）
llm = ChatOpenAI(
    model="glm-4.6v",  # 注意：应该是glm-4v，不是glm-4.6v
    api_key="463507a247554a75be6d1f09df1fcbee.lxvtfyBmpbQ0DCqq",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    max_tokens=1000  # 增加token限制以容纳JSON输出
)

# 定义新闻文本
claim_text = """

Pfizer vaccine "page 132" warns not to have unprotected sex for 28 days after the second dose of the COVID-19 vaccine because "genetic manipulation" may cause birth defects.
"""

# 创建分析提示模板
analysis_prompt = PromptTemplate(
    template="""
    你是一个新闻分析专家，仅输出json格式结果，不要额外文字。
    用户请求: 文本: {claim_text}
    
    分析声明图片和文本。有三个任务：
    1. 判断是否有可疑或者情感夸大部分。如果有同时输出可疑部分描述。
       - "suspicious_check"返回True/False
       - "suspicious_section"返回描述，没有则为""
    
    2. 判断是否需要搜索证据。如果需要搜索证据，还要输出在浏览器搜索的内容。
       - "evidence_check"返回True/False
       - "evidence_search"返回搜索内容，没有则为""
    3. 图片的文本描述。
      -  "img_description":返回对图片重要内容的文本描述。
    输出格式: {{
        "suspicious_check": "True/False",
        "suspicious_section": "可疑部分描述,没有则为空",
        "evidence_check": "True/False", 
        "evidence_search": "搜索内容描述，没有则为空"
        "img_description":"返回对图片重要内容的文本描述"
    }}
    """,
    input_variables=['claim_text']
)

# 格式化提示文本
prompt_text = analysis_prompt.format(claim_text=claim_text)

# 构建HumanMessage，包含多模态内容（图片和文本分析指令）
messages = [
    HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base}"  # 添加MIME类型
                }
            },
            {
                "type": "text",
                "text": prompt_text  # 使用格式化后的分析指令
            }
        ]
    )
]

print("=== 发送给模型的内容 ===")
print(f"图片: {img_path}")
print(f"文本: {claim_text}")
print(f"分析指令: {prompt_text}")
print("-" * 50)
import time 
# 调用模型
start=time.time()
try:
    response = llm.invoke(messages)
    
    print("=== 模型原始响应 ===")
    print(response.content)
    
except Exception as e:
    print(f"❌ API调用失败: {e}")
end=time.time()
print(end-start)