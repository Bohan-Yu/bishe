# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-668d96dae1e24ec788ab0bb2859b5a5c",
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        # {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "伊朗目前仅剩约100个可正常使用的导弹发射器。这个新闻是真的还是假的？今天是什么日期？"},
    ],
    stream=False
)

print(response.choices[0].message.content)

#关于伊朗导弹发射器数量的具体军事信息属于高度机密的国防数据，我无法核实其真实性。军事装备的数量和状态通常涉及国家安全，公开报道未必准确。

# 今天是2025年1月23日。对于涉及他国军事能力的报道，建议参考多方权威来源，并注意信息发布方
# 的背景和意图。如果您对中东地区安全形势感兴趣，可以关注联合国安理会报告、国际原子能机构等国际组织的公开信息。