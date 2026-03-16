### 本示例展示 agent 如何使用搜索引擎 tavily
### tavily: 搜索引擎：给 agent 用的，提供两种api：
# tavily_search
# tavily_extract
# https://www.tavily.com/ 
# 注册，自动获取获取 TAVILY_API_KEY="..." 
# 放到 .env 里面, 
# 1000次/月，免费的

## pip install tavily-python langchain-tavily

from langchain.agents import create_agent

from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient
from typing import Literal

tavilyClient = TavilyClient()

def internet_search(
    query: str,
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavilyClient.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# System prompt to steer the agent to be an expert researcher
research_instructions = """
你是一个专业的证据收集员。你的任务是根据声明文本和搜索内容找到相关的证据，证据能辅助验证声明文本真假。
仅输出json格式结果，不要额外文字。
    
你可以使用以下工具：
- internet_search: 用于搜索互联网信息
    
请确保：
1. 进行全面的搜索来收集信息
2. 验证搜索内容的准确性
3. 证据能辅助验证声明文本真假
4. 证据条数最多为3
5. 根据网页内容判断权威性，结合原图片文本描述和文本，给每条证据权威性和相关性进行综合打分，分数区间为0-10，并输出理由
    - 综合评估每个来源的可信度。包括来源的域名类型、发布者的粉丝过去发布内容、文章的回复内容、点赞数等等
输出证据列表和证据所在的网页。
输出格式:{{"evidence_list":[证据1,证据2...]},{"web_list":[证据1的网页，证据2的网页...]},
{"scores":[证据1的分数，证据2的分数...]},{"reasons":[证据1的理由，证据2的理由...]}}

"""

agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[internet_search],
    system_prompt=research_instructions
)

claim_text="""
Pfizer vaccine "page 132" warns not to have unprotected sex for 28 days after the second dose of the COVID-19 vaccine because "genetic manipulation" may cause birth defects.
"""
search_text="辉瑞疫苗 出生缺陷"
img_description="一个人戴着蓝色手套拿着一个疫苗瓶"
text1="原声明文本:"+claim_text+"\n原声明图片描述:"+img_description+"\n\n需要搜索的内容:"+search_text+"\n\n请根据上述内容搜索证据"
for event in agent.stream(
    {"messages":[{"role": "user", 
    "content": text1}]
    },
    stream_mode="values"
):
    event["messages"][-1].pretty_print()

# 下面是输出
# ================================ Human Message =================================

# 原声明文本:
# Pfizer vaccine "page 132" warns not to have unprotected sex for 28 days after the second dose of the COVID-19 vaccine because "genetic manipulation" may cause birth defects.

# 原声明图片描述:一个人戴着蓝色手套拿着一个疫苗瓶

# 需要搜索的内容:辉瑞疫苗 出生缺陷

# 请根据上述内容搜索证据
# ================================== Ai Message ==================================

# 我将搜索相关信息来验证这个声明。
# Tool Calls:
#   internet_search (call_00_SoKR4zkxlDdvTgdvf5NqBq5g)
#  Call ID: call_00_SoKR4zkxlDdvTgdvf5NqBq5g
#   Args:
#     query: 辉瑞疫苗 出生缺陷 无保护性行为 28天 警告
#     max_results: 5
# ================================= Tool Message =================================
# Name: internet_search

# {"query": "辉瑞疫苗 出生缺陷 无保护性行为 28天 警告", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://vocus.cc/article/61834331fd89780001d63523", "title": "專家呼籲立即停止孕婦,哺乳婦女接種mRNA新冠疫苗", "content": "thalidomide沙利竇邁最終導致數千名兒童嚴重出生缺陷，其中包括數十名出生時沒有四肢的兒童。 ... 在周產期（出生至28 天）期間對嬰兒進行了28 天的追蹤。 這些", "score": 0.4137856, "raw_content": null}, {"url": "https://www.mohw.gov.tw/cp-17-67958-1.html", "title": "網傳「FDA敗訴！輝瑞被迫公開疫苗數據！副作用足足9頁…」", "content": "接種疫苗者在接種疫苗之後，任何時間所出現任何身體上的不良情況都可以通報為不良事件，但不代表直接與疫苗接種有因果關係，仍需要經過嚴謹的審查才能確立與疫苗的相關性。", "score": 0.3179242, "raw_content": null}, {"url": "https://cofacts.tw/article/pkta0p4x7gby", "title": "輝瑞公司的疫苗文件被法院裁定必須公佈，顯示BNT新冠疫苗在做 ...", "content": "輝瑞公司的疫苗文件被法院裁定必須公佈，顯示BNT新冠疫苗在做臨床試驗的前28天，就有1223人接受疫苗注射後死亡，而且動物試驗發現所有接受疫苗注射的動物全部死亡。負責疫苗業務的輝瑞副總裁近日被法官裁定收押，原因是多重詐欺，隱瞞疫苗的真實風險。現在網路上有許多人發聲，要求逮捕更多輝瑞公司員工，因為他們犯了反人性罪行。. 假的，由法院裁定FDA應加速公開輝瑞疫苗相關文件。此事件爭點不在於是否疫苗資料公開，而在於公開的「時程」，且此事件中必須加速公開文件的單位為FDA而非輝瑞。而不良事件並未確認與疫苗接種有因果相關性，也不等同是疫苗造成的副作用或不良反應。. 疫苗注射的動物全部死亡也並非事實，「1223人」，並非輝瑞疫苗臨床實驗的參與者，而是輝瑞疫苗自獲得緊急授權後至2021年2月28日間，所收到的不良事件通報案例。. ### 資料佐證. # 【錯誤】網傳「FDA敗訴！輝瑞被迫公開疫苗數據！副作用足足9頁」、「接受實驗的46000人中有42000人有不良反應！有1200人死亡」？. 【報告將隨時更新 2022/3/11版】 一、傳言指稱的事件，為美國民間團體依資訊自由法要求美國FDA公開輝瑞疫苗提交的相關文件，而FDA公布的速度不如民間團體預期，因此由法院裁定FDA應加速公開輝瑞疫苗相關文件。此事件爭點不在於是否疫苗資料公開，而在於公開的「時程」，且此事件中必須加速公開文件的單位為FDA而非輝瑞，因此傳言稱「FDA敗訴！輝瑞被迫公開疫苗數據」、「如未有敗訴官司，輝瑞可持有秘密. 一、假訊息(詳台灣事實查核中心事實查核報告#1568「【錯誤】網傳「FDA敗訴！輝瑞被迫公開疫苗數據！副作用足足9頁」、「接受實驗的46000人中有42000人有不良反應！有1200人死亡」？」，發佈日期2022-03-11). 3. 散佈及操作誤導訊息(misinformation)，意圖破壞防疫，藉此惡化疫情，破壞社會秩序。. ### 資料佐證. 1. 【錯誤】網傳「FDA敗訴！輝瑞被迫公開疫苗數據！副作用足足9頁」、「接受實驗的46000人中有42000人有不良反應！有1200人死亡」？. # 【錯誤】網傳「FDA敗訴！輝瑞被迫公開疫苗數據！副作用足足9頁」、「接受實驗的46000人中有42000人有不良反應！有1200人死亡」？. 【報告將隨時更新 2022/3/11版】 一、傳言指稱的事件，為美國民間團體依資訊自由法要求美國FDA公開輝瑞疫苗提交的相關文件，而FDA公布的速度不如民間團體預期，因此由法院裁定FDA應加速公開輝瑞疫苗相關文件。此事件爭點不在於是否疫苗資料公開，而在於公開的「時程」，且此事件中必須加速公開文件的單位為FDA而非輝瑞，因此傳言稱「FDA敗訴！輝瑞被迫公開疫苗數據」、「如未有敗訴官司，輝瑞可持有秘密. # 假訊息查證管道. # 疫苗不實訊息專區. 以上內容由「Cofacts 真的假的」訊息回報機器人與查證協作社群提供，以 CC授權 姓名標示-相同方式分享 4.0 (CC BY-SA 4.0) 釋出，於後續重製或散布時，原社群顯名及每一則查證的出處連結皆必須被完整引用。.", "score": 0.28099155, "raw_content": null}, {"url": "https://www.pfizermedicalinformation.cn/products/medication-guidelines/cibinqo", "title": "希必可用药指南​ | 辉瑞医学信息网站", "content": "适用人群出现严重的出生缺陷和流产的可能的风险尚不明确1。 目前尚无阿布昔替尼 ... 在开始阿布昔替尼治疗前，建议患者根据现行免疫接种指南完善所有的免疫接种，包括预防性带", "score": 0.27640525, "raw_content": null}, {"url": "https://patents.google.com/patent/HK40061483A/zh", "title": "HK40061483A - 冠状病毒疫苗", "content": "本公开涉及用于预防或治疗冠状病毒感染的RNA的领域。特别地，本公开涉及用于针对冠状病毒感染进行疫苗接种并诱导有效的冠状病毒抗原特异性免疫应答如抗体和/或T细胞应答", "score": 0.07311853, "raw_content": null}], "response_time": 0.63, "request_id": "7565c265-4476-4d4b-9aa0-b398611147ce"}