## 假新闻检测系统

当前后端已调整为“四智能体编排 + 多个 MCP 工具”的结构：

- 主 Agent: app/agent.py
- MCP 工具服务: app/mcp_tools_server.py
- Flask 入口: run.py
- 前端页面: templates/chat.html

### 向量知识库

- 向量模型: Qwen/Qwen3-Embedding-0.6B
- Embedding API: https://api.siliconflow.cn/v1
- 本地索引文件: static/vector_store/news_embeddings.json
- 检索策略: 工具 A 优先使用本地向量索引召回，相似检索失败时回退到原有词面相似度检索
- 增量更新: 每次事实核查结束并写入数据库后，会为新纪录生成向量并写回本地索引

### 安装依赖

```powershell
E:\envs\py311\python.exe -m pip install -r requirements.txt
```

### 启动项目

```powershell
python run.py
```

启动后访问:

- 聊天页面: http://127.0.0.1:5000/
- 数据库页面: http://127.0.0.1:5000/database

### 架构说明

四个智能体分别承担不同阶段职责：

- 初步搜索智能体：先结合知识库参考构建首轮搜索计划
- 收集证据智能体：围绕搜索项调用搜索、片段抽取、全文读取和信源知识库比对工具，输出结构化证据
- 补充搜索智能体：评估当前证据是否充分，不足时生成更聚焦的补充搜索项
- 分析智能体：基于结构化证据、交叉验证结果和关键网页全文做最终判断

MCP 工具集合包括：

- knowledge_base_lookup
- build_search_plan
- search_result_list
- extract_relevant_segments
- read_full_page
- source_credibility_lookup
- cross_source_verify
- finalize_and_store
- save_result

其中检索主路径采用 search_result_list → source_credibility_lookup → extract_relevant_segments，
先做纯搜索，再显式比对信源知识库，之后提取网页命中片段；只有片段不足或关键网页需要上下文时，才继续调用 read_full_page。


### 当前核查方法

主流程现在按“知识库预判 → 初步搜索计划 → 多轮证据收集 → 补充搜索决策 → 交叉验证 → 最终分析”推进。

