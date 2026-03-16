import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 加载 .env 文件中的环境变量
load_dotenv()

async def main():
    # 采用图2中的 Stdio 方式本地运行 bing-cn-mcp
    server_params_bing = StdioServerParameters(
        command="npx",
        args=["bing-cn-mcp-enhanced"],
        env=os.environ.copy()
    )

    # 增加 fetch 工具
    server_params_fetch = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
        env=os.environ.copy()
    )
    
    print("启动 MCP 客户端并连接本地服务...")
    async with stdio_client(server_params_bing) as (read_b, write_b), \
               stdio_client(server_params_fetch) as (read_f, write_f):
        async with ClientSession(read_b, write_b) as session_b, \
                   ClientSession(read_f, write_f) as session_f:
            # 初始化会话
            await session_b.initialize()
            await session_f.initialize()

            # 2. 从 MCP 服务器获取所有可用的工具，并转换为 OpenAI/GLM 兼容的格式
            session_map = {}
            glm_tools = []
            
            for session in [session_b, session_f]:
                mcp_tools = await session.list_tools()
                for tool in mcp_tools.tools:
                    session_map[tool.name] = session
                    glm_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
            print(f"成功加载工具: {[t['function']['name'] for t in glm_tools]}")

            # 3. 初始化 GLM 模型客户端 (智谱 API 兼容 OpenAI 格式)
            client = AsyncOpenAI(
                api_key=os.environ.get("SF_API_KEY"),
                base_url=os.environ.get("SF_BASE_URL")
            )

            # 4. 给大模型发送用户请求
            user_prompt = """搜"北京小客车指标2023年4月将过期",给我8条结果。"""
            # user_prompt = "帮我抓取https://www.piyao.org.cn/20221214/5f6449a021154278a7a1e05e30ec1215/c.html这一个网页的完整内容"
            messages = [{"role": "user", "content": user_prompt}]
            
            print(f"\n用户提问: {user_prompt}")
            
            response = await client.chat.completions.create(
                model=os.environ.get("LLM_MODEL2", "glm-4.5-air"),
                messages=messages,
                tools=glm_tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            
            # 5. 检查大模型是否要求调用工具
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    print(f"\n[大模型决定调用工具]: {func_name}")
                    print(f"传入参数: {func_args}")
                    
                    # 6. 在本地 MCP 客户端上真正执行计算
                    target_session = session_map.get(func_name)
                    if target_session:
                        result = await target_session.call_tool(func_name, arguments=func_args)
                    else:
                        print(f"找不到工具: {func_name}")
                        continue
                    
                    # 解析结果
                    # result.content 是一个数组，获取其中的 text
                    tool_output = "\n".join([item.text for item in result.content if item.type == "text"])
                    print(f"工具返回结果: {tool_output}")
                    
            else:
                print(f"\n模型直接回答: {response_message.content}")

if __name__ == "__main__":
    asyncio.run(main())