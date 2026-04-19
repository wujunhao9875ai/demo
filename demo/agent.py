# agent_rag 把检索制作成工具
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient



# 检索
embeddings = OpenAIEmbeddings(
    model='Qwen/Qwen3-Embedding-0.6B',
    base_url="https://api.siliconflow.cn/v1",  # 模型服务接口基础地址
    api_key="/"  # 账号api密钥
)
# 加载本地知识库, 并指定允许危险的反序列化,指定相同的词嵌入模型
vector_store = FAISS.load_local('yw_info',
                                embeddings,
                                allow_dangerous_deserialization=True)
# 制作检索工具
@tool(parse_docstring=True, response_format="content_and_artifact")
def retrieve_context(query: str):
    """检索信息以辅助回答问题
    
    Args:
        query: 要检索的内容
    """
    # 从向量数据库中检索相关文档
    docs = vector_store.similarity_search(query, k=5)
    # 提取文档内容
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs

async def main():
    # 通过MCP得到MCP服务器上可以使用的工具
    # 1. 初始化多MCP客户端
    client = MultiServerMCPClient({
        "math": {
            # Make sure you start your weather server on port 8000
            "url": "http://127.0.0.1:9901/mcp",
            "transport": "streamable_http",
        }
    })
    # 2. 获取工具
    tools = await client.get_tools()+[retrieve_context]
    print(tools)
    line_model = init_chat_model(
        model="deepseek-ai/DeepSeek-V3.2",
        model_provider="openai",
        base_url="https://api.siliconflow.cn/v1",  # 模型服务接口基础地址
        api_key="/"  # 账号api密钥
    )
    agent = create_agent(line_model,
                        tools=tools,
                        system_prompt='你是一个乐于助人的助手，你可以使用工具来检索信息辅助回答用户的问题')
    while True:
        query = input("请输入您的问题(/bye退出）：")
        if query == '/bye':
            break

        async for step in agent.astream({"messages": query}, stream_mode="values"):
            step["messages"][-1].pretty_print()
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())