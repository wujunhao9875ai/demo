# 已经完成了知识库的构建，做完了一次读取，切割，向量化，存储。之后就可以直接加载这个知识库来使用
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# 检索
embeddings = OpenAIEmbeddings(
    model='Qwen/Qwen3-Embedding-0.6B',
    base_url="https://api.siliconflow.cn/v1",  # 模型服务接口基础地址
    api_key="sk-ibvvmpsjhjxkldajxuonfinzneiwwuwbrxquozorvugaufgk"  # 账号api密钥
)
# 加载本地知识库, 并指定允许危险的反序列化,指定相同的词嵌入模型
vector_store = FAISS.load_local('yw_info',
                                embeddings,
                                allow_dangerous_deserialization=True)


# 生成
@dynamic_prompt
def prompt_with_content(request: ModelRequest):
    """将上下文信息注入状态消息"""
    last_query = request.state['messages'][-1].text  # 提取最后一次用户查询，最新的用户信息
    retrive_docs = vector_store.similarity_search(
        last_query,  # 查询文本
        k=5  # 返回前5个最相似的文档块
    )  # 从知识库中检索出最相关的文档块 List[Document]
    docs_content = "\n\n".join([doc.page_content
                                for doc in retrive_docs])  # 合并文档块内容，作为上下文
    system_message = f"你是一个乐于助人的助手，请结合以下上下文信息回答问题：\n\n{docs_content}"
    return system_message


line_model = init_chat_model(
    model="deepseek-ai/DeepSeek-V3.2",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1",  # 模型服务接口基础地址
    api_key="sk-ibvvmpsjhjxkldajxuonfinzneiwwuwbrxquozorvugaufgk"  # 账号api密钥
)
agent = create_agent(line_model, middleware=[prompt_with_content])

while True:
    query = input("请输入您的问题(/bye退出）：")
    if query == '/bye':
        break

    for step in agent.stream({"messages": query}, stream_mode="values"):
        step["messages"][-1].pretty_print()
