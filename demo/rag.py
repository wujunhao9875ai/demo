from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # 使用FAISS保存向量
from langchain_community.docstore.in_memory import InMemoryDocstore  # 内存文档存储
import faiss
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 1.def加载知识库文档
# 加载PDF文档
pdf_loader = PyPDFLoader(r'桌面运维内部知识库.pdf')
datas = pdf_loader.load()
# print(datas)

# 2.对知识库文档进行切块
pdf_split = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n',' '],
    chunk_size=300,
    chunk_overlap=30,
)
chunks = pdf_split.split_documents(datas)
# print(chunks)
# 3.构建词嵌入模型
embeddings = OpenAIEmbeddings(
    model='Qwen/Qwen3-Embedding-0.6B',
    base_url="https://api.siliconflow.cn/v1",  # 模型服务接口基础地址
    api_key="sk-ibvvmpsjhjxkldajxuonfinzneiwwuwbrxquozorvugaufgk"  # 账号api密钥
)
# 4.构建向量数据库索引
vector_store = FAISS(
    embedding_function=embeddings,  # 词嵌入模型
    index=faiss.IndexFlatL2(1024),  # 索引对象
    docstore=InMemoryDocstore(),  # 文档存储
    index_to_docstore_id={}  # 索引到文档的映射
)
# 5. 存知识库文本数据和向量数据，查询知识库
vector_store.add_documents(chunks)
# vector_store.similarity_search(
#     '运维⼯具共享平台网址',  # 查询文本
#     k=2  # 返回前2个最相似的文档块
# )
# 6.创建语言模型对象

# 线上模型 - SiliconFlow
# 创建对话模型-使用的通过siliconflow部署的Qwen3-8B模型
line_model = init_chat_model(
    model="deepseek-ai/DeepSeek-V3.2",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1",  # 模型服务接口基础地址
    api_key="sk-ibvvmpsjhjxkldajxuonfinzneiwwuwbrxquozorvugaufgk"  # 账号api密钥
)
# 把问题和相关的知识库块文本数据，作为上下文-提示词，输入到LLM模型中
@dynamic_prompt
def prompt_with_content(request: ModelRequest):
    """将上下文信息注入状态消息"""
    last_query = request.state['messages'][-1].text  # 提取最后一次用户查询，最新的用户信息
    # print(last_query)
    retrive_docs = vector_store.similarity_search(
        last_query,  # 查询文本
        k=5  # 返回前5个最相似的文档块
    )  # 从知识库中检索出最相关的文档块 List[Document]
    docs_content = "\n\n".join([doc.page_content
                                for doc in retrive_docs])  # 合并文档块内容，作为上下文
    # print(docs_content)
    system_message = f"你是一个乐于助人的助手，请结合以下上下文信息回答问题：\n\n{docs_content}"
    return system_message
# 7.创建智能体对象
agent = create_agent(model=line_model, middleware=[prompt_with_content])
# 流式输出
query = '常用运维工具有哪些？'
for step in agent.stream({"messages": query}, stream_mode="values"):
    step["messages"][-1].pretty_print()
# 保存知识库到本地目录 yw_info
vector_store.save_local('yw_info')
