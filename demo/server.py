"""
Web 服务端 - 管理 MCP 服务器和 Agent 对话
"""
import sys
import io

# 设置标准输出编码为 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import asyncio
import subprocess
import threading
import uuid
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

import os

app = Flask(__name__)
CORS(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

mcp_process = None
agent = None
agent_lock = threading.Lock()


def start_mcp_server():
    """在新线程中启动 MCP 服务器"""
    global mcp_process
    mcp_process = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        cwd="e:/ai/新课表/第一阶段智能体/demo"
    )


async def init_agent():
    """初始化 Agent"""
    global agent
    
    from langchain.tools import tool
    
    embeddings = OpenAIEmbeddings(
        model='Qwen/Qwen3-Embedding-0.6B',
        base_url="https://api.siliconflow.cn/v1",
        api_key="/"
    )
    
    vector_store = FAISS.load_local(
        'yw_info',
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """检索信息以辅助回答问题"""
        try:
            docs = vector_store.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context, docs
        except Exception as e:
            return f"检索时出错: {str(e)}", []
    
    client = MultiServerMCPClient({
        "math": {
            "url": "http://127.0.0.1:9901/mcp",
            "transport": "streamable_http",
        }
    })
    
    tools = await client.get_tools() + [retrieve_context]
    
    line_model = init_chat_model(
        model="deepseek-ai/DeepSeek-V3.2",
        model_provider="openai",
        base_url="https://api.siliconflow.cn/v1",
        api_key="/",
        streaming=True
    )
    
    agent = create_agent(
        line_model,
        tools=tools,
        system_prompt='你是一个乐于助人的助手，你可以使用工具来检索信息辅助回答用户的问题'
    )


def sse_stream(query):
    """生成 SSE 流"""
    print(f"[DEBUG] 收到查询: {query}")
    try:
        messages = [{"role": "user", "content": query}]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_agent():
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2"
            ):
                event_type = event.get("event", "")
                
                if event_type == "on_chat_model_stream":
                    data = event.get("data", {})
                    chunk = data.get("chunk")
                    if chunk:
                        content = getattr(chunk, 'content', None)
                        if content:
                            if isinstance(content, str) and content:
                                escaped = content.replace('\n', '\\n')
                                yield f"data: {escaped}\n\n"
                            elif isinstance(content, list):
                                for item in content:
                                    if hasattr(item, 'text') and item.text:
                                        escaped = item.text.replace('\n', '\\n')
                                        yield f"data: {escaped}\n\n"
                                    elif isinstance(item, dict) and item.get('text'):
                                        escaped = item['text'].replace('\n', '\\n')
                                        yield f"data: {escaped}\n\n"
        
        try:
            async_gen = run_agent()
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
            
    except Exception as e:
        import traceback
        error_msg = f"处理请求时出错: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        yield f"data: {error_msg}\n\n"


@app.route('/')
def index():
    """提供前端页面"""
    return send_file(os.path.join(APP_ROOT, 'index.html'))


@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    data = request.json
    query = data.get('message', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    return Response(
        sse_stream(query),
        mimetype='text/event-stream; charset=utf-8',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'text/event-stream; charset=utf-8',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/status', methods=['GET'])
def status():
    """获取服务状态"""
    return jsonify({
        'mcp_running': mcp_process is not None and mcp_process.poll() is None,
        'agent_ready': agent is not None
    })


if __name__ == "__main__":
    print("正在启动 MCP 服务器...")
    start_mcp_server()
    print("MCP 服务器已启动 (端口 9901)")
    
    print("正在初始化 Agent...")
    asyncio.run(init_agent())
    print("Agent 初始化完成！")
    
    print("启动 Web 服务...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
