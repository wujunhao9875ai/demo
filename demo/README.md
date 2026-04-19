# AI 智能助手 - 基于 MCP + LangChain 的多模态助手

一个基于 MCP (Model Context Protocol) 和 LangChain 框架的 AI 智能助手，支持工具调用、知识库检索和流式对话。

## 功能特性

- **MCP 工具集成**：通过 MCP 协议集成多种工具，支持数学计算、日期查询等功能
- **RAG 知识库检索**：基于 FAISS 向量数据库实现本地知识库检索
- **流式对话**：支持 SSE 流式输出，实时显示 AI 回答
- **打字机效果**：前端实现逐字输出的打字机效果
- **二次元风格 UI**：精美的粒子背景和动画效果

## 项目结构

```
demo/
├── agent.py          # Agent 核心逻辑，集成 MCP 工具和 RAG 检索
├── mcp_server.py     # MCP 服务器，提供工具调用接口
├── server.py         # Web 服务端，处理 HTTP 请求和 SSE 流
├── index.html        # 前端界面，二次元风格聊天窗口
├── rag.py            # RAG 知识库处理脚本
├── rag1.py           # RAG 知识库处理脚本（备用）
├── yw_info/          # FAISS 向量数据库目录
│   ├── index.faiss   # FAISS 索引文件
│   └── index.pkl     # 文档元数据
└── 桌面运维内部知识库.pdf  # 知识库原始文档
```

## 技术栈

- **后端**：Python 3.x, Flask, LangChain, MCP
- **前端**：HTML5, CSS3, JavaScript (原生)
- **向量数据库**：FAISS
- **模型服务**：SiliconFlow API (Qwen 系列模型)

## 环境要求

- Python 3.8+
- 依赖包：
  ```
  langchain
  langchain-community
  langchain-openai
  langchain-mcp-adapters
  flask
  flask-cors
  faiss-cpu
  mcp
  uvicorn
  ```

## 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-community langchain-openai langchain-mcp-adapters flask flask-cors faiss-cpu mcp uvicorn
```

### 2. 配置 API Key

在 `agent.py` 中修改 SiliconFlow API Key：

```python
embeddings = OpenAIEmbeddings(
    model='Qwen/Qwen3-Embedding-0.6B',
    base_url="https://api.siliconflow.cn/v1",
    api_key="your-api-key-here"
)
```

### 3. 启动服务

```bash
python server.py
```

服务启动后会自动：
- 启动 MCP 服务器（端口 9901）
- 初始化 Agent
- 启动 Web 服务（端口 5000）

### 4. 访问界面

打开浏览器访问：http://localhost:5000

## 使用说明

1. 在输入框中输入问题
2. AI 助手会调用工具或检索知识库来回答问题
3. 回答以打字机效果逐字显示

## 示例问题

- 明天星期几？
- 1+1 等于多少？
- 桌面运维相关问题
- 运维工具共享平台网址是什么？

## 注意事项

- 本项目仅供学习和研究使用
- API Key 请勿泄露给他人
- 知识库文档为企业内部资料，请勿外传

## License

MIT License
