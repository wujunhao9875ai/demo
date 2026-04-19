"""
MCP 服务器
pip install mcp -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
"""
from mcp.server.fastmcp import FastMCP
import datetime
# 创建 FastMCP 服务器
mcp_server = FastMCP(name="Math", port=9901)


# 程序-函数是让模型来调用的-定义工具
@mcp_server.tool()
def add(a: float, b: float) -> float:
    """计算两个数值的和"""
    return a + b


@mcp_server.tool()
def multiply(a: float, b: float) -> float:
    """计算两个数值的乘积"""
    print(f"multiply: {a}, {b}", a * b)
    return a * b


@mcp_server.tool()
def get_now_datetime() -> dict:
    """获取当前时间，日期，星期信息"""
    now = datetime.datetime.now()
    now_date = now.strftime("%Y-%m-%d")
    now_time = now.strftime("%H:%M:%S")
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    now_weekday = weekdays[now.weekday()]
    return {
        "now_date": now_date,
        "now_time": now_time,
        "now_weekday": now_weekday
    }


if __name__ == "__main__":
    # 启动 MCP 服务器
    mcp_server.run(transport="streamable-http")
