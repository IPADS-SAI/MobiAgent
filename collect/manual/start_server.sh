#!/bin/bash
# 启动Manual数据收集工具服务器脚本
# 使用方法: ./start_server.sh

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 进入脚本目录
cd "$SCRIPT_DIR"

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "❌ 错误：找不到Python命令"
    echo "请先安装Python 3.7或更高版本"
    exit 1
fi

# 检查依赖包
echo "📦 检查依赖包..."
python -c "import fastapi, uvicorn, pydantic" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少依赖包，正在安装..."
    pip install fastapi uvicorn pydantic uiautomator2 hmdriver2
fi

# 打印启动信息
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  📱 Manual数据收集工具服务器"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "✅ 启动服务器..."
echo "🌐 访问地址: http://localhost:9000"
echo "📝 操作说明:"
echo "   1. 在浏览器中打开上述地址"
echo "   2. 选择设备类型（Android 或 Harmony）"
echo "   3. 点击'🔗 连接设备'进行连接"
echo "   4. 连接成功后开始数据收集"
echo ""
echo "⚠️  按 Ctrl+C 停止服务器"
echo "════════════════════════════════════════════════════════════"
echo ""

# 启动服务器
python server.py
