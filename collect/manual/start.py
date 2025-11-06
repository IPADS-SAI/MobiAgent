#!/usr/bin/env python3
"""
Manual Data Collection Tool - Start Server Script

使用方法:
    python start.py              # 默认：localhost:9000
    python start.py --host 0.0.0.0 --port 8080   # 自定义主机和端口
"""

import sys
import os
import argparse
import logging

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

import uvicorn
from server import app

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Manual Data Collection Tool Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python start.py                          # 启动服务器（localhost:9000）
  python start.py --host 0.0.0.0          # 所有网卡可访问
  python start.py --port 8080              # 自定义端口
  python start.py --reload                 # 开发模式（自动重载）
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务器绑定的主机地址 (默认: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=9000,
        help='服务器绑定的端口 (默认: 9000)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='启用自动重载模式 (开发环境)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='工作进程数 (默认: 1)'
    )
    
    args = parser.parse_args()
    
    # 打印启动信息
    print("")
    print("=" * 70)
    print("  📱 Manual 数据收集工具服务器")
    print("=" * 70)
    print("")
    print("🚀 启动信息:")
    print(f"   🌐 服务地址: http://localhost:{args.port}")
    if args.host == '0.0.0.0':
        print(f"   🌍 远程访问: http://<your-ip>:{args.port}")
    else:
        print(f"   📍 绑定地址: {args.host}:{args.port}")
    print("")
    print("📖 使用步骤:")
    print("   1. 打开浏览器，访问上述地址")
    print("   2. 选择设备类型（Android 或 Harmony）")
    print("   3. 点击'🔗 连接设备'进行连接")
    print("   4. 连接成功后开始数据收集")
    print("")
    print("⚙️  当前模式:")
    if args.reload:
        print("   ♻️  开发模式（自动重载已启用）")
    else:
        print("   🏭 生产模式")
    print(f"   👷 工作进程: {args.workers}")
    print("")
    print("⏹️  按 Ctrl+C 停止服务器")
    print("=" * 70)
    print("")
    
    # 启动服务器
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n✅ 服务器已停止")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 服务器启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
