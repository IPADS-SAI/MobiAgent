#!/usr/bin/env python3
"""
模型连通性测试程序
用于测试配置中的API、MODEL、BASE_URL的可用性和连通性
"""

import sys
import time
import requests
import openai
from typing import Dict, Any, Optional, Tuple
import logging

# 导入配置
try:
    from llmconfig import API_KEY, BASE_URL, MODEL
except ImportError:
    print("错误: 无法导入llmconfig配置文件")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelConnectivityTester:
    """模型连通性测试器"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        """
        初始化测试器
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        
        # 验证配置
        self._validate_config()
        
        # 创建OpenAI客户端
        if self.api_key and self.base_url:
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except Exception as e:
                logger.error(f"创建OpenAI客户端失败: {e}")
    
    def _validate_config(self) -> None:
        """验证配置完整性"""
        logger.info("=== 配置验证 ===")
        
        if not self.api_key:
            logger.error("❌ API_KEY 未配置")
        else:
            logger.info(f"✅ API_KEY: {self.api_key[:10]}...{self.api_key[-4:]}")
        
        if not self.base_url:
            logger.error("❌ BASE_URL 未配置")
        else:
            logger.info(f"✅ BASE_URL: {self.base_url}")
        
        if not self.model:
            logger.error("❌ MODEL 未配置")
        else:
            logger.info(f"✅ MODEL: {self.model}")
        
        if not all([self.api_key, self.base_url, self.model]):
            logger.error("配置不完整，无法进行测试")
            sys.exit(1)
    
    def test_basic_connectivity(self) -> Tuple[bool, str]:
        """
        测试基础连通性（网络连接）
        
        Returns:
            (是否成功, 详细信息)
        """
        logger.info("=== 基础连通性测试 ===")
        
        try:
            # 移除/v1后缀进行基础连接测试
            test_url = self.base_url.rstrip('/v1').rstrip('/')
            
            response = requests.get(
                test_url,
                timeout=10,
                headers={'User-Agent': 'ModelConnectivityTester/1.0'}
            )
            
            logger.info(f"✅ 网络连接正常 (状态码: {response.status_code})")
            return True, f"连接成功，状态码: {response.status_code}"
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"连接失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except requests.exceptions.Timeout as e:
            error_msg = f"连接超时: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"网络测试异常: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    def test_api_endpoint(self) -> Tuple[bool, str]:
        """
        测试API端点可用性
        
        Returns:
            (是否成功, 详细信息)
        """
        logger.info("=== API端点测试 ===")
        
        try:
            # 测试模型列表端点
            models_url = f"{self.base_url.rstrip('/')}/models"
            
            response = requests.get(
                models_url,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    model_count = len(data['data'])
                    logger.info(f"✅ API端点可用，发现 {model_count} 个模型")
                    
                    # 检查目标模型是否可用
                    available_models = [model['id'] for model in data['data']]
                    if self.model in available_models:
                        logger.info(f"✅ 目标模型 '{self.model}' 可用")
                        return True, f"API端点可用，目标模型可用"
                    else:
                        logger.warning(f"⚠️ 目标模型 '{self.model}' 不在可用模型列表中")
                        logger.info(f"可用模型: {', '.join(available_models[:5])}...")
                        return True, f"API端点可用，但目标模型可能不可用"
                else:
                    logger.info("✅ API端点响应正常，但格式可能不同")
                    return True, "API端点可用"
            else:
                error_msg = f"API端点返回错误状态码: {response.status_code}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = f"API端点测试失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    def test_model_inference(self) -> Tuple[bool, str]:
        """
        测试模型推理能力
        
        Returns:
            (是否成功, 详细信息)
        """
        logger.info("=== 模型推理测试 ===")
        
        if not self.client:
            error_msg = "OpenAI客户端未初始化"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        
        try:
            # 发送简单的测试请求
            test_message = "请回复'测试成功'四个字"
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": test_message
                    }
                ],
                max_tokens=50,
                temperature=0.1,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response and response.choices:
                response_text = response.choices[0].message.content
                logger.info(f"✅ 模型推理成功 (响应时间: {response_time:.2f}秒)")
                logger.info(f"模型响应: {response_text}")
                
                return True, f"推理成功，响应时间: {response_time:.2f}秒"
            else:
                error_msg = "模型返回空响应"
                logger.error(f"❌ {error_msg}")
                return False, error_msg
                
        except openai.AuthenticationError as e:
            error_msg = f"认证失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except openai.APIError as e:
            error_msg = f"API错误: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"模型推理测试失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        运行综合测试
        
        Returns:
            测试结果字典
        """
        logger.info("开始模型连通性综合测试...")
        logger.info("=" * 50)
        
        results = {
            'config_valid': True,
            'basic_connectivity': False,
            'api_endpoint': False,
            'model_inference': False,
            'overall_status': 'FAILED',
            'details': {}
        }
        
        # 1. 基础连通性测试
        success, detail = self.test_basic_connectivity()
        results['basic_connectivity'] = success
        results['details']['basic_connectivity'] = detail
        
        if not success:
            logger.error("基础连通性测试失败，停止后续测试")
            return results
        
        # 2. API端点测试
        success, detail = self.test_api_endpoint()
        results['api_endpoint'] = success
        results['details']['api_endpoint'] = detail
        
        if not success:
            logger.warning("API端点测试失败，但继续进行模型推理测试")
        
        # 3. 模型推理测试
        success, detail = self.test_model_inference()
        results['model_inference'] = success
        results['details']['model_inference'] = detail
        
        # 4. 总体评估
        if results['model_inference']:
            results['overall_status'] = 'SUCCESS'
        elif results['api_endpoint']:
            results['overall_status'] = 'PARTIAL'
        else:
            results['overall_status'] = 'FAILED'
        
        return results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """打印测试结果摘要"""
        logger.info("=" * 50)
        logger.info("=== 测试结果摘要 ===")
        
        status_map = {
            'SUCCESS': '✅ 全部通过',
            'PARTIAL': '⚠️ 部分通过',
            'FAILED': '❌ 测试失败'
        }
        
        logger.info(f"总体状态: {status_map.get(results['overall_status'], '未知')}")
        logger.info(f"配置验证: {'✅' if results['config_valid'] else '❌'}")
        logger.info(f"基础连通性: {'✅' if results['basic_connectivity'] else '❌'}")
        logger.info(f"API端点: {'✅' if results['api_endpoint'] else '❌'}")
        logger.info(f"模型推理: {'✅' if results['model_inference'] else '❌'}")
        
        logger.info("\n详细信息:")
        for test_name, detail in results['details'].items():
            logger.info(f"  {test_name}: {detail}")
        
        logger.info("=" * 50)


def main():
    """主函数"""
    print("模型连通性测试程序")
    print("=" * 50)
    
    # 创建测试器
    tester = ModelConnectivityTester(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL
    )
    
    # 运行测试
    results = tester.run_comprehensive_test()
    
    # 打印摘要
    tester.print_summary(results)
    
    # 设置退出码
    if results['overall_status'] == 'SUCCESS':
        sys.exit(0)
    elif results['overall_status'] == 'PARTIAL':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
