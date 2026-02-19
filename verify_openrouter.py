import os
from openai import OpenAI

def verify_openrouter_gemini(api_key):
    """
    验证 OpenRouter Token 是否可以访问 Google Gemini 模型
    """
    
    # 配置 OpenRouter 客户端
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # 目标模型：OpenRouter 上的模型 ID
    # 注意：目前 OpenRouter 上常见的 Flash 模型 ID 如下：
    # - google/gemini-2.0-flash-001 (最新预览版)
    # - google/gemini-flash-1.5
    # 如果真的有 gemini-2.5-flash，请替换下面的字符串
    model_id = "google/gemini-2.5-flash"

    print(f"正在尝试使用 Token 访问模型: {model_id} ...")

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://your-site.com", # OpenRouter 建议填写的头部
                "X-Title": "Test Script",
            },
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "如果你能看到这条消息，请回复'Connection Successful'。",
                },
            ],
        )
        
        # 获取返回结果
        result = completion.choices[0].message.content
        print("-" * 30)
        print("✅ 验证成功！")
        print(f"模型回复: {result}")
        print("-" * 30)
        return True

    except Exception as e:
        print("-" * 30)
        print("❌ 验证失败！")
        print(f"错误信息: {e}")
        print("-" * 30)
        return False

if __name__ == "__main__":
    # 在这里填入你的 OpenRouter API Key
    # 建议从环境变量获取，或者直接粘贴在下面（注意不要泄露）
    my_api_key = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-你的key在这里")
    
    verify_openrouter_gemini(my_api_key)