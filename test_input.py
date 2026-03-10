#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试ADB输入功能"""

import uiautomator2 as u2
import base64
import time

print("=" * 60)
print("ADB 输入功能测试")
print("=" * 60)

try:
    # 连接设备
    d = u2.connect()
    print("[OK] 已连接到设备")
    
    # 获取当前输入法
    current_ime = d.current_ime()
    print(f"[INFO] 当前输入法: {current_ime}")
    
    # 清空输入框
    d.shell(['input', 'keyevent', '3'])  # HOME键
    time.sleep(0.5)
    
    # 打开Chrome搜索框（或其他应用）
    print("\n[TEST] 1. 打开搜索应用...")
    d.shell(['am', 'start', '-a', 'android.intent.action.WEB_SEARCH'])
    time.sleep(2)
    
    # 点击搜索框
    print("[TEST] 2. 点击搜索框...")
    d.click(500, 300)  # 调整坐标
    time.sleep(0.5)
    
    # 测试1：ADB IME Base64输入
    print("\n[TEST] 3. 测试 ADB IME Base64 输入...")
    test_text = "电动牙刷"
    try:
        # 切换到ADB IME
        d.shell(['ime', 'set', 'com.android.adbkeyboard/.AdbIME'])
        time.sleep(0.5)
        
        # 发送Base64编码的文本
        charsb64 = base64.b64encode(test_text.encode('utf-8')).decode('utf-8')
        print(f"   Text: {test_text}")
        print(f"   Base64: {charsb64}")
        
        d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', 
                 '--es', 'msg', charsb64])
        time.sleep(1.5)
        
        # 恢复原输入法
        d.shell(['ime', 'set', current_ime])
        print("   [OK] ADB IME 输入成功")
        
    except Exception as e:
        print(f"   [ERROR] {e}")
    
    time.sleep(1)
    
    # 截图查看结果
    print("\n[TEST] 4. 获取截图...")
    d.screenshot('test_input.jpg')
    print("   [OK] 截图已保存: test_input.jpg")
    
    print("\n[INFO] 请检查截图中是否成功输入了文本")
    
except Exception as e:
    print(f"[ERROR] {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
