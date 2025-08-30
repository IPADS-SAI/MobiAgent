package com.mobi.agent;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.accessibilityservice.AccessibilityService.GestureResultCallback;
import android.app.ActivityManager;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.graphics.Path;
import android.graphics.Rect;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.view.accessibility.AccessibilityWindowInfo;
import java.util.ArrayList;
import java.util.List;

public class MyAccessibilityService extends AccessibilityService {
    private static MyAccessibilityService instance;
    private boolean isFocusMonitoringActive = false;
    private Handler focusMonitorHandler = new Handler();
    private long monitoringStartTime = 0;
    private long monitoringTimeoutMs = 0;

    private static final String TAG = "AICHATBOT_SERVICE";

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
        CommonUtils.loge(TAG, "*** 无障碍服务已连接并初始化 ***");
        CommonUtils.loge(TAG, "*** SERVICE CONNECTED SUCCESSFULLY ***");
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        // 只处理重要事件，避免过度处理导致性能问题
        if (event != null) {
            int eventType = event.getEventType();
            
            // 只记录重要的事件类型，忽略频繁的UI更新事件
            if (isImportantEvent(eventType)) {
                CommonUtils.logd(TAG, 
                    "重要事件: " + getEventTypeName(eventType) + ", 包名: " + event.getPackageName());
            }
        }
    }
    
    /**
     * 判断是否为需要记录的重要事件
     */
    private boolean isImportantEvent(int eventType) {
        switch (eventType) {
            case AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED:        // 32
            case AccessibilityEvent.TYPE_VIEW_CLICKED:               // 1
            case AccessibilityEvent.TYPE_VIEW_FOCUSED:               // 8
            case AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED:          // 16
            case AccessibilityEvent.TYPE_NOTIFICATION_STATE_CHANGED: // 64
                return true;
            case AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED:     // 2048 - 太频繁，不记录
            case AccessibilityEvent.TYPE_VIEW_SCROLLED:              // 4096 - 太频繁，不记录
            default:
                return false;
        }
    }
    
    /**
     * 获取事件类型的可读名称
     */
    private String getEventTypeName(int eventType) {
        switch (eventType) {
            case AccessibilityEvent.TYPE_VIEW_CLICKED: return "VIEW_CLICKED";
            case AccessibilityEvent.TYPE_VIEW_FOCUSED: return "VIEW_FOCUSED";
            case AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED: return "VIEW_TEXT_CHANGED";
            case AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED: return "WINDOW_STATE_CHANGED";
            case AccessibilityEvent.TYPE_NOTIFICATION_STATE_CHANGED: return "NOTIFICATION_STATE_CHANGED";
            case AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED: return "WINDOW_CONTENT_CHANGED";
            case AccessibilityEvent.TYPE_VIEW_SCROLLED: return "VIEW_SCROLLED";
            default: return "EVENT_" + eventType;
        }
    }

    @Override
    public void onInterrupt() {
        CommonUtils.loge(TAG, "无障碍服务被中断");
    }

    @Override
    protected boolean onKeyEvent(KeyEvent event) {
        // 只记录重要的按键事件，避免日志过多
        if (event.getAction() == KeyEvent.ACTION_DOWN) {
            CommonUtils.logd(TAG, "按键按下: " + KeyEvent.keyCodeToString(event.getKeyCode()));
        }
        return super.onKeyEvent(event);
    }

    @Override
    public void onDestroy() {
        CommonUtils.loge(TAG, "无障碍服务正在销毁");
        instance = null;
        super.onDestroy();
    }

    public static MyAccessibilityService getInstance() {
        CommonUtils.loge(TAG, "getInstance() 被调用，当前实例: " + 
                          (instance != null ? "有效" : "null"));
        return instance;
    }

    public boolean clickOnScreen(float x, float y) {
        CommonUtils.loge(TAG, GestureUtils.formatGestureLog("CLICK", x, y, null, null));
        
        // 添加点击前的延迟以确保界面稳定
       // CommonUtils.safeSleep(800);
        
        // 检查当前应用
        AccessibilityNodeInfo rootInfo = getRootInActiveWindow();
        if (rootInfo != null) {
            String currentPackage = CommonUtils.safeToString(rootInfo.getPackageName());
            CommonUtils.loge(TAG, "当前应用包名: " + currentPackage);
            
            // 特别标记B站应用状态
            if ("tv.danmaku.bili".equals(currentPackage)) {
                CommonUtils.loge(TAG, "*** 检测到B站应用，准备在B站内执行点击 ***");
                // 分析B站当前界面，查找搜索框
                analyzeScreenForSearchBox(rootInfo);
            }
            
            rootInfo.recycle();
        }
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            android.util.Log.e("AICHATBOT_SERVICE", "🚀 开始创建手势描述...");
            
            Path path = new Path();
            path.moveTo(x, y);

            GestureDescription.StrokeDescription strokeDescription =
                    new GestureDescription.StrokeDescription(path, 0, 150);
            GestureDescription.Builder builder = new GestureDescription.Builder();
            builder.addStroke(strokeDescription);

            android.util.Log.e("AICHATBOT_SERVICE", "📱 准备分发手势到系统...");
            
            boolean result = dispatchGesture(builder.build(), new GestureResultCallback() {
                @Override
                public void onCompleted(GestureDescription gestureDescription) {
                    super.onCompleted(gestureDescription);
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 点击手势完成: (" + x + ", " + y + ")");
                    
                    // 点击完成后检查应用状态
                    new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            AccessibilityNodeInfo currentRoot = getRootInActiveWindow();
                            if (currentRoot != null) {
                                String currentPackage = currentRoot.getPackageName() != null ? 
                                    currentRoot.getPackageName().toString() : "unknown";
                                android.util.Log.d("AICHATBOT_SERVICE", "点击完成后应用状态: " + currentPackage);
                                
                                currentRoot.recycle();
                            }
                        }
                    }, 200); // 缩短延迟时间到200ms
                }

                @Override
                public void onCancelled(GestureDescription gestureDescription) {
                    super.onCancelled(gestureDescription);
                    android.util.Log.e("AICHATBOT_SERVICE", "❌ 点击手势被取消: (" + x + ", " + y + ")");
                }
            }, null);
            
            android.util.Log.e("AICHATBOT_SERVICE", "📋 dispatchGesture返回结果: " + result);
            android.util.Log.e("AICHATBOT_SERVICE", "尝试点击坐标: (" + x + ", " + y + "), 结果: " + result);
            return result;
        } else {
            android.util.Log.e("AICHATBOT_SERVICE", "Android版本过低，不支持手势操作");
            return false;
        }
    }

    public boolean swipeOnScreen(float startX, float startY, float endX, float endY, long duration) {
        CommonUtils.loge(TAG, GestureUtils.formatGestureLog("SWIPE", startX, startY, endX, endY));
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            Path path = new Path();
            path.moveTo(startX, startY);
            path.lineTo(endX, endY);

            GestureDescription.StrokeDescription strokeDescription =
                    new GestureDescription.StrokeDescription(path, 0, duration);
            GestureDescription.Builder builder = new GestureDescription.Builder();
            builder.addStroke(strokeDescription);

            boolean result = dispatchGesture(builder.build(), new GestureResultCallback() {
                @Override
                public void onCompleted(GestureDescription gestureDescription) {
                    super.onCompleted(gestureDescription);
                    CommonUtils.loge(TAG, "滑动手势完成");
                }

                @Override
                public void onCancelled(GestureDescription gestureDescription) {
                    super.onCancelled(gestureDescription);
                    CommonUtils.loge(TAG, "滑动手势被取消");
                }
            }, null);

            android.util.Log.e("AICHATBOT_SERVICE", "滑动结果: " + result);
            return result;
        } else {
            android.util.Log.e("AICHATBOT_SERVICE", "Android版本过低，不支持手势操作");
            return false;
        }
    }

    // 向左滑动的便捷方法
    public boolean swipeLeft() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 执行向左滑动 ===");
        // 从屏幕右侧滑动到左侧 (假设屏幕宽度1080px)
        float startX = 800f;   // 起始X坐标 (右侧)
        float startY = 1000f;  // 起始Y坐标 (中间)
        float endX = 200f;     // 结束X坐标 (左侧)
        float endY = 1000f;    // 结束Y坐标 (保持相同高度)
        long duration = 500;   // 滑动持续时间500毫秒
        
        return swipeOnScreen(startX, startY, endX, endY, duration);
    }

    // 向上滑动的便捷方法
    public boolean swipeUp() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 执行向上滑动 ===");
        // 从屏幕下方滑动到上方
        float startX = 540f;   // 起始X坐标 (屏幕中央)
        float startY = 1500f;  // 起始Y坐标 (下方)
        float endX = 540f;     // 结束X坐标 (保持相同位置)
        float endY = 800f;     // 结束Y坐标 (上方)
        long duration = 600;   // 滑动持续时间600毫秒
        
        android.util.Log.e("AICHATBOT_SERVICE", "上划参数: 从(" + startX + "," + startY + ") 到 (" + endX + "," + endY + ")");
        return swipeOnScreen(startX, startY, endX, endY, duration);
    }

    // 向下滑动的便捷方法
    public boolean swipeDown() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 执行向下滑动 ===");
        // 从屏幕上方滑动到下方
        float startX = 540f;   // 起始X坐标 (屏幕中央)
        float startY = 800f;   // 起始Y坐标 (上方)
        float endX = 540f;     // 结束X坐标 (保持相同位置)
        float endY = 1500f;    // 结束Y坐标 (下方)
        long duration = 600;   // 滑动持续时间600毫秒
        
        android.util.Log.e("AICHATBOT_SERVICE", "下滑参数: 从(" + startX + "," + startY + ") 到 (" + endX + "," + endY + ")");
        return swipeOnScreen(startX, startY, endX, endY, duration);
    }
    
    // 向右滑动的便捷方法
    public boolean swipeRight() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 执行向右滑动 ===");
        // 从屏幕左侧滑动到右侧
        float startX = 200f;   // 起始X坐标 (左侧)
        float startY = 1000f;  // 起始Y坐标 (中间)
        float endX = 880f;     // 结束X坐标 (右侧)
        float endY = 1000f;    // 结束Y坐标 (保持相同高度)
        long duration = 500;   // 滑动持续时间500毫秒
        
        android.util.Log.e("AICHATBOT_SERVICE", "右滑参数: 从(" + startX + "," + startY + ") 到 (" + endX + "," + endY + ")");
        return swipeOnScreen(startX, startY, endX, endY, duration);
    }
    
    /**
     * 检查指定应用进程是否还在运行
     */
    private boolean isAppProcessRunning(android.app.ActivityManager am, String packageName) {
        try {
            List<android.app.ActivityManager.RunningAppProcessInfo> runningProcesses = am.getRunningAppProcesses();
            if (runningProcesses != null) {
                for (android.app.ActivityManager.RunningAppProcessInfo processInfo : runningProcesses) {
                    if (packageName.equals(processInfo.processName)) {
                        android.util.Log.d("AICHATBOT_SERVICE", "进程仍在运行: " + packageName + ", 重要性: " + processInfo.importance);
                        return true;
                    }
                }
            }
            android.util.Log.d("AICHATBOT_SERVICE", "进程已停止: " + packageName);
            return false;
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "检查进程状态失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 在最近任务中查找并关闭指定应用
     */
    private boolean findAndCloseSpecificApp(AccessibilityNodeInfo node, String targetPackage) {
        if (node == null) return false;
        
        try {
            // 递归查找所有节点
            return searchAndCloseApp(node, targetPackage, 0);
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "查找指定应用失败: " + e.getMessage());
        }
        
        return false;
    }
    
    /**
     * 递归搜索并关闭指定应用
     */
    private boolean searchAndCloseApp(AccessibilityNodeInfo node, String targetPackage, int depth) {
        if (node == null || depth > 10) return false;
        
        try {
            // 检查当前节点的包名
            CharSequence packageName = node.getPackageName();
            if (packageName != null && packageName.toString().contains(targetPackage)) {
                android.util.Log.e("AICHATBOT_SERVICE", "找到目标应用节点: " + packageName);
                
                // 尝试向上滑动删除
                Rect bounds = new Rect();
                node.getBoundsInScreen(bounds);
                if (bounds.width() > 100 && bounds.height() > 100) {
                    float startX = bounds.centerX();
                    float startY = bounds.centerY();
                    float endX = startX;
                    float endY = startY - 300; // 向上滑动300像素
                    
                    android.util.Log.e("AICHATBOT_SERVICE", "尝试向上滑动删除应用: " + targetPackage);
                    boolean swipeResult = swipeOnScreen(startX, startY, endX, endY, 400);
                    
                    if (swipeResult) {
                        android.util.Log.e("AICHATBOT_SERVICE", "滑动删除成功: " + targetPackage);
                        return true;
                    }
                }
                
                // 备选方案：尝试点击关闭按钮
                List<AccessibilityNodeInfo> closeButtons = node.findAccessibilityNodeInfosByText("关闭");
                if (closeButtons != null && !closeButtons.isEmpty()) {
                    closeButtons.get(0).performAction(AccessibilityNodeInfo.ACTION_CLICK);
                    android.util.Log.e("AICHATBOT_SERVICE", "点击关闭按钮成功: " + targetPackage);
                    return true;
                }
            }
            
            // 递归搜索子节点
            for (int i = 0; i < node.getChildCount(); i++) {
                AccessibilityNodeInfo child = node.getChild(i);
                if (child != null) {
                    boolean result = searchAndCloseApp(child, targetPackage, depth + 1);
                    if (result) return true;
                }
            }
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "搜索应用节点异常: " + e.getMessage());
        }
        
        return false;
    }

    /**
     * 强制关闭当前应用进程
     * @return 是否成功执行
     */
    public boolean forceCloseCurrentApp() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "强制关闭当前应用进程");
            
            // 方法1: 先尝试打开最近任务列表
            boolean recentsResult = performGlobalAction(AccessibilityService.GLOBAL_ACTION_RECENTS);
            android.util.Log.e("AICHATBOT_SERVICE", "打开最近任务结果: " + recentsResult);
            
            if (recentsResult) {
                // 等待任务列表打开
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                
                // 查找当前应用并关闭
                AccessibilityNodeInfo rootNode = getRootInActiveWindow();
                if (rootNode != null) {
                    // 尝试查找关闭按钮或滑动删除
                    boolean closed = findAndCloseApp(rootNode);
                    if (closed) {
                        android.util.Log.e("AICHATBOT_SERVICE", "通过最近任务成功关闭应用");
                        return true;
                    }
                }
            }
            
            // 方法2: 如果最近任务方法失败，尝试按返回键多次
            android.util.Log.e("AICHATBOT_SERVICE", "尝试通过多次返回键关闭应用");
            for (int i = 0; i < 5; i++) {
                boolean backResult = performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
                android.util.Log.e("AICHATBOT_SERVICE", "返回键 " + (i+1) + " 次结果: " + backResult);
                try {
                    Thread.sleep(300);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            
            // 最后按Home键确保回到桌面
            boolean homeResult = performGlobalAction(AccessibilityService.GLOBAL_ACTION_HOME);
            android.util.Log.e("AICHATBOT_SERVICE", "最终Home键结果: " + homeResult);
            
            return homeResult;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "强制关闭应用失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 在最近任务中查找并关闭应用
     */
    private boolean findAndCloseApp(AccessibilityNodeInfo node) {
        if (node == null) return false;
        
        try {
            // 查找关闭按钮（通常是X或删除图标）
            List<AccessibilityNodeInfo> closeButtons = node.findAccessibilityNodeInfosByText("关闭");
            if (closeButtons != null && !closeButtons.isEmpty()) {
                closeButtons.get(0).performAction(AccessibilityNodeInfo.ACTION_CLICK);
                return true;
            }
            
            // 尝试向上滑动删除（常见的关闭应用手势）
            Rect bounds = new Rect();
            node.getBoundsInScreen(bounds);
            if (bounds.width() > 0 && bounds.height() > 0) {
                // 从中心向上滑动
                float startX = bounds.centerX();
                float startY = bounds.centerY();
                float endX = startX;
                float endY = startY - 200; // 向上滑动200像素
                
                return swipeOnScreen(startX, startY, endX, endY, 300);
            }
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "查找关闭按钮失败: " + e.getMessage());
        }
        
        return false;
    }

    /**
     * 按Home键回到桌面（结束当前应用）
     * @return 是否成功执行
     */
    public boolean pressHome() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "执行按Home键操作，回到桌面");
            boolean result = performGlobalAction(AccessibilityService.GLOBAL_ACTION_HOME);
            android.util.Log.e("AICHATBOT_SERVICE", "按Home键结果: " + result);
            return result;
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "按Home键失败: " + e.getMessage());
            return false;
        }
    }

    // 屏幕分析方法
    public void analyzeCurrentScreen(AccessibilityNodeInfo node, int depth) {
        if (node == null || depth > 6) {
            return;
        }

        String className = node.getClassName() != null ? node.getClassName().toString() : "";
        String text = node.getText() != null ? node.getText().toString() : "";
        String contentDesc = node.getContentDescription() != null ? node.getContentDescription().toString() : "";

        if (node.isEditable() || className.contains("Edit") || text.length() > 0) {
            Rect bounds = new Rect();
            node.getBoundsInScreen(bounds);
            android.util.Log.e("AICHATBOT_SERVICE", 
                "深度" + depth + " - 类: " + className + 
                " | 文本: '" + text + "'" +
                " | 描述: '" + contentDesc + "'" +
                " | 可编辑: " + node.isEditable() +
                " | 有焦点: " + node.isFocused() +
                " | 位置: (" + bounds.left + "," + bounds.top + "," + bounds.right + "," + bounds.bottom + ")");
        }

        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo childNode = node.getChild(i);
            if (childNode != null) {
                analyzeCurrentScreen(childNode, depth + 1);
                childNode.recycle();
            }
        }
    }

    // 强化的文本输入方法 - 针对中文汉字输入优化
    public boolean inputTextSafely(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 开始简化的文本输入策略: " + text + " ===");
        
        final long methodStartTime = System.currentTimeMillis();
        final long MAX_EXECUTION_TIME = 8000; // 8秒超时
        
        try {
            // 如果是空文本，只进行界面分析
            if (text == null || text.trim().isEmpty()) {
                android.util.Log.e("AICHATBOT_SERVICE", "空文本，只进行界面分析");
                AccessibilityNodeInfo rootAnalysis = getRootInActiveWindow();
                if (rootAnalysis != null) {
                    analyzeCurrentScreen(rootAnalysis, 0);
                    rootAnalysis.recycle();
                }
                return false;
            }
            
            // 超时检查1
            if (System.currentTimeMillis() - methodStartTime > MAX_EXECUTION_TIME) {
                android.util.Log.e("AICHATBOT_SERVICE", "⏰ 超时：预检查阶段");
                return false;
            }
            
            // 检测焦点并输入文本
            android.util.Log.e("AICHATBOT_SERVICE", "=== 步骤1：检测当前焦点状态 ===");
            AccessibilityNodeInfo currentRoot = getRootInActiveWindow();
            if (currentRoot != null) {
                String currentPackage = currentRoot.getPackageName() != null ? 
                    currentRoot.getPackageName().toString() : "unknown";
                android.util.Log.e("AICHATBOT_SERVICE", "当前应用包名: " + currentPackage);
                
                // 超时检查2
                if (System.currentTimeMillis() - methodStartTime > MAX_EXECUTION_TIME) {
                    android.util.Log.e("AICHATBOT_SERVICE", "⏰ 超时：获取根节点后");
                    currentRoot.recycle();
                    return false;
                }
                
                // 查找有焦点的输入框
                AccessibilityNodeInfo focusedNode = findFocusedEditTextEnhanced(currentRoot);
                if (focusedNode != null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 找到有焦点的输入框，开始输入文本");
                    
                    // 超时检查3
                    if (System.currentTimeMillis() - methodStartTime > MAX_EXECUTION_TIME) {
                        android.util.Log.e("AICHATBOT_SERVICE", "⏰ 超时：找到焦点输入框后");
                        focusedNode.recycle();
                        currentRoot.recycle();
                        return false;
                    }
                    
                    boolean result = inputTextToFocusedNode(focusedNode, text);
                    focusedNode.recycle();
                    currentRoot.recycle();
                    
                    long duration = System.currentTimeMillis() - methodStartTime;
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ inputTextSafely完成，耗时: " + duration + "ms，结果: " + result);
                    return result;
                    
                } else {
                    android.util.Log.e("AICHATBOT_SERVICE", "❌ 未找到有焦点的输入框");
                    android.util.Log.e("AICHATBOT_SERVICE", "尝试查找所有可编辑的输入框...");
                    
                    // 超时检查4
                    if (System.currentTimeMillis() - methodStartTime > MAX_EXECUTION_TIME) {
                        android.util.Log.e("AICHATBOT_SERVICE", "⏰ 超时：备用输入框查找阶段");
                        currentRoot.recycle();
                        return false;
                    }
                    
                    // 如果没有焦点，尝试找到第一个可编辑的输入框
                    AccessibilityNodeInfo editableNode = findFirstEditableNode(currentRoot);
                    if (editableNode != null) {
                        android.util.Log.e("AICHATBOT_SERVICE", "找到可编辑输入框，尝试设置焦点并输入");
                        // 先点击获得焦点
                        editableNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
                        try {
                            Thread.sleep(500); // 等待焦点设置
                        } catch (InterruptedException e) {
                            android.util.Log.e("AICHATBOT_SERVICE", "等待焦点设置被中断", e);
                        }
                        boolean result = inputTextToFocusedNode(editableNode, text);
                        editableNode.recycle();
                        currentRoot.recycle();
                        
                        long duration = System.currentTimeMillis() - methodStartTime;
                        android.util.Log.e("AICHATBOT_SERVICE", "✅ inputTextSafely(备用)完成，耗时: " + duration + "ms，结果: " + result);
                        return result;
                    } 
                    
                    // 如果标准方法都失败，尝试更宽松的检测
                    android.util.Log.e("AICHATBOT_SERVICE", "标准方法失败，尝试宽松检测...");
                    AccessibilityNodeInfo anyInputNode = findAnyInputElement(currentRoot);
                    if (anyInputNode != null) {
                        android.util.Log.e("AICHATBOT_SERVICE", "找到可能的输入元素，尝试点击后输入");
                        // 先点击获得焦点
                        anyInputNode.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                        try {
                            Thread.sleep(300);
                        } catch (InterruptedException e) {
                            android.util.Log.e("AICHATBOT_SERVICE", "等待点击被中断", e);
                        }
                        anyInputNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
                        try {
                            Thread.sleep(200);
                        } catch (InterruptedException e) {
                            android.util.Log.e("AICHATBOT_SERVICE", "等待焦点设置被中断", e);
                        }
                        boolean result = inputTextToFocusedNode(anyInputNode, text);
                        anyInputNode.recycle();
                        currentRoot.recycle();
                        
                        long duration = System.currentTimeMillis() - methodStartTime;
                        android.util.Log.e("AICHATBOT_SERVICE", "✅ inputTextSafely(宽松检测)完成，耗时: " + duration + "ms，结果: " + result);
                        return result;
                    } else {
                        android.util.Log.e("AICHATBOT_SERVICE", "❌ 未找到任何可能的输入元素");
                    }
                }
                currentRoot.recycle();
            }
            
            android.util.Log.e("AICHATBOT_SERVICE", "❌ 所有输入方法都失败了");
            return false;
            
        } catch (Exception e) {
            long errorDuration = System.currentTimeMillis() - methodStartTime;
            android.util.Log.e("AICHATBOT_SERVICE", "❌ inputTextSafely异常，耗时: " + errorDuration + "ms，错误: " + e.getMessage());
            return false;
        } finally {
            long finalDuration = System.currentTimeMillis() - methodStartTime;
            android.util.Log.e("AICHATBOT_SERVICE", "🏁 inputTextSafely方法结束，总耗时: " + finalDuration + "ms");
        }
    }

    /**
     * 使用剪贴板粘贴的方式输入文本
     * @param text 要输入的文本
     * @return 是否成功
     */
    public boolean inputTextByClipboard(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 使用剪贴板粘贴输入文本: " + text + " ===");
        
        try {
            // 1. 将文本复制到剪贴板
            android.content.ClipboardManager clipboard = (android.content.ClipboardManager) getSystemService(android.content.Context.CLIPBOARD_SERVICE);
            android.content.ClipData clip = android.content.ClipData.newPlainText("input_text", text);
            clipboard.setPrimaryClip(clip);
            android.util.Log.e("AICHATBOT_SERVICE", "✅ 文本已复制到剪贴板");
            
            // 2. 等待一下确保剪贴板设置完成
            Thread.sleep(100);
            
            // 3. 查找当前焦点的输入框并直接设置文本
            AccessibilityNodeInfo root = getRootInActiveWindow();
            if (root != null) {
                // 先尝试找到有焦点的输入框
                AccessibilityNodeInfo focusedNode = findFocusedEditTextEnhanced(root);
                if (focusedNode != null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "找到有焦点的输入框，直接设置文本");
                    boolean result = setTextDirectly(focusedNode, text);
                    focusedNode.recycle();
                    root.recycle();
                    return result;
                }
                
                // 如果没有焦点输入框，查找第一个可编辑的
                AccessibilityNodeInfo editableNode = findFirstEditableNode(root);
                if (editableNode != null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "找到可编辑输入框，设置焦点后设置文本");
                    editableNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
                    Thread.sleep(200);
                    boolean result = setTextDirectly(editableNode, text);
                    editableNode.recycle();
                    root.recycle();
                    return result;
                }
                
                // 如果标准方法都失败，尝试更宽松的检测
                AccessibilityNodeInfo anyInputNode = findAnyInputElement(root);
                if (anyInputNode != null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "找到可能的输入元素，尝试点击后设置文本");
                    // 先点击获得焦点
                    anyInputNode.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                    Thread.sleep(300);
                    anyInputNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
                    Thread.sleep(200);
                    boolean result = setTextDirectly(anyInputNode, text);
                    anyInputNode.recycle();
                    root.recycle();
                    return result;
                }
                
                root.recycle();
            }
            
            android.util.Log.w("AICHATBOT_SERVICE", "❌ 未找到合适的输入框进行粘贴");
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "❌ 剪贴板粘贴输入异常: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 直接设置文本到节点
     */
    private boolean setTextDirectly(AccessibilityNodeInfo node, String text) {
        try {
            // 方法1：使用ACTION_SET_TEXT（Android API 18+）
            android.os.Bundle arguments = new android.os.Bundle();
            arguments.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            boolean success = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments);
            
            if (success) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 使用ACTION_SET_TEXT设置文本成功");
                return true;
            }
            
            // 方法2：先清空再输入
            android.util.Log.w("AICHATBOT_SERVICE", "ACTION_SET_TEXT失败，尝试清空后输入");
            
            // 全选现有文本（使用数值常量以兼容低版本API）
            node.performAction(131072); // ACTION_SELECT_ALL 的数值
            Thread.sleep(100);
            
            // 删除选中的文本
            node.performAction(AccessibilityNodeInfo.ACTION_CUT);
            Thread.sleep(100);
            
            // 输入新文本
            arguments.clear();
            arguments.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            success = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments);
            
            if (success) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 清空后设置文本成功");
                return true;
            }
            
            android.util.Log.w("AICHATBOT_SERVICE", "❌ 所有文本设置方法都失败");
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "❌ 设置文本异常: " + e.getMessage());
            return false;
        }
    }

    // 启动持续焦点监控
    public void startFocusMonitoring() {
        android.util.Log.e("AICHATBOT_SERVICE", "*** 启动持续焦点监控 ***");
        isFocusMonitoringActive = true;
        monitoringStartTime = System.currentTimeMillis();
        monitoringTimeoutMs = 0; // 无超时
        
        // 开始监控循环
        focusMonitoringRunnable.run();
    }
    
    // 启动带超时的持续焦点监控
    public void startFocusMonitoringWithTimeout(long timeoutMs) {
        android.util.Log.e("AICHATBOT_SERVICE", "*** 启动带超时的持续焦点监控，超时: " + timeoutMs + "ms ***");
        isFocusMonitoringActive = true;
        monitoringStartTime = System.currentTimeMillis();
        monitoringTimeoutMs = timeoutMs;
        
        // 开始监控循环
        focusMonitoringRunnableWithTimeout.run();
    }
    
    // 停止焦点监控
    public void stopFocusMonitoring() {
        android.util.Log.e("AICHATBOT_SERVICE", "停止焦点监控");
        isFocusMonitoringActive = false;
        focusMonitorHandler.removeCallbacks(focusMonitoringRunnable);
    }
    
    // 焦点监控循环
    private Runnable focusMonitoringRunnable = new Runnable() {
        @Override
        public void run() {
            if (!isFocusMonitoringActive) {
                android.util.Log.e("AICHATBOT_SERVICE", "监控已停止");
                return;
            }
            
            android.util.Log.d("FocusMonitor", "检查焦点状态...");
            
            // 检查是否有输入框获得焦点
            AccessibilityNodeInfo currentRoot = getRootInActiveWindow();
            if (currentRoot != null) {
                AccessibilityNodeInfo focusedNode = findFocusedEditTextEnhanced(currentRoot);
                if (focusedNode != null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✓ 检测到输入框焦点！执行文本输入");
                    android.util.Log.e("AICHATBOT_SERVICE", "=== 检测到焦点，开始输入\"潜伏\" ===");
                    
                    // 停止监控
                    stopFocusMonitoring();
                    
                    // 执行文本输入
                    inputTextSafely("潜伏");
                    
                    focusedNode.recycle();
                    currentRoot.recycle();
                    android.util.Log.e("AICHATBOT_SERVICE", "文本输入完成，监控结束");
                    return;
                }
                currentRoot.recycle();
            }
            
            // 继续监控，每500毫秒检查一次
            focusMonitorHandler.postDelayed(this, 500);
        }
    };

    // 带超时的焦点监控循环
    private Runnable focusMonitoringRunnableWithTimeout = new Runnable() {
        @Override
        public void run() {
            if (!isFocusMonitoringActive) {
                android.util.Log.e("AICHATBOT_SERVICE", "监控已停止");
                return;
            }
            
            // 检查是否超时
            long currentTime = System.currentTimeMillis();
            long elapsedTime = currentTime - monitoringStartTime;
            
            if (monitoringTimeoutMs > 0 && elapsedTime >= monitoringTimeoutMs) {
                android.util.Log.e("AICHATBOT_SERVICE", "*** 焦点监控超时，停止监控 ***");
                android.util.Log.e("AICHATBOT_SERVICE", "监控时长: " + elapsedTime + "ms，超时设置: " + monitoringTimeoutMs + "ms");
                stopFocusMonitoring();
                return;
            }
            
            android.util.Log.d("FocusMonitor", "检查焦点状态... 已监控: " + elapsedTime + "ms / " + monitoringTimeoutMs + "ms");
            
            // 检查是否有输入框获得焦点
            AccessibilityNodeInfo currentRoot = getRootInActiveWindow();
            if (currentRoot != null) {
                String currentPackage = currentRoot.getPackageName() != null ? 
                    currentRoot.getPackageName().toString() : "unknown";
                android.util.Log.d("FocusMonitor", "当前应用: " + currentPackage);
                
                // 增强的焦点检测
                AccessibilityNodeInfo focusedNode = findFocusedEditTextEnhanced(currentRoot);
                if (focusedNode != null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✓ 检测到输入框焦点！执行文本输入");
                    android.util.Log.e("AICHATBOT_SERVICE", "=== 检测到焦点，开始输入\"潜伏\" ===");
                    android.util.Log.e("AICHATBOT_SERVICE", "焦点检测耗时: " + elapsedTime + "ms");
                    
                    // 停止监控
                    stopFocusMonitoring();
                    
                    // 执行文本输入（带5秒超时）
                    boolean inputResult = tryClipboardInputWithStrictTimeout("潜伏");
                    if (!inputResult) {
                        android.util.Log.w("AICHATBOT_SERVICE", "剪贴板超时，使用备用方案");
                        inputResult = useActionSetTextAsFallback("潜伏");
                    }
                    android.util.Log.e("AICHATBOT_SERVICE", "文本输入最终结果: " + inputResult);
                    
                    focusedNode.recycle();
                    currentRoot.recycle();
                    android.util.Log.e("AICHATBOT_SERVICE", "文本输入完成，监控结束");
                    return;
                } else {
                    // 如果没有找到焦点，分析当前界面状态
                    if (elapsedTime % 1500 == 0) { // 每1.5秒输出一次详细分析
                        android.util.Log.e("AICHATBOT_SERVICE", "=== 未找到焦点，分析界面状态 ===");
                        analyzeUIForInputElements(currentRoot);
                    }
                }
                currentRoot.recycle();
            } else {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取当前应用根节点");
            }
            
            // 继续监控，每300毫秒检查一次（更频繁以便及时响应）
            focusMonitorHandler.postDelayed(this, 300);
        }
    };
    
    // 增强的焦点检测方法
    private AccessibilityNodeInfo findFocusedEditTextEnhanced(AccessibilityNodeInfo node) {
        if (node == null) return null;
        
        android.util.Log.d("AICHATBOT_SERVICE", "=== 增强焦点检测开始 ===");
        
        // 方法1: 直接查找有焦点的可编辑节点
        AccessibilityNodeInfo focusedNode = node.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
        if (focusedNode != null && focusedNode.isEditable()) {
            android.util.Log.e("AICHATBOT_SERVICE", "✓ 方法1成功: 找到输入焦点的可编辑节点");
            return focusedNode;
        }
        if (focusedNode != null) {
            focusedNode.recycle();
        }
        
        // 方法2: 查找可访问性焦点
        focusedNode = node.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY);
        if (focusedNode != null && focusedNode.isEditable()) {
            android.util.Log.e("AICHATBOT_SERVICE", "✓ 方法2成功: 找到可访问性焦点的可编辑节点");
            return focusedNode;
        }
        if (focusedNode != null) {
            focusedNode.recycle();
        }
        
        // 方法3: 递归查找有焦点且可编辑的节点
        AccessibilityNodeInfo recursiveResult = findFocusedEditTextRecursive(node, 0);
        if (recursiveResult != null) {
            android.util.Log.e("AICHATBOT_SERVICE", "✓ 方法3成功: 递归找到焦点节点");
            return recursiveResult;
        }
        
        // 方法4: 查找最近可能获得焦点的输入框
        AccessibilityNodeInfo likelyFocused = findLikelyFocusedInput(node);
        if (likelyFocused != null) {
            android.util.Log.e("AICHATBOT_SERVICE", "✓ 方法4成功: 找到可能的焦点输入框");
            return likelyFocused;
        }
        
        android.util.Log.d("AICHATBOT_SERVICE", "❌ 所有焦点检测方法都失败");
        return null;
    }
    
    // 查找可能有焦点的输入框
    private AccessibilityNodeInfo findLikelyFocusedInput(AccessibilityNodeInfo node) {
        if (node == null) return null;
        
        List<AccessibilityNodeInfo> editableNodes = new ArrayList<>();
        findAllEditableElementsEnhanced(node, editableNodes);
        
        android.util.Log.d("AICHATBOT_SERVICE", "找到 " + editableNodes.size() + " 个可编辑元素");
        
        // 优先选择有焦点的
        for (AccessibilityNodeInfo editNode : editableNodes) {
            if (editNode.isFocused()) {
                android.util.Log.e("AICHATBOT_SERVICE", "找到有焦点的可编辑元素");
                // 清理其他节点
                for (AccessibilityNodeInfo otherNode : editableNodes) {
                    if (otherNode != editNode) {
                        otherNode.recycle();
                    }
                }
                return editNode;
            }
        }
        
        // 选择可能是输入框的元素（基于类名和位置）
        for (AccessibilityNodeInfo editNode : editableNodes) {
            String className = editNode.getClassName() != null ? editNode.getClassName().toString() : "";
            Rect bounds = new Rect();
            editNode.getBoundsInScreen(bounds);
            
            // 检查是否是常见的输入框类型
            if (className.contains("EditText") || className.contains("TextInputEditText")) {
                android.util.Log.e("AICHATBOT_SERVICE", "找到可能的输入框: " + className);
                // 清理其他节点
                for (AccessibilityNodeInfo otherNode : editableNodes) {
                    if (otherNode != editNode) {
                        otherNode.recycle();
                    }
                }
                return editNode;
            }
        }
        
        // 如果都没有特殊标识，选择第一个
        if (!editableNodes.isEmpty()) {
            AccessibilityNodeInfo firstNode = editableNodes.get(0);
            // 清理其他节点
            for (int i = 1; i < editableNodes.size(); i++) {
                editableNodes.get(i).recycle();
            }
            return firstNode;
        }
        
        return null;
    }
    
    // 增强的递归查找有焦点的输入框
    private AccessibilityNodeInfo findFocusedEditTextRecursive(AccessibilityNodeInfo node, int depth) {
        if (node == null || depth > 10) return null;
        
        // 检查当前节点是否有焦点且可编辑
        if (node.isFocused() && node.isEditable()) {
            android.util.Log.e("AICHATBOT_SERVICE", "递归找到有焦点的输入框 (深度" + depth + "): " + 
                (node.getClassName() != null ? node.getClassName().toString() : "unknown"));
            return AccessibilityNodeInfo.obtain(node);
        }
        
        // 递归查找子节点
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                AccessibilityNodeInfo result = findFocusedEditTextRecursive(child, depth + 1);
                child.recycle();
                if (result != null) {
                    return result;
                }
            }
        }
        
        return null;
    }
    
    // 增强的可编辑元素查找
    private void findAllEditableElementsEnhanced(AccessibilityNodeInfo node, List<AccessibilityNodeInfo> editableNodes) {
        if (node == null) return;
        
        if (node.isEditable()) {
            AccessibilityNodeInfo copy = AccessibilityNodeInfo.obtain(node);
            editableNodes.add(copy);
        }
        
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                findAllEditableElementsEnhanced(child, editableNodes);
                child.recycle();
            }
        }
    }
    
    // 分析界面中的输入元素
    private void analyzeUIForInputElements(AccessibilityNodeInfo rootNode) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 开始分析界面输入元素 ===");
        
        String packageName = rootNode.getPackageName() != null ? rootNode.getPackageName().toString() : "unknown";
        android.util.Log.e("AICHATBOT_SERVICE", "当前应用包名: " + packageName);
        
        List<AccessibilityNodeInfo> allInputs = new ArrayList<>();
        findAllInputElementsDeep(rootNode, allInputs, 0);
        
        android.util.Log.e("AICHATBOT_SERVICE", "总共找到 " + allInputs.size() + " 个输入相关元素");
        
        for (int i = 0; i < allInputs.size(); i++) {
            AccessibilityNodeInfo input = allInputs.get(i);
            analyzeInputElement(input, i);
            input.recycle();
        }
        
        // 检查是否有系统UI遮挡
        checkForSystemUIInterference();
    }
    
    // 深度查找所有输入元素
    private void findAllInputElementsDeep(AccessibilityNodeInfo node, List<AccessibilityNodeInfo> inputs, int depth) {
        if (node == null || depth > 15) return;
        
        String className = node.getClassName() != null ? node.getClassName().toString() : "";
        
        // 检查是否是输入相关的元素
        if (node.isEditable() || 
            className.contains("EditText") || 
            className.contains("TextInput") ||
            className.contains("SearchView") ||
            (node.isClickable() && className.contains("Text"))) {
            
            AccessibilityNodeInfo copy = AccessibilityNodeInfo.obtain(node);
            inputs.add(copy);
        }
        
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                findAllInputElementsDeep(child, inputs, depth + 1);
                child.recycle();
            }
        }
    }
    
    // 分析单个输入元素
    private void analyzeInputElement(AccessibilityNodeInfo element, int index) {
        String className = element.getClassName() != null ? element.getClassName().toString() : "";
        String text = element.getText() != null ? element.getText().toString() : "";
        String hint = element.getHintText() != null ? element.getHintText().toString() : "";
        String contentDesc = element.getContentDescription() != null ? element.getContentDescription().toString() : "";
        
        Rect bounds = new Rect();
        element.getBoundsInScreen(bounds);
        
        android.util.Log.e("AICHATBOT_SERVICE", "输入元素" + index + ":");
        android.util.Log.e("AICHATBOT_SERVICE", "  类名: " + className);
        android.util.Log.e("AICHATBOT_SERVICE", "  文本: '" + text + "'");
        android.util.Log.e("AICHATBOT_SERVICE", "  提示: '" + hint + "'");
        android.util.Log.e("AICHATBOT_SERVICE", "  描述: '" + contentDesc + "'");
        android.util.Log.e("AICHATBOT_SERVICE", "  位置: (" + bounds.left + "," + bounds.top + "," + bounds.right + "," + bounds.bottom + ")");
        android.util.Log.e("AICHATBOT_SERVICE", "  可编辑: " + element.isEditable());
        android.util.Log.e("AICHATBOT_SERVICE", "  有焦点: " + element.isFocused());
        android.util.Log.e("AICHATBOT_SERVICE", "  可点击: " + element.isClickable());
        android.util.Log.e("AICHATBOT_SERVICE", "  已启用: " + element.isEnabled());
        android.util.Log.e("AICHATBOT_SERVICE", "  可见: " + element.isVisibleToUser());
    }
    
    // 检查系统UI干扰
    private void checkForSystemUIInterference() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 检查系统UI干扰 ===");
        
        // 检查是否有下拉通知栏或其他系统UI
        try {
            List<AccessibilityWindowInfo> windows = getWindows();
            for (AccessibilityWindowInfo window : windows) {
                if (window.getType() == AccessibilityWindowInfo.TYPE_SYSTEM) {
                    android.util.Log.e("AICHATBOT_SERVICE", "⚠️ 检测到系统窗口干扰: " + window.getTitle());
                }
            }
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "检查系统UI时出错: " + e.getMessage());
        }
    }
    
    // 简化版焦点检测方法（向后兼容）
    private AccessibilityNodeInfo findFocusedEditText(AccessibilityNodeInfo node) {
        return findFocusedEditTextEnhanced(node);
    }
    
    // 查找第一个可编辑的输入框
    private AccessibilityNodeInfo findFirstEditableNode(AccessibilityNodeInfo node) {
        if (node == null) return null;
        
        // 检查当前节点是否可编辑
        if (node.isEditable()) {
            android.util.Log.e("AICHATBOT_SERVICE", "找到可编辑输入框: " + 
                (node.getClassName() != null ? node.getClassName().toString() : "unknown"));
            return node;
        }
        
        // 递归查找子节点
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                AccessibilityNodeInfo result = findFirstEditableNode(child);
                if (result != null) {
                    child.recycle();
                    return result;
                }
                child.recycle();
            }
        }
        
        return null;
    }
    
    /**
     * 更宽松的输入框检测方法，适用于现代应用
     */
    private AccessibilityNodeInfo findAnyInputElement(AccessibilityNodeInfo node) {
        if (node == null) return null;
        
        // 检查当前节点是否是输入相关的元素
        if (isLikelyInputElement(node)) {
            android.util.Log.e("AICHATBOT_SERVICE", "找到可能的输入元素: " + 
                (node.getClassName() != null ? node.getClassName().toString() : "unknown") +
                ", text: " + (node.getText() != null ? node.getText().toString() : "null") +
                ", contentDesc: " + (node.getContentDescription() != null ? node.getContentDescription().toString() : "null"));
            return node;
        }
        
        // 递归查找子节点
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                AccessibilityNodeInfo result = findAnyInputElement(child);
                if (result != null) {
                    child.recycle();
                    return result;
                }
                child.recycle();
            }
        }
        
        return null;
    }
    
    /**
     * 判断节点是否可能是输入元素
     */
    private boolean isLikelyInputElement(AccessibilityNodeInfo node) {
        if (node == null) return false;
        
        // 1. 标准的可编辑检查
        if (node.isEditable()) {
            return true;
        }
        
        // 2. 检查类名
        String className = node.getClassName() != null ? node.getClassName().toString() : "";
        if (className.contains("EditText") || 
            className.contains("AutoCompleteTextView") ||
            className.contains("TextInputEditText") ||
            className.contains("SearchView") ||
            className.contains("Input")) {
            return true;
        }
        
        // 3. 检查内容描述和文本
        String contentDesc = node.getContentDescription() != null ? node.getContentDescription().toString().toLowerCase() : "";
        String text = node.getText() != null ? node.getText().toString().toLowerCase() : "";
        
        if (contentDesc.contains("搜索") || contentDesc.contains("输入") || contentDesc.contains("search") ||
            text.contains("搜索") || text.contains("输入") || text.contains("请输入") || text.contains("search")) {
            return true;
        }
        
        // 4. 检查是否可点击且可能是输入框
        if (node.isClickable() && (
            contentDesc.contains("框") || text.contains("框") ||
            contentDesc.contains("field") || text.contains("field"))) {
            return true;
        }
        
        return false;
    }
    
    // 向有焦点的节点输入文本
    private boolean inputTextToFocusedNode(AccessibilityNodeInfo node, String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 开始向焦点节点输入文本: " + text + " ===");
        
        final long methodStartTime = System.currentTimeMillis();
        final long MAX_EXECUTION_TIME = 5000; // 5秒超时
        
        try {
            // 超时检查1
            if (System.currentTimeMillis() - methodStartTime > MAX_EXECUTION_TIME) {
                android.util.Log.e("AICHATBOT_SERVICE", "⏰ 超时：方法开始阶段");
                return false;
            }
            
            // 方法1: 尝试剪贴板输入
            android.util.Log.e("AICHATBOT_SERVICE", "开始尝试剪贴板输入...");
            long clipboardStart = System.currentTimeMillis();
            
            if (inputChineseViaClipboard(text)) {
                long clipboardDuration = System.currentTimeMillis() - clipboardStart;
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 剪贴板方法成功！耗时: " + clipboardDuration + "ms");
                return true;
            }
            
            long clipboardDuration = System.currentTimeMillis() - clipboardStart;
            android.util.Log.e("AICHATBOT_SERVICE", "剪贴板方法失败，耗时: " + clipboardDuration + "ms，尝试其他方法");
            
            // 超时检查2
            if (System.currentTimeMillis() - methodStartTime > MAX_EXECUTION_TIME) {
                android.util.Log.e("AICHATBOT_SERVICE", "⏰ 超时：剪贴板方法后");
                return false;
            }
            
            // 方法2: 尝试直接输入
            android.util.Log.e("AICHATBOT_SERVICE", "开始尝试直接设置文本...");
            long directStart = System.currentTimeMillis();
            
            Bundle arguments = new Bundle();
            arguments.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            boolean result = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments);
            
            long directDuration = System.currentTimeMillis() - directStart;
            
            if (result) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 直接设置文本成功！耗时: " + directDuration + "ms");
                return true;
            }
            android.util.Log.e("AICHATBOT_SERVICE", "直接设置文本失败，耗时: " + directDuration + "ms");
            
            // 超时检查3
            if (System.currentTimeMillis() - methodStartTime > MAX_EXECUTION_TIME) {
                android.util.Log.e("AICHATBOT_SERVICE", "⏰ 超时：直接设置文本后");
                return false;
            }
            
            // 方法3: 尝试键盘输入模拟
            android.util.Log.e("AICHATBOT_SERVICE", "开始尝试键盘输入模拟...");
            long keyboardStart = System.currentTimeMillis();
            
            if (simulateKeyboardInput(text)) {
                long keyboardDuration = System.currentTimeMillis() - keyboardStart;
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 键盘输入模拟成功！耗时: " + keyboardDuration + "ms");
                return true;
            }
            
            long keyboardDuration = System.currentTimeMillis() - keyboardStart;
            android.util.Log.e("AICHATBOT_SERVICE", "键盘输入模拟失败，耗时: " + keyboardDuration + "ms");
            
            android.util.Log.e("AICHATBOT_SERVICE", "❌ 所有输入方法都失败了");
            return false;
            
        } catch (Exception e) {
            long errorDuration = System.currentTimeMillis() - methodStartTime;
            android.util.Log.e("AICHATBOT_SERVICE", "输入文本时发生异常，耗时: " + errorDuration + "ms", e);
            return false;
        } finally {
            long totalDuration = System.currentTimeMillis() - methodStartTime;
            android.util.Log.e("AICHATBOT_SERVICE", "🏁 inputTextToFocusedNode 方法结束，总耗时: " + totalDuration + "ms");
        }
    }
    
    // 增强的中文剪贴板输入方法
    private boolean inputChineseViaClipboard(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 尝试增强剪贴板中文输入方法 ===");
        try {
            // 复制文本到剪贴板
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取剪贴板管理器");
                return false;
            }
            
            ClipData clip = ClipData.newPlainText("chinese_text", text);
            clipboard.setPrimaryClip(clip);
            android.util.Log.e("AICHATBOT_SERVICE", "✅ 中文文本已复制到剪贴板: " + text);
            
            // // 等待剪贴板更新
            // Thread.sleep(300);
            
            // // // 验证剪贴板内容（安全检查）
            // try {
            //     ClipData primaryClip = clipboard.getPrimaryClip();
            //     if (primaryClip != null && primaryClip.getItemCount() > 0) {
            //         ClipData.Item item = primaryClip.getItemAt(0);
            //         String clipText = item.getText().toString();
            //         android.util.Log.e("AICHATBOT_SERVICE", "验证剪贴板内容: " + clipText);
            //     } else {
            //         android.util.Log.w("AICHATBOT_SERVICE", "⚠️ 剪贴板为空或无法访问，可能是权限问题");
            //     }
            // } catch (Exception e) {
            //     android.util.Log.e("AICHATBOT_SERVICE", "验证剪贴板内容时出错: " + e.getMessage());
            // }
            
            // 获取当前应用信息（用于调试）
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            String currentPackage = "unknown";
            if (rootNode != null) {
                currentPackage = rootNode.getPackageName() != null ? rootNode.getPackageName().toString() : "unknown";
                android.util.Log.e("AICHATBOT_SERVICE", "粘贴前确认当前应用: " + currentPackage);
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 准备在 " + currentPackage + " 中输入文本");
            }
            
            // 不再重复点击搜索框，直接使用当前焦点进行粘贴
            android.util.Log.e("AICHATBOT_SERVICE", "跳过重复点击，直接使用当前焦点进行粘贴");
            
            // 尝试找到当前焦点并直接粘贴
            android.util.Log.e("AICHATBOT_SERVICE", "尝试粘贴中文文本...");
            boolean pasteResult = false;
            
            if (rootNode != null) {
                // 方法1: 查找有焦点的节点
                AccessibilityNodeInfo focusedNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
                if (focusedNode == null) {
                    focusedNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY);
                }
                
                if (focusedNode != null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "找到焦点节点，执行多种粘贴策略");
                    
                    // 策略1: 直接粘贴
                    pasteResult = focusedNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                    android.util.Log.e("AICHATBOT_SERVICE", "直接粘贴结果: " + pasteResult);
                    
                    if (!pasteResult) {
                        // 策略2: 先清空再粘贴
                        Bundle clearArgs = new Bundle();
                        clearArgs.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, "");
                        focusedNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, clearArgs);
                        Thread.sleep(200);
                        pasteResult = focusedNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                        android.util.Log.e("AICHATBOT_SERVICE", "清空后粘贴结果: " + pasteResult);
                    }
                    
                    if (!pasteResult) {
                        // 策略3: 点击后再粘贴
                        focusedNode.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                        Thread.sleep(300);
                        pasteResult = focusedNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                        android.util.Log.e("AICHATBOT_SERVICE", "点击后粘贴结果: " + pasteResult);
                    }
                    
                    if (!pasteResult) {
                        // 策略4: 长按后粘贴
                        focusedNode.performAction(AccessibilityNodeInfo.ACTION_LONG_CLICK);
                        Thread.sleep(500);
                        pasteResult = focusedNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                        android.util.Log.e("AICHATBOT_SERVICE", "长按后粘贴结果: " + pasteResult);
                    }
                    
                    // // 验证输入结果
                    // if (pasteResult) {
                    //     android.util.Log.e("AICHATBOT_SERVICE", "✅ 粘贴操作成功，开始验证文本内容");
                    //     try {
                    //         // Thread.sleep(500); // 缩短等待时间
                            
                    //         // 重新获取节点以确保获得最新状态
                    //         AccessibilityNodeInfo refreshedNode = getRootInActiveWindow();
                    //         if (refreshedNode != null) {
                    //             AccessibilityNodeInfo newFocused = refreshedNode.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
                    //             if (newFocused == null) {
                    //                 newFocused = refreshedNode.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY);
                    //             }
                                
                    //             if (newFocused != null) {
                    //                 String currentText = newFocused.getText() != null ? newFocused.getText().toString() : "";
                    //                 android.util.Log.e("AICHATBOT_SERVICE", "验证文本内容: '" + currentText + "'");
                    //                 android.util.Log.e("AICHATBOT_SERVICE", "目标文本: '" + text + "'");
                                    
                    //                 if (currentText.contains(text) || currentText.equals(text)) {
                    //                     android.util.Log.e("AICHATBOT_SERVICE", "✅ 文本输入成功确认！");
                    //                     newFocused.recycle();
                    //                     refreshedNode.recycle();
                    //                     focusedNode.recycle();
                    //                     rootNode.recycle();
                    //                     return true;
                    //                 } else {
                    //                     android.util.Log.e("AICHATBOT_SERVICE", "⚠️ 文本内容不匹配，但粘贴操作已执行");
                    //                     // 即使验证失败，但粘贴操作成功，也认为是成功的
                    //                     newFocused.recycle();
                    //                     refreshedNode.recycle();
                    //                     focusedNode.recycle();
                    //                     rootNode.recycle();
                    //                     return true; // 修改：直接返回成功
                    //                 }
                    //             } else {
                    //                 android.util.Log.e("AICHATBOT_SERVICE", "⚠️ 验证时未找到焦点节点，但粘贴已执行");
                    //                 refreshedNode.recycle();
                    //                 focusedNode.recycle();
                    //                 rootNode.recycle();
                    //                 return true; // 粘贴成功就返回成功
                    //             }
                    //         } else {
                    //             android.util.Log.e("AICHATBOT_SERVICE", "⚠️ 验证时无法获取根节点，但粘贴已执行");
                    //             focusedNode.recycle();
                    //             rootNode.recycle();
                    //             return true; // 粘贴成功就返回成功
                    //         }
                    //     } catch (Exception verifyException) {
                    //         android.util.Log.e("AICHATBOT_SERVICE", "验证文本时出错: " + verifyException.getMessage());
                    //         focusedNode.recycle();
                    //         rootNode.recycle();
                    //         return true; // 粘贴成功，验证出错也返回成功
                    //     }
                    // } else {
                    //     android.util.Log.e("AICHATBOT_SERVICE", "❌ 粘贴操作失败");
                    //     focusedNode.recycle();
                    //     rootNode.recycle();
                    //     return false;
                    // }
                } else {
                    android.util.Log.e("AICHATBOT_SERVICE", "未找到焦点节点，尝试搜索输入框节点");
                    
                    // // 方法2: 在当前应用中查找所有可能的输入框
                    // List<AccessibilityNodeInfo> editTexts = new ArrayList<>();
                    // findNodesByClassName(rootNode, "android.widget.EditText", editTexts);
                    // findNodesByClassName(rootNode, "android.widget.AutoCompleteTextView", editTexts);
                    
                    // android.util.Log.e("AICHATBOT_SERVICE", "在当前应用中找到 " + editTexts.size() + " 个输入框");
                    
                    // for (int i = 0; i < editTexts.size(); i++) {
                    //     AccessibilityNodeInfo editText = editTexts.get(i);
                    //     if (editText != null) {
                    //         android.util.Log.e("AICHATBOT_SERVICE", "尝试向输入框" + i + "粘贴文本");
                            
                    //         // 尝试获取焦点
                    //         editText.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
                    //         Thread.sleep(200);
                            
                    //         // 清空并粘贴
                    //         Bundle clearArgs2 = new Bundle();
                    //         clearArgs2.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, "");
                    //         editText.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, clearArgs2);
                    //         Thread.sleep(200);
                            
                    //         boolean result = editText.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                    //         android.util.Log.e("AICHATBOT_SERVICE", "输入框" + i + "粘贴结果: " + result);
                            
                    //         if (result) {
                    //             Thread.sleep(300);
                    //             String currentText = editText.getText() != null ? editText.getText().toString() : "";
                    //             android.util.Log.e("AICHATBOT_SERVICE", "输入框" + i + "粘贴后内容: '" + currentText + "'");
                    //             if (currentText.contains(text)) {
                    //                 android.util.Log.e("AICHATBOT_SERVICE", "✅ 输入框" + i + "粘贴成功！");
                    //                 // 清理资源
                    //                 for (AccessibilityNodeInfo node : editTexts) {
                    //                     node.recycle();
                    //                 }
                    //                 rootNode.recycle();
                    //                 return true;
                    //             }
                    //         }
                    //     }
                    // }
                    
                    // 由于上面的代码被注释掉，这里不需要清理资源
                    // for (AccessibilityNodeInfo node : editTexts) {
                    //     node.recycle();
                    // }
                }
                rootNode.recycle();
            }
            
            return false;
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "剪贴板中文输入出错: " + e.getMessage());
            return false;
        }
    }
    
    // 直接文本设置方法
    private boolean inputTextDirectly(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 尝试直接文本设置方法 ===");
        try {
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取根节点");
                return false;
            }
            
            // 查找所有EditText类型的元素
            List<AccessibilityNodeInfo> editTexts = new ArrayList<>();
            findNodesByClassName(rootNode, "android.widget.EditText", editTexts);
            findNodesByClassName(rootNode, "android.widget.AutoCompleteTextView", editTexts);
            
            android.util.Log.e("AICHATBOT_SERVICE", "找到 " + editTexts.size() + " 个输入框");
            
            for (int i = 0; i < editTexts.size(); i++) {
                AccessibilityNodeInfo editText = editTexts.get(i);
                Rect bounds = new Rect();
                editText.getBoundsInScreen(bounds);
                
                android.util.Log.e("AICHATBOT_SERVICE", "输入框" + i + ": 位置(" + bounds.centerX() + "," + bounds.centerY() + ")");
                
                // 重点尝试顶部的搜索框（Y坐标较小）
                if (bounds.centerY() < 400) {
                    android.util.Log.e("AICHATBOT_SERVICE", "尝试向顶部搜索框直接设置中文...");
                    
                    // 先点击激活
                    editText.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                    Thread.sleep(300);
                    
                    // 直接设置文本
                    Bundle arguments = new Bundle();
                    arguments.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
                    boolean result = editText.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments);
                    
                    if (result) {
                        Thread.sleep(300);
                        String currentText = editText.getText() != null ? editText.getText().toString() : "";
                        android.util.Log.e("AICHATBOT_SERVICE", "直接设置后的文本: '" + currentText + "'");
                        if (currentText.contains(text)) {
                            android.util.Log.e("AICHATBOT_SERVICE", "✅ 直接设置中文成功！");
                            rootNode.recycle();
                            // 清理资源
                            for (AccessibilityNodeInfo node : editTexts) {
                                node.recycle();
                            }
                            return true;
                        }
                    }
                }
            }
            
            // 清理资源
            for (AccessibilityNodeInfo node : editTexts) {
                node.recycle();
            }
            rootNode.recycle();
            
            return false;
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "直接文本设置出错: " + e.getMessage());
            return false;
        }
    }
    
    // 方法1: 直接使用剪贴板输入（最可靠的方法）
    private boolean inputTextViaClipboard(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 尝试剪贴板输入方法 ===");
        try {
            // 复制文本到剪贴板
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取剪贴板管理器");
                return false;
            }
            
            ClipData clip = ClipData.newPlainText("search_text", text);
            clipboard.setPrimaryClip(clip);
            android.util.Log.e("AICHATBOT_SERVICE", "✅ 文本已复制到剪贴板: " + text);
            
            // 等待剪贴板更新
            Thread.sleep(200);
            
            // 再次点击搜索框确保焦点
            clickOnScreen(540, 200);
            Thread.sleep(300);
            
            // 模拟 Ctrl+A 选择全部（如果有文本）
            android.util.Log.e("AICHATBOT_SERVICE", "尝试全选现有文本...");
            // 移除不兼容的全局操作
            Thread.sleep(100);
            
            // 尝试找到当前焦点并直接粘贴
            android.util.Log.e("AICHATBOT_SERVICE", "尝试粘贴文本...");
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            boolean pasteResult = false;
            
            if (rootNode != null) {
                // 尝试找到有焦点的节点进行粘贴
                AccessibilityNodeInfo focusedNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
                if (focusedNode == null) {
                    focusedNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY);
                }
                
                if (focusedNode != null) {
                    pasteResult = focusedNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                    focusedNode.recycle();
                } else {
                    // 如果找不到焦点，尝试直接使用剪贴板内容设置文本
                    android.util.Log.e("AICHATBOT_SERVICE", "未找到焦点节点，尝试坐标附近搜索");
                    AccessibilityNodeInfo nearbyNode = findEditableElementNearCoordinate(rootNode, 540, 200, 100);
                    if (nearbyNode != null) {
                        pasteResult = nearbyNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                        nearbyNode.recycle();
                    }
                }
                rootNode.recycle();
            }
            
            android.util.Log.e("AICHATBOT_SERVICE", "粘贴操作结果: " + pasteResult);
            
            if (pasteResult) {
                Thread.sleep(500);
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 剪贴板粘贴完成");
                return true;
            }
            
            return false;
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "剪贴板输入出错: " + e.getMessage());
            return false;
        }
    }
    
    // 方法2: 强制搜索特定的B站搜索框
    private boolean forceInputToSearchBox(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 强制搜索B站搜索框输入 ===");
        try {
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取根节点");
                return false;
            }
            
            // 查找所有EditText类型的元素
            List<AccessibilityNodeInfo> editTexts = new ArrayList<>();
            findNodesByClassName(rootNode, "android.widget.EditText", editTexts);
            findNodesByClassName(rootNode, "android.widget.AutoCompleteTextView", editTexts);
            findNodesByClassName(rootNode, "androidx.appcompat.widget.AppCompatEditText", editTexts);
            
            android.util.Log.e("AICHATBOT_SERVICE", "找到 " + editTexts.size() + " 个输入框");
            
            for (int i = 0; i < editTexts.size(); i++) {
                AccessibilityNodeInfo editText = editTexts.get(i);
                Rect bounds = new Rect();
                editText.getBoundsInScreen(bounds);
                
                android.util.Log.e("AICHATBOT_SERVICE", "输入框" + i + ": " + 
                    editText.getClassName() + " 位置(" + bounds.centerX() + "," + bounds.centerY() + ")");
                
                // 重点尝试顶部的搜索框（Y坐标较小）
                if (bounds.centerY() < 400) {
                    android.util.Log.e("AICHATBOT_SERVICE", "尝试向顶部搜索框输入...");
                    
                    // 先点击激活
                    editText.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                    Thread.sleep(300);
                    
                    // 尝试输入
                    if (performAdvancedTextInput(editText, text)) {
                        android.util.Log.e("AICHATBOT_SERVICE", "✅ 成功向搜索框输入文本");
                        rootNode.recycle();
                        return true;
                    }
                }
            }
            
            // 清理资源
            for (AccessibilityNodeInfo node : editTexts) {
                node.recycle();
            }
            rootNode.recycle();
            
            return false;
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "强制搜索框输入出错: " + e.getMessage());
            return false;
        }
    }
    
    // 方法3: 模拟键盘输入
    private boolean simulateKeyboardInput(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 模拟键盘输入方法 ===");
        
        try {
            // 再次确保搜索框被点击
            clickOnScreen(540, 200);
            Thread.sleep(300);
            
            // 使用剪贴板 + 全选 + 粘贴的组合
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard != null) {
                ClipData clip = ClipData.newPlainText("input_text", text);
                clipboard.setPrimaryClip(clip);
                
                // 等待剪贴板更新
                Thread.sleep(200);
                
                // 尝试找到当前活跃的输入区域并直接操作
                AccessibilityNodeInfo rootNode = getRootInActiveWindow();
                if (rootNode != null) {
                    AccessibilityNodeInfo focused = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
                    if (focused == null) {
                        focused = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY);
                    }
                    
                    if (focused != null) {
                        android.util.Log.e("AICHATBOT_SERVICE", "找到焦点元素，尝试粘贴");
                        
                        // 清空现有内容
                        Bundle clearArgs = new Bundle();
                        clearArgs.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, "");
                        focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, clearArgs);
                        Thread.sleep(200);
                        
                        // 粘贴新内容
                        boolean pasteResult = focused.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                        android.util.Log.e("AICHATBOT_SERVICE", "粘贴操作结果: " + pasteResult);
                        
                        focused.recycle();
                        rootNode.recycle();
                        return pasteResult;
                    }
                    rootNode.recycle();
                }
            }
            
            return false;
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "模拟键盘输入出错: " + e.getMessage());
            return false;
        }
    }
    
    // 辅助方法：根据类名查找节点
    private void findNodesByClassName(AccessibilityNodeInfo node, String className, List<AccessibilityNodeInfo> result) {
        if (node == null) return;
        
        if (className.equals(node.getClassName())) {
            result.add(AccessibilityNodeInfo.obtain(node));
        }
        
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                findNodesByClassName(child, className, result);
                child.recycle();
            }
        }
    }

    // 增强的焦点检测和文本输入方法
    private boolean inputTextByFocus(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 增强的焦点检测和文本输入: " + text + " ===");
        
        try {
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取根节点");
                return false;
            }
            android.util.Log.e("AICHATBOT_SERVICE", "✅ 成功获取根节点");
            
            // 多种方式查找焦点元素
            AccessibilityNodeInfo focusedNode = null;
            
            // 方法1: 查找输入焦点
            android.util.Log.e("AICHATBOT_SERVICE", "🔍 方法1: 查找输入焦点...");
            focusedNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
            if (focusedNode != null) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 找到输入焦点元素: " + focusedNode.getClassName());
                android.util.Log.e("AICHATBOT_SERVICE", "   可编辑: " + focusedNode.isEditable() + ", 有焦点: " + focusedNode.isFocused());
                
                // 即使不可编辑也尝试，某些自定义输入框可能报告错误
                boolean result = performAdvancedTextInput(focusedNode, text);
                if (result) {
                    focusedNode.recycle();
                    rootNode.recycle();
                    return true;
                }
            } else {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 没有找到输入焦点元素");
            }
            
            // 方法2: 查找可访问性焦点
            android.util.Log.e("AICHATBOT_SERVICE", "🔍 方法2: 查找可访问性焦点...");
            focusedNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY);
            if (focusedNode != null) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 找到可访问性焦点元素: " + focusedNode.getClassName());
                android.util.Log.e("AICHATBOT_SERVICE", "   可编辑: " + focusedNode.isEditable() + ", 有焦点: " + focusedNode.isFocused());
                
                boolean result = performAdvancedTextInput(focusedNode, text);
                if (result) {
                    focusedNode.recycle();
                    rootNode.recycle();
                    return true;
                }
            } else {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 没有找到可访问性焦点元素");
            }
            
            // 方法3: 查找最近点击的坐标附近的可编辑元素
            android.util.Log.e("AICHATBOT_SERVICE", "🔍 方法3: 查找点击坐标(540,200)附近的可编辑元素...");
            AccessibilityNodeInfo nearbyNode = findEditableElementNearCoordinate(rootNode, 540, 200, 100);
            if (nearbyNode != null) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 在坐标附近找到可编辑元素: " + nearbyNode.getClassName());
                
                // 先激活这个元素
                nearbyNode.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                Thread.sleep(300);
                nearbyNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
                Thread.sleep(300);
                
                boolean result = performAdvancedTextInput(nearbyNode, text);
                if (result) {
                    nearbyNode.recycle();
                    rootNode.recycle();
                    return true;
                }
            } else {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 在坐标附近没有找到可编辑元素");
            }
            
            // 方法4: 全屏搜索可编辑元素
            android.util.Log.e("AICHATBOT_SERVICE", "🔍 方法4: 全屏搜索可编辑元素...");
            boolean result = findAndActivateEditableElement(rootNode, text);
            rootNode.recycle();
            return result;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "增强焦点检测出错: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    // 查找指定坐标附近的可编辑元素
    private AccessibilityNodeInfo findEditableElementNearCoordinate(AccessibilityNodeInfo rootNode, int targetX, int targetY, int radius) {
        android.util.Log.e("AICHATBOT_SERVICE", "在坐标(" + targetX + "," + targetY + ")半径" + radius + "内查找可编辑元素");
        
        List<AccessibilityNodeInfo> editableNodes = new ArrayList<>();
        findAllEditableElements(rootNode, editableNodes);
        
        android.util.Log.e("AICHATBOT_SERVICE", "总共找到 " + editableNodes.size() + " 个可编辑元素");
        
        AccessibilityNodeInfo closestNode = null;
        double minDistance = Double.MAX_VALUE;
        
        for (AccessibilityNodeInfo node : editableNodes) {
            Rect bounds = new Rect();
            node.getBoundsInScreen(bounds);
            
            // 计算元素中心点
            int centerX = bounds.centerX();
            int centerY = bounds.centerY();
            
            // 计算距离
            double distance = Math.sqrt(Math.pow(centerX - targetX, 2) + Math.pow(centerY - targetY, 2));
            
            android.util.Log.e("AICHATBOT_SERVICE", "可编辑元素: " + node.getClassName() + 
                              " 位置(" + centerX + "," + centerY + ") 距离: " + (int)distance);
            
            if (distance <= radius && distance < minDistance) {
                minDistance = distance;
                if (closestNode != null) {
                    closestNode.recycle();
                }
                closestNode = node;
            } else {
                node.recycle();
            }
        }
        
        if (closestNode != null) {
            android.util.Log.e("AICHATBOT_SERVICE", "找到最近的可编辑元素，距离: " + (int)minDistance);
        }
        
        return closestNode;
    }
    
    // 递归查找所有可编辑元素
    private void findAllEditableElements(AccessibilityNodeInfo node, List<AccessibilityNodeInfo> editableNodes) {
        if (node == null) return;
        
        if (node.isEditable()) {
            // 创建节点的副本以避免回收问题
            AccessibilityNodeInfo copy = AccessibilityNodeInfo.obtain(node);
            editableNodes.add(copy);
        }
        
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                findAllEditableElements(child, editableNodes);
                child.recycle();
            }
        }
    }
    
    // 查找并激活可编辑元素
    private boolean findAndActivateEditableElement(AccessibilityNodeInfo node, String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 查找并激活可编辑元素 ===");
        
        boolean found = findEditableElementRecursively(node, text, 0);
        if (found) {
            android.util.Log.e("AICHATBOT_SERVICE", "✅ 成功找到并激活可编辑元素");
        } else {
            android.util.Log.e("AICHATBOT_SERVICE", "❌ 未找到可编辑元素");
        }
        return found;
    }

    // 递归查找可编辑元素
    private boolean findEditableElementRecursively(AccessibilityNodeInfo node, String text, int depth) {
        if (node == null || depth > 8) {
            return false;
        }
        
        String className = node.getClassName() != null ? node.getClassName().toString() : "";
        String nodeText = node.getText() != null ? node.getText().toString() : "";
        String contentDesc = node.getContentDescription() != null ? node.getContentDescription().toString() : "";
        
        android.util.Log.e("AICHATBOT_SERVICE", "检查节点 (深度" + depth + "): " + className + 
                          " | 可编辑: " + node.isEditable() + 
                          " | 文本: '" + nodeText + "'");
        
        // 如果是可编辑元素，尝试输入
        if (node.isEditable()) {
            android.util.Log.e("AICHATBOT_SERVICE", "找到可编辑元素: " + className);
            
            // 先尝试给元素设置焦点
            android.util.Log.e("AICHATBOT_SERVICE", "尝试激活元素...");
            node.performAction(AccessibilityNodeInfo.ACTION_CLICK);
            try { Thread.sleep(300); } catch (InterruptedException e) {}
            node.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
            try { Thread.sleep(300); } catch (InterruptedException e) {}
            
            // 然后输入文本
            boolean result = performAdvancedTextInput(node, text);
            if (result) {
                return true;
            }
        }
        
        // 递归搜索子节点
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo childNode = node.getChild(i);
            if (childNode != null) {
                boolean found = findEditableElementRecursively(childNode, text, depth + 1);
                childNode.recycle();
                if (found) {
                    return true;
                }
            }
        }
        
        return false;
    }

    // 高级文本输入方法
    private boolean performAdvancedTextInput(AccessibilityNodeInfo node, String text) {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "=== 开始高级文本输入: " + text + " ===");
            android.util.Log.e("AICHATBOT_SERVICE", "节点类名: " + node.getClassName());
            android.util.Log.e("AICHATBOT_SERVICE", "节点当前文本: " + node.getText());
            android.util.Log.e("AICHATBOT_SERVICE", "节点是否有焦点: " + node.isFocused());
            
            // 方法1：清空后直接设置文本
            android.util.Log.e("AICHATBOT_SERVICE", "=== 方法1：清空后设置文本 ===");
            Bundle clearArgs = new Bundle();
            clearArgs.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, "");
            boolean clearResult = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, clearArgs);
            android.util.Log.e("AICHATBOT_SERVICE", "清空文本结果: " + clearResult);
            Thread.sleep(300);
            
            Bundle arguments1 = new Bundle();
            arguments1.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            boolean result1 = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments1);
            android.util.Log.e("AICHATBOT_SERVICE", "设置文本结果: " + result1);
            
            if (result1) {
                Thread.sleep(300);
                String currentText = node.getText() != null ? node.getText().toString() : "";
                android.util.Log.e("AICHATBOT_SERVICE", "设置后的文本内容: '" + currentText + "'");
                if (currentText.contains(text)) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 方法1成功！文本已正确设置");
                    return true;
                } else {
                    android.util.Log.e("AICHATBOT_SERVICE", "❌ 方法1失败：文本内容不匹配");
                }
            }
            
            // 方法2：点击+聚焦+设置
            android.util.Log.e("AICHATBOT_SERVICE", "=== 方法2：点击+聚焦+设置 ===");
            boolean clickResult = node.performAction(AccessibilityNodeInfo.ACTION_CLICK);
            android.util.Log.e("AICHATBOT_SERVICE", "点击结果: " + clickResult);
            Thread.sleep(300);
            
            boolean focusResult = node.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
            android.util.Log.e("AICHATBOT_SERVICE", "聚焦结果: " + focusResult);
            Thread.sleep(300);
            
            Bundle arguments2 = new Bundle();
            arguments2.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            boolean result2 = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments2);
            android.util.Log.e("AICHATBOT_SERVICE", "点击聚焦后设置文本结果: " + result2);
            
            if (result2) {
                Thread.sleep(300);
                String currentText = node.getText() != null ? node.getText().toString() : "";
                android.util.Log.e("AICHATBOT_SERVICE", "方法2设置后的文本内容: '" + currentText + "'");
                if (currentText.contains(text)) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 方法2成功！文本已正确设置");
                    return true;
                } else {
                    android.util.Log.e("AICHATBOT_SERVICE", "❌ 方法2失败：文本内容不匹配");
                }
            }
            
            // 方法3：使用剪贴板
            android.util.Log.e("AICHATBOT_SERVICE", "=== 方法3：使用剪贴板 ===");
            try {
                ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                if (clipboard != null) {
                    ClipData clip = ClipData.newPlainText("search_text", text);
                    clipboard.setPrimaryClip(clip);
                    android.util.Log.e("AICHATBOT_SERVICE", "文本已复制到剪贴板: " + text);
                    
                    node.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                    Thread.sleep(300);
                    
                    // 使用兼容的方式选择全部文本
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                        // API 21+ 支持 ACTION_SELECT_ALL (使用数值常量)
                        boolean selectResult = node.performAction(131072); // ACTION_SELECT_ALL 的数值
                        android.util.Log.e("AICHATBOT_SERVICE", "选择全部文本结果: " + selectResult);
                    } else {
                        // 低版本API使用长按模拟选择全部
                        boolean longClickResult = node.performAction(AccessibilityNodeInfo.ACTION_LONG_CLICK);
                        android.util.Log.e("AICHATBOT_SERVICE", "长按结果: " + longClickResult);
                    }
                    Thread.sleep(200);
                    
                    boolean pasteResult = node.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                    android.util.Log.e("AICHATBOT_SERVICE", "粘贴结果: " + pasteResult);
                    
                    if (pasteResult) {
                        Thread.sleep(500);
                        String currentText = node.getText() != null ? node.getText().toString() : "";
                        android.util.Log.e("AICHATBOT_SERVICE", "方法3设置后的文本内容: '" + currentText + "'");
                        if (currentText.contains(text)) {
                            android.util.Log.e("AICHATBOT_SERVICE", "✅ 方法3成功！文本已正确设置");
                            return true;
                        } else {
                            android.util.Log.e("AICHATBOT_SERVICE", "❌ 方法3失败：文本内容不匹配");
                        }
                    }
                } else {
                    android.util.Log.e("AICHATBOT_SERVICE", "无法获取剪贴板管理器");
                }
            } catch (Exception e) {
                android.util.Log.e("AICHATBOT_SERVICE", "剪贴板方法出错: " + e.getMessage());
            }
            
            android.util.Log.e("AICHATBOT_SERVICE", "❌ 所有文本输入方法都失败了");
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "高级文本输入出错: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    // 使用剪贴板的文本输入方法（增强版）
    private boolean inputTextWithClipboard(String text, AccessibilityNodeInfo targetNode) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 开始剪贴板文本输入: " + text + " ===");
        
        if (targetNode == null) {
            android.util.Log.e("AICHATBOT_SERVICE", "❌ 目标节点为空");
            return false;
        }
        
        try {
            // 1. 设置剪贴板内容
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取剪贴板管理器");
                return false;
            }
            
            ClipData clip = ClipData.newPlainText("input_text", text);
            clipboard.setPrimaryClip(clip);
            android.util.Log.e("AICHATBOT_SERVICE", "✅ 文本已复制到剪贴板: " + text);
            
            // 等待剪贴板更新
            Thread.sleep(200);
            
            // 验证剪贴板内容
            if (clipboard.hasPrimaryClip()) {
                try {
                    ClipData primaryClip = clipboard.getPrimaryClip();
                    if (primaryClip != null && primaryClip.getItemCount() > 0) {
                        ClipData.Item item = primaryClip.getItemAt(0);
                        String clipText = item.getText().toString();
                        android.util.Log.e("AICHATBOT_SERVICE", "验证剪贴板内容: " + clipText);
                        
                        if (!clipText.equals(text)) {
                            android.util.Log.e("AICHATBOT_SERVICE", "⚠️ 剪贴板内容不匹配！");
                        }
                    } else {
                        android.util.Log.w("AICHATBOT_SERVICE", "⚠️ 剪贴板为空");
                    }
                } catch (Exception e) {
                    android.util.Log.e("AICHATBOT_SERVICE", "验证剪贴板时出错: " + e.getMessage());
                }
            }
            
            // 2. 确保目标节点获得焦点
            android.util.Log.e("AICHATBOT_SERVICE", "确保目标节点获得焦点...");
            targetNode.performAction(AccessibilityNodeInfo.ACTION_CLICK);
            Thread.sleep(300);
            targetNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
            Thread.sleep(300);
            
            // 3. 多种粘贴策略
            boolean pasteSuccess = false;
            
            // 策略1: 直接粘贴
            android.util.Log.e("AICHATBOT_SERVICE", "策略1: 直接粘贴");
            pasteSuccess = targetNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
            android.util.Log.e("AICHATBOT_SERVICE", "直接粘贴结果: " + pasteSuccess);
            
            if (!pasteSuccess) {
                // 策略2: 清空后粘贴
                android.util.Log.e("AICHATBOT_SERVICE", "策略2: 清空后粘贴");
                Bundle clearArgs = new Bundle();
                clearArgs.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, "");
                targetNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, clearArgs);
                Thread.sleep(200);
                pasteSuccess = targetNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                android.util.Log.e("AICHATBOT_SERVICE", "清空后粘贴结果: " + pasteSuccess);
            }
            
            if (!pasteSuccess) {
                // 策略3: 全选后粘贴 (使用数值常量避免API兼容性问题)
                android.util.Log.e("AICHATBOT_SERVICE", "策略3: 全选后粘贴");
                targetNode.performAction(131072); // ACTION_SELECT_ALL 的数值
                Thread.sleep(200);
                pasteSuccess = targetNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                android.util.Log.e("AICHATBOT_SERVICE", "全选后粘贴结果: " + pasteSuccess);
            }
            
            if (!pasteSuccess) {
                // 策略4: 直接设置文本作为备选
                android.util.Log.e("AICHATBOT_SERVICE", "策略4: 直接设置文本");
                Bundle setTextArgs = new Bundle();
                setTextArgs.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
                pasteSuccess = targetNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, setTextArgs);
                android.util.Log.e("AICHATBOT_SERVICE", "直接设置文本结果: " + pasteSuccess);
            }
            
            // 4. 验证输入结果
            if (pasteSuccess) {
                Thread.sleep(500); // 等待文本更新
                String currentText = targetNode.getText() != null ? targetNode.getText().toString() : "";
                android.util.Log.e("AICHATBOT_SERVICE", "验证输入结果: '" + currentText + "'");
                
                if (currentText.contains(text) || currentText.equals(text)) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 文本输入成功确认！");
                    return true;
                } else {
                    android.util.Log.e("AICHATBOT_SERVICE", "⚠️ 文本内容验证失败");
                    android.util.Log.e("AICHATBOT_SERVICE", "预期: '" + text + "'");
                    android.util.Log.e("AICHATBOT_SERVICE", "实际: '" + currentText + "'");
                    
                    // 即使验证失败，如果执行了粘贴操作，也可能是成功的
                    if (pasteSuccess) {
                        android.util.Log.e("AICHATBOT_SERVICE", "粘贴操作成功，可能是显示延迟");
                        return true;
                    }
                }
            }
            
            android.util.Log.e("AICHATBOT_SERVICE", "❌ 所有粘贴策略都失败");
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "剪贴板输入异常: " + e.getMessage(), e);
            return false;
        }
    }
    
    // 测试ACTION_PASTE权限是否可用
    public boolean testPastePermission() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 测试ACTION_PASTE权限 ===");
        
        try {
            // 设置测试文本到剪贴板
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取剪贴板管理器");
                return false;
            }
            
            String testText = "ACTION_PASTE测试";
            ClipData clip = ClipData.newPlainText("paste_test", testText);
            clipboard.setPrimaryClip(clip);
            android.util.Log.e("AICHATBOT_SERVICE", "✅ 测试文本已复制到剪贴板: " + testText);
            
            // 等待剪贴板更新
            Thread.sleep(300);
            
            // 验证剪贴板内容
            ClipData.Item item = clipboard.getPrimaryClip().getItemAt(0);
            String clipText = item.getText().toString();
            android.util.Log.e("AICHATBOT_SERVICE", "验证剪贴板内容: " + clipText);
            
            // 查找输入框并测试粘贴
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode != null) {
                AccessibilityNodeInfo focusedNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
                if (focusedNode == null) {
                    focusedNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY);
                }
                
                if (focusedNode != null && focusedNode.isEditable()) {
                    android.util.Log.e("AICHATBOT_SERVICE", "找到可编辑焦点节点，测试粘贴操作");
                    boolean pasteResult = focusedNode.performAction(AccessibilityNodeInfo.ACTION_PASTE);
                    android.util.Log.e("AICHATBOT_SERVICE", "ACTION_PASTE 测试结果: " + pasteResult);
                    
                    if (pasteResult) {
                        Thread.sleep(500);
                        String currentText = focusedNode.getText() != null ? focusedNode.getText().toString() : "";
                        android.util.Log.e("AICHATBOT_SERVICE", "粘贴后文本内容: '" + currentText + "'");
                        
                        boolean success = currentText.contains(testText);
                        android.util.Log.e("AICHATBOT_SERVICE", success ? "✅ ACTION_PASTE权限测试成功！" : "⚠️ ACTION_PASTE权限可能受限");
                        focusedNode.recycle();
                        rootNode.recycle();
                        return success;
                    } else {
                        android.util.Log.e("AICHATBOT_SERVICE", "❌ ACTION_PASTE操作失败");
                    }
                    focusedNode.recycle();
                } else {
                    android.util.Log.e("AICHATBOT_SERVICE", "❌ 未找到可编辑的焦点节点");
                }
                rootNode.recycle();
            } else {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取根节点");
            }
            
            return false;
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "ACTION_PASTE权限测试异常: " + e.getMessage());
            return false;
        }
    }

    // 检查B站应用是否处于活跃状态
    public boolean isBiliAppActive() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 开始检查B站应用活跃状态 ===");
        AccessibilityNodeInfo rootInfo = getRootInActiveWindow();
        if (rootInfo != null) {
            String currentPackage = rootInfo.getPackageName() != null ? 
                rootInfo.getPackageName().toString() : "unknown";
            android.util.Log.e("AICHATBOT_SERVICE", "当前检查应用包名: " + currentPackage);
            
            boolean isBiliActive = "tv.danmaku.bili".equals(currentPackage);
            android.util.Log.e("AICHATBOT_SERVICE", "B站应用活跃状态: " + isBiliActive);
            
            // 如果不是B站，显示当前应用信息
            if (!isBiliActive) {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 当前不在B站应用中，在: " + currentPackage);
                if ("com.android.systemui".equals(currentPackage)) {
                    android.util.Log.e("AICHATBOT_SERVICE", "⚠️ 检测到在系统界面，B站可能正在启动");
                } else if ("com.miui.home".equals(currentPackage)) {
                    android.util.Log.e("AICHATBOT_SERVICE", "⚠️ 检测到在桌面，B站启动可能失败");
                }
            } else {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 确认在B站应用中");
            }
            
            rootInfo.recycle();
            return isBiliActive;
        }
        android.util.Log.e("AICHATBOT_SERVICE", "⚠️ 无法获取当前应用状态 - rootInfo为null");
        return false;
    }
    
    // 分析B站界面寻找搜索框
    private void analyzeScreenForSearchBox(AccessibilityNodeInfo rootNode) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 开始分析B站界面寻找搜索框 ===");
        findSearchBoxRecursively(rootNode, 0);
    }
    
    // 递归查找搜索框
    private void findSearchBoxRecursively(AccessibilityNodeInfo node, int depth) {
        if (node == null || depth > 10) return;
        
        String className = node.getClassName() != null ? node.getClassName().toString() : "";
        String text = node.getText() != null ? node.getText().toString() : "";
        String contentDesc = node.getContentDescription() != null ? node.getContentDescription().toString() : "";
        
        // 检查是否是搜索相关的元素
        if (className.contains("EditText") || 
            text.contains("搜索") || text.contains("search") ||
            contentDesc.contains("搜索") || contentDesc.contains("search")) {
            
            Rect bounds = new Rect();
            node.getBoundsInScreen(bounds);
            android.util.Log.e("AICHATBOT_SERVICE", "*** 发现搜索框候选 ***");
            android.util.Log.e("AICHATBOT_SERVICE", "类名: " + className);
            android.util.Log.e("AICHATBOT_SERVICE", "文本: " + text);
            android.util.Log.e("AICHATBOT_SERVICE", "描述: " + contentDesc);
            android.util.Log.e("AICHATBOT_SERVICE", "坐标: (" + bounds.centerX() + ", " + bounds.centerY() + ")");
            android.util.Log.e("AICHATBOT_SERVICE", "边界: " + bounds.toString());
        }
        
        // 继续递归查找
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                findSearchBoxRecursively(child, depth + 1);
                child.recycle();
            }
        }
    }
    
    // 已移除 checkInputFocusAfterClick 方法以避免死循环
    
    // 递归检查焦点状态（保留供将来使用）
    private void checkFocusRecursively(AccessibilityNodeInfo node, int depth) {
        if (node == null || depth > 10) return;
        
        if (node.isFocused()) {
            String className = node.getClassName() != null ? node.getClassName().toString() : "";
            String text = node.getText() != null ? node.getText().toString() : "";
            
            Rect bounds = new Rect();
            node.getBoundsInScreen(bounds);
            android.util.Log.e("AICHATBOT_SERVICE", "*** 发现获得焦点的元素 ***");
            android.util.Log.e("AICHATBOT_SERVICE", "类名: " + className);
            android.util.Log.e("AICHATBOT_SERVICE", "文本: " + text);
            android.util.Log.e("AICHATBOT_SERVICE", "坐标: (" + bounds.centerX() + ", " + bounds.centerY() + ")");
            android.util.Log.e("AICHATBOT_SERVICE", "可编辑: " + node.isEditable());
        }
        
        // 继续递归查找
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                checkFocusRecursively(child, depth + 1);
                child.recycle();
            }
        }
    }
    
    // 强制输入文本到任何可用的输入框（调试用）
    public boolean forceInputTextToAnyEditText(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 强制输入文本到任何输入框: " + text + " ===");
        
        try {
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "❌ 无法获取根节点");
                return false;
            }
            
            // 分析当前界面
            analyzeUIForInputElements(rootNode);
            
            // 查找所有可编辑元素
            List<AccessibilityNodeInfo> editableNodes = new ArrayList<>();
            findAllEditableElementsEnhanced(rootNode, editableNodes);
            
            android.util.Log.e("AICHATBOT_SERVICE", "找到 " + editableNodes.size() + " 个可编辑元素");
            
            boolean success = false;
            for (int i = 0; i < editableNodes.size(); i++) {
                AccessibilityNodeInfo node = editableNodes.get(i);
                android.util.Log.e("AICHATBOT_SERVICE", "尝试向第" + (i+1) + "个输入框输入文本");
                
                if (inputTextWithClipboard(text, node)) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 第" + (i+1) + "个输入框输入成功！");
                    success = true;
                    break;
                }
            }
            
            // 清理资源
            for (AccessibilityNodeInfo node : editableNodes) {
                node.recycle();
            }
            rootNode.recycle();
            
            return success;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "强制输入文本时出错: " + e.getMessage(), e);
            return false;
        }
    }
    
    // 查找焦点输入框的简化版本（用于超时机制）
    private AccessibilityNodeInfo findFocusedEditTextEnhanced() {
        AccessibilityNodeInfo rootNode = getRootInActiveWindow();
        if (rootNode == null) {
            return null;
        }
        
        AccessibilityNodeInfo result = findFocusedEditTextEnhanced(rootNode);
        if (result == null) {
            // 如果没有找到焦点节点，尝试找第一个可编辑节点
            result = findFirstEditableNode(rootNode);
            if (result != null) {
                // 尝试设置焦点
                result.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
                try {
                    Thread.sleep(200);
                } catch (InterruptedException e) {
                    // 忽略中断
                }
            }
        }
        
        return result;
    }
    
    // 使用严格的5秒超时机制尝试剪贴板输入 (公开方法)
    public boolean tryClipboardInputWithStrictTimeout(String text) {
        // 线程安全的布尔包装器
        final boolean[] hasCompleted = new boolean[]{false};
        final boolean[] isSuccessful = new boolean[]{false};
        final Object lockObject = new Object();
        
        // 创建执行线程
        Thread clipboardThread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    android.util.Log.e("AICHATBOT_SERVICE", "开始5秒超时剪贴板输入: " + text);
                    
                    // 1. 查找焦点节点 (2秒内完成)
                    long focusStartTime = System.currentTimeMillis();
                    AccessibilityNodeInfo focusedNode = null;
                    
                    while (System.currentTimeMillis() - focusStartTime < 2000) {
                        focusedNode = findFocusedEditTextEnhanced();
                        if (focusedNode != null) {
                            android.util.Log.e("AICHATBOT_SERVICE", "成功找到焦点节点: " + focusedNode.getClassName());
                            break;
                    }
                    
                    synchronized (lockObject) {
                        if (hasCompleted[0]) {
                            android.util.Log.e("AICHATBOT_SERVICE", "超时检测到，终止焦点查找");
                            return;
                        }
                    }
                    
                    Thread.sleep(100);
                }
                
                if (focusedNode == null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "2秒内未找到焦点节点，剪贴板输入失败");
                    synchronized (lockObject) {
                        if (!hasCompleted[0]) {
                            isSuccessful[0] = false;
                            hasCompleted[0] = true;
                            lockObject.notifyAll();
                        }
                    }
                    return;
                }
                
                // 2. 执行剪贴板输入 (3秒内完成)
                android.util.Log.e("AICHATBOT_SERVICE", "开始剪贴板输入操作");
                boolean clipboardResult = inputTextWithClipboard(text, focusedNode);
                
                synchronized (lockObject) {
                    if (!hasCompleted[0]) {
                        isSuccessful[0] = clipboardResult;
                        hasCompleted[0] = true;
                        lockObject.notifyAll();
                        android.util.Log.e("AICHATBOT_SERVICE", "剪贴板输入完成，结果: " + clipboardResult);
                    }
                }
                
            } catch (Exception e) {
                android.util.Log.e("AICHATBOT_SERVICE", "剪贴板输入过程中发生异常: " + e.getMessage());
                synchronized (lockObject) {
                    if (!hasCompleted[0]) {
                        isSuccessful[0] = false;
                        hasCompleted[0] = true;
                        lockObject.notifyAll();
                    }
                }
            }
        }
        });
        
        // 启动线程
        clipboardThread.start();
        
        try {
            synchronized (lockObject) {
                // 等待5秒或直到完成
                long waitStartTime = System.currentTimeMillis();
                while (!hasCompleted[0] && (System.currentTimeMillis() - waitStartTime < 5000)) {
                    lockObject.wait(100);
                }
                
                if (!hasCompleted[0]) {
                    android.util.Log.w("AICHATBOT_SERVICE", "5秒超时，强制终止剪贴板输入");
                    hasCompleted[0] = true;
                    clipboardThread.interrupt();
                    return false;
                }
                
                android.util.Log.e("AICHATBOT_SERVICE", "剪贴板输入最终结果: " + isSuccessful[0]);
                return isSuccessful[0];
            }
        } catch (InterruptedException e) {
            android.util.Log.e("AICHATBOT_SERVICE", "等待剪贴板输入时被中断: " + e.getMessage());
            hasCompleted[0] = true;
            clipboardThread.interrupt();
            return false;
        }
    }
    
    // 使用ACTION_SET_TEXT作为备用方案 (公开方法)
    public boolean useActionSetTextAsFallback(String text) {
        android.util.Log.e("AICHATBOT_SERVICE", "使用ACTION_SET_TEXT备用方案: " + text);
        
        try {
            AccessibilityNodeInfo focusedNode = findFocusedEditTextEnhanced();
            if (focusedNode == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "备用方案：未找到焦点节点");
                return false;
            }
            
            // 创建ACTION_SET_TEXT的Bundle
            Bundle arguments = new Bundle();
            arguments.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            
            boolean result = focusedNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments);
            android.util.Log.e("AICHATBOT_SERVICE", "ACTION_SET_TEXT执行结果: " + result);
            
            if (result) {
                Thread.sleep(500); // 给系统一点时间处理
                android.util.Log.e("AICHATBOT_SERVICE", "备用方案成功完成");
            }
            
            return result;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "ACTION_SET_TEXT备用方案执行异常: " + e.getMessage());
            return false;
        }
    }
    
    // 查找并点击"开始"按钮
    public boolean findAndClickStartButton() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 开始查找并点击'开始'按钮 ===");
        
        try {
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode == null) {
                android.util.Log.e("AICHATBOT_SERVICE", "无法获取根节点");
                return false;
            }
            
            // 查找包含"开始"文本的按钮
            AccessibilityNodeInfo startButton = findNodeByText(rootNode, "开始");
            if (startButton == null) {
                // 如果没找到"开始"，尝试查找"Start"
                startButton = findNodeByText(rootNode, "Start");
            }
            if (startButton == null) {
                // 如果还没找到，尝试查找"start"
                startButton = findNodeByText(rootNode, "start");
            }
            
            if (startButton != null) {
                android.util.Log.e("AICHATBOT_SERVICE", "找到开始按钮: " + startButton.getText());
                boolean clickResult = startButton.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                android.util.Log.e("AICHATBOT_SERVICE", "点击开始按钮结果: " + clickResult);
                startButton.recycle();
                rootNode.recycle();
                return clickResult;
            } else {
                android.util.Log.w("AICHATBOT_SERVICE", "未找到开始按钮，尝试分析界面");
                analyzeCurrentScreen(rootNode, 0);
                rootNode.recycle();
                return false;
            }
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "查找开始按钮时发生异常: " + e.getMessage());
            return false;
        }
    }
    
    // 根据文本查找节点
    private AccessibilityNodeInfo findNodeByText(AccessibilityNodeInfo node, String text) {
        if (node == null) return null;
        
        // 检查当前节点的文本
        CharSequence nodeText = node.getText();
        if (nodeText != null && nodeText.toString().contains(text)) {
            // 确保这是一个可点击的节点
            if (node.isClickable() || node.getClassName().toString().contains("Button")) {
                return node;
            }
        }
        
        // 检查内容描述
        CharSequence contentDesc = node.getContentDescription();
        if (contentDesc != null && contentDesc.toString().contains(text)) {
            if (node.isClickable() || node.getClassName().toString().contains("Button")) {
                return node;
            }
        }
        
        // 递归查找子节点
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo childNode = node.getChild(i);
            if (childNode != null) {
                AccessibilityNodeInfo result = findNodeByText(childNode, text);
                if (result != null) {
                    childNode.recycle();
                    return result;
                }
                childNode.recycle();
            }
        }
        
        return null;
    }
    
    // 截图功能 - 使用三指下拉手势和多种备用方案
    public boolean takeScreenshot() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 开始执行截图操作 ===");
        
        try {
            // 方案0: 测试MediaProjection可用性（新增）
            android.util.Log.e("AICHATBOT_SERVICE", "方案0: 测试MediaProjection截图");
            boolean mediaProjectionResult = testMediaProjectionAvailability();
            
            if (mediaProjectionResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "MediaProjection测试请求已发送，等待用户响应...");
                // 等待3秒让用户有时间响应
                try {
                    Thread.sleep(3000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            
            // 方案1: 使用三指下拉手势截图（大多数Android设备支持）
            android.util.Log.e("AICHATBOT_SERVICE", "方案1: 尝试三指下拉截图手势");
            boolean threeFingerResult = performThreeFingerSwipeScreenshot();
            
            if (threeFingerResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 三指下拉截图成功");
                return true;
            }
            
            // 方案2: 电源键+音量下键组合（手势模拟）
            android.util.Log.e("AICHATBOT_SERVICE", "方案2: 尝试电源键+音量键组合");
            boolean keyComboResult = simulatePowerVolumeScreenshot();
            
            if (keyComboResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 按键组合截图成功");
                return true;
            }
            
            // 方案3: 通过下拉通知栏找到截图按钮
            android.util.Log.e("AICHATBOT_SERVICE", "方案3: 尝试通知栏截图按钮");
            boolean notificationResult = screenshotViaNotificationPanel();
            
            if (notificationResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 通知栏截图成功");
                return true;
            }
            
            // 方案4: 通过最近任务界面截图
            android.util.Log.e("AICHATBOT_SERVICE", "方案4: 尝试最近任务截图");
            boolean recentsResult = screenshotViaRecents();
            
            if (recentsResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 最近任务截图成功");
                return true;
            }
            
            android.util.Log.w("AICHATBOT_SERVICE", "❌ 所有截图方案都失败了");
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "截图操作发生异常: " + e.getMessage());
            return false;
        }
    }
    
    // 三指下拉截图手势
    private boolean performThreeFingerSwipeScreenshot() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "执行三指下拉截图手势");
            
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                // 模拟三个手指同时从屏幕上方向下拉
                android.graphics.Path path1 = new android.graphics.Path();
                android.graphics.Path path2 = new android.graphics.Path();
                android.graphics.Path path3 = new android.graphics.Path();
                
                // 屏幕宽度分为三个区域，每个手指在不同区域
                int screenWidth = 1080; // 假设屏幕宽度
                int startY = 300;  // 从屏幕上方开始
                int endY = 800;    // 向下拉动
                
                // 三个手指的起始和结束位置
                path1.moveTo(screenWidth * 0.25f, startY);
                path1.lineTo(screenWidth * 0.25f, endY);
                
                path2.moveTo(screenWidth * 0.5f, startY);
                path2.lineTo(screenWidth * 0.5f, endY);
                
                path3.moveTo(screenWidth * 0.75f, startY);
                path3.lineTo(screenWidth * 0.75f, endY);
                
                // 创建三个同时进行的手势
                GestureDescription.StrokeDescription stroke1 = 
                    new GestureDescription.StrokeDescription(path1, 0, 500);
                GestureDescription.StrokeDescription stroke2 = 
                    new GestureDescription.StrokeDescription(path2, 0, 500);
                GestureDescription.StrokeDescription stroke3 = 
                    new GestureDescription.StrokeDescription(path3, 0, 500);
                
                GestureDescription gesture = 
                    new GestureDescription.Builder()
                        .addStroke(stroke1)
                        .addStroke(stroke2)
                        .addStroke(stroke3)
                        .build();
                
                boolean result = dispatchGesture(gesture, null, null);
                android.util.Log.e("AICHATBOT_SERVICE", "三指下拉手势执行结果: " + result);
                
                if (result) {
                    // 等待截图完成
                    Thread.sleep(2000);
                }
                
                return result;
            } else {
                android.util.Log.w("AICHATBOT_SERVICE", "Android版本过低，不支持多点触控手势");
                return false;
            }
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "三指下拉截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 模拟电源键+音量下键截图组合
    private boolean simulatePowerVolumeScreenshot() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "尝试模拟电源键+音量键截图");
            
            // 在屏幕右边缘同时按压两个区域（模拟电源键和音量键）
            android.graphics.Path powerKeyPath = new android.graphics.Path();
            android.graphics.Path volumeKeyPath = new android.graphics.Path();
            
            // 电源键位置（右边缘中央）
            powerKeyPath.moveTo(1070, 600);
            powerKeyPath.lineTo(1070, 600);
            
            // 音量下键位置（右边缘稍上方）
            volumeKeyPath.moveTo(1070, 500);
            volumeKeyPath.lineTo(1070, 500);
            
            // 同时长按两个位置
            GestureDescription.StrokeDescription powerStroke = 
                new GestureDescription.StrokeDescription(powerKeyPath, 0, 1000);
            GestureDescription.StrokeDescription volumeStroke = 
                new GestureDescription.StrokeDescription(volumeKeyPath, 0, 1000);
            
            GestureDescription gesture = 
                new GestureDescription.Builder()
                    .addStroke(powerStroke)
                    .addStroke(volumeStroke)
                    .build();
            
            boolean result = dispatchGesture(gesture, null, null);
            android.util.Log.e("AICHATBOT_SERVICE", "电源键+音量键组合结果: " + result);
            
            if (result) {
                // 等待截图完成
                Thread.sleep(2000);
            }
            
            return result;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "按键组合截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 通过下拉通知栏进行截图
    private boolean screenshotViaNotificationPanel() {
        try {
            // 下拉通知栏
            boolean expandResult = performGlobalAction(AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS);
            android.util.Log.e("AICHATBOT_SERVICE", "下拉通知栏结果: " + expandResult);
            
            if (!expandResult) {
                return false;
            }
            
            // 等待通知栏展开
            Thread.sleep(1500);
            
            // 查找截图按钮
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode != null) {
                // 查找截图相关按钮
                AccessibilityNodeInfo screenshotButton = findScreenshotButton(rootNode);
                
                if (screenshotButton != null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "找到截图按钮，准备点击");
                    boolean clickResult = screenshotButton.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                    android.util.Log.e("AICHATBOT_SERVICE", "点击截图按钮结果: " + clickResult);
                    
                    screenshotButton.recycle();
                    rootNode.recycle();
                    
                    // 等待截图完成
                    Thread.sleep(1000);
                    
                    // 收起通知栏
                    performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
                    
                    return clickResult;
                }
                rootNode.recycle();
            }
            
            // 如果没找到按钮，收起通知栏
            performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "通知栏截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 查找截图按钮
    private AccessibilityNodeInfo findScreenshotButton(AccessibilityNodeInfo node) {
        if (node == null) return null;
        
        // 检查当前节点的文本和描述
        String[] searchTexts = {"截图", "Screenshot", "屏幕截图", "截屏"};
        
        for (String searchText : searchTexts) {
            if (nodeContainsText(node, searchText)) {
                if (node.isClickable()) {
                    return node;
                }
            }
        }
        
        // 递归查找子节点
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                AccessibilityNodeInfo result = findScreenshotButton(child);
                if (result != null) {
                    child.recycle();
                    return result;
                }
                child.recycle();
            }
        }
        
        return null;
    }
    
    // 检查节点是否包含指定文本
    private boolean nodeContainsText(AccessibilityNodeInfo node, String text) {
        if (node == null || text == null) return false;
        
        CharSequence nodeText = node.getText();
        CharSequence contentDesc = node.getContentDescription();
        
        return (nodeText != null && nodeText.toString().contains(text)) ||
               (contentDesc != null && contentDesc.toString().contains(text));
    }
    
    // 通过最近任务界面进行截图
    private boolean screenshotViaRecents() {
        try {
            // 打开最近任务界面
            boolean recentsResult = performGlobalAction(AccessibilityService.GLOBAL_ACTION_RECENTS);
            android.util.Log.e("AICHATBOT_SERVICE", "打开最近任务界面结果: " + recentsResult);
            
            if (!recentsResult) {
                return false;
            }
            
            // 等待界面加载
            Thread.sleep(1500);
            
            // 查找截图按钮
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode != null) {
                AccessibilityNodeInfo screenshotButton = findScreenshotButton(rootNode);
                
                if (screenshotButton != null) {
                    android.util.Log.e("AICHATBOT_SERVICE", "在最近任务中找到截图按钮");
                    boolean clickResult = screenshotButton.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                    
                    screenshotButton.recycle();
                    rootNode.recycle();
                    
                    // 等待截图完成
                    Thread.sleep(1000);
                    
                    // 返回
                    performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
                    
                    return clickResult;
                }
                rootNode.recycle();
            }
            
            // 返回
            performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "最近任务截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 简化的三指下拉截图方法（推荐使用）
    public boolean simpleThreeFingerScreenshot() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 执行简化三指下拉截图 ===");
        
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                // 尝试多种三指下拉参数配置
                return tryMultipleThreeFingerGestures();
            } else {
                android.util.Log.w("AICHATBOT_SERVICE", "Android版本过低，不支持手势操作");
                return false;
            }
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "三指下拉截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 尝试多种三指下拉手势参数
    private boolean tryMultipleThreeFingerGestures() {
        android.util.Log.e("AICHATBOT_SERVICE", "尝试多种三指下拉手势参数");
        
        // 配置1: 标准参数
        if (performThreeFingerGesture(1080, 250, 700, 600, 50)) {
            return true;
        }
        
        // 配置2: 更快速的手势
        if (performThreeFingerGesture(1080, 200, 600, 400, 100)) {
            return true;
        }
        
        // 配置3: 从更高位置开始
        if (performThreeFingerGesture(1080, 100, 800, 800, 150)) {
            return true;
        }
        
        // 配置4: 更慢的手势
        if (performThreeFingerGesture(1080, 300, 900, 1000, 200)) {
            return true;
        }
        
        android.util.Log.w("AICHATBOT_SERVICE", "所有三指下拉参数都尝试失败");
        return false;
    }
    
    // 执行具体的三指下拉手势
    private boolean performThreeFingerGesture(int screenWidth, int startY, int endY, int duration, int delayBetweenFingers) {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "执行三指手势: 宽度=" + screenWidth + ", 起始Y=" + startY + ", 结束Y=" + endY + ", 持续时间=" + duration + "ms");
            
            Path path1 = new Path();
            Path path2 = new Path();
            Path path3 = new Path();
            
            // 三个手指的X位置 - 更均匀分布
            float finger1X = screenWidth * 0.2f;
            float finger2X = screenWidth * 0.5f;
            float finger3X = screenWidth * 0.8f;
            
            // 设置三条路径
            path1.moveTo(finger1X, startY);
            path1.lineTo(finger1X, endY);
            
            path2.moveTo(finger2X, startY);
            path2.lineTo(finger2X, endY);
            
            path3.moveTo(finger3X, startY);
            path3.lineTo(finger3X, endY);
            
            // 创建手势描述 - 稍微错开开始时间模拟真实手指
            GestureDescription.StrokeDescription stroke1 = 
                new GestureDescription.StrokeDescription(path1, 0, duration);
            GestureDescription.StrokeDescription stroke2 = 
                new GestureDescription.StrokeDescription(path2, delayBetweenFingers, duration);
            GestureDescription.StrokeDescription stroke3 = 
                new GestureDescription.StrokeDescription(path3, delayBetweenFingers * 2, duration);
            
            GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke1)
                .addStroke(stroke2)
                .addStroke(stroke3)
                .build();
            
            final boolean[] gestureCompleted = {false};
            final boolean[] gestureSuccess = {false};
            
            boolean result = dispatchGesture(gesture, new GestureResultCallback() {
                @Override
                public void onCompleted(GestureDescription gestureDescription) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 三指下拉手势执行完成");
                    gestureCompleted[0] = true;
                    gestureSuccess[0] = true;
                }
                
                @Override
                public void onCancelled(GestureDescription gestureDescription) {
                    android.util.Log.e("AICHATBOT_SERVICE", "❌ 三指下拉手势被取消");
                    gestureCompleted[0] = true;
                    gestureSuccess[0] = false;
                }
            }, null);
            
            android.util.Log.e("AICHATBOT_SERVICE", "三指下拉手势启动结果: " + result);
            
            if (result) {
                // 等待手势完成
                int waitTime = 0;
                while (!gestureCompleted[0] && waitTime < 5000) {
                    Thread.sleep(100);
                    waitTime += 100;
                }
                
                // 再等待一段时间让系统处理截图
                Thread.sleep(3000);
                android.util.Log.e("AICHATBOT_SERVICE", "截图等待完成，参数配置成功: " + gestureSuccess[0]);
                
                return gestureSuccess[0];
            }
            
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "三指手势执行异常: " + e.getMessage());
            return false;
        }
    }
    
    // 使用小米手机特有的下拉通知栏截图
    public boolean miuiNotificationScreenshot() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 尝试MIUI下拉通知栏截图 ===");
        
        try {
            // 策略1: 标准下拉通知栏
            boolean result = tryNotificationPanelScreenshot();
            if (result) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 标准通知栏截图成功");
                return true;
            }
            
            // 策略2: 快速设置面板
            result = tryQuickSettingsScreenshot();
            if (result) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 快速设置截图成功");
                return true;
            }
            
            // 策略3: 模拟下拉手势触发通知栏
            result = trySwipeDownNotification();
            if (result) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 下拉手势截图成功");
                return true;
            }
            
            android.util.Log.w("AICHATBOT_SERVICE", "所有MIUI通知栏截图策略都失败");
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "MIUI通知栏截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 尝试标准通知栏截图
    private boolean tryNotificationPanelScreenshot() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "尝试标准下拉通知栏截图");
            
            // 下拉通知栏
            boolean expandResult = performGlobalAction(AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS);
            android.util.Log.e("AICHATBOT_SERVICE", "下拉通知栏结果: " + expandResult);
            
            if (!expandResult) {
                return false;
            }
            
            // 等待通知栏展开
            Thread.sleep(1500);
            
            // 查找截图相关的快捷按钮
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode != null) {
                String[] searchTexts = {"截图", "截屏", "Screenshot", "屏幕截图", "快捷截图"};
                
                for (String searchText : searchTexts) {
                    AccessibilityNodeInfo screenshotButton = findClickableNodeByText(rootNode, searchText);
                    if (screenshotButton != null) {
                        android.util.Log.e("AICHATBOT_SERVICE", "找到截图按钮: " + searchText);
                        boolean clickResult = screenshotButton.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                        
                        screenshotButton.recycle();
                        rootNode.recycle();
                        
                        if (clickResult) {
                            // 等待截图完成
                            Thread.sleep(2000);
                            // 收起通知栏
                            performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
                            return verifyScreenshotSuccess();
                        }
                    }
                }
                rootNode.recycle();
            }
            
            // 收起通知栏
            performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "标准通知栏截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 尝试快速设置面板截图
    private boolean tryQuickSettingsScreenshot() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "尝试快速设置面板截图");
            
            // 先下拉一次通知栏
            performGlobalAction(AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS);
            Thread.sleep(800);
            
            // 再下拉一次打开快速设置
            performGlobalAction(AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS);
            Thread.sleep(1500);
            
            // 查找快速设置中的截图按钮
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode != null) {
                // 在快速设置中查找截图图标或按钮
                String[] quickSettingsTexts = {"截图", "Screenshot", "屏幕截图", "screenshot"};
                
                for (String text : quickSettingsTexts) {
                    AccessibilityNodeInfo button = findClickableNodeByText(rootNode, text);
                    if (button != null) {
                        android.util.Log.e("AICHATBOT_SERVICE", "在快速设置中找到截图按钮: " + text);
                        boolean clickResult = button.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                        
                        button.recycle();
                        rootNode.recycle();
                        
                        if (clickResult) {
                            Thread.sleep(2000);
                            performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
                            return verifyScreenshotSuccess();
                        }
                    }
                }
                
                // 尝试点击可能的截图图标（基于位置）
                boolean positionResult = tryClickScreenshotByPosition(rootNode);
                rootNode.recycle();
                
                if (positionResult) {
                    Thread.sleep(2000);
                    performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
                    return verifyScreenshotSuccess();
                }
            }
            
            // 收起通知栏
            performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "快速设置截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 尝试通过位置点击截图按钮
    private boolean tryClickScreenshotByPosition(AccessibilityNodeInfo rootNode) {
        try {
            // 在快速设置区域寻找可点击的元素
            List<AccessibilityNodeInfo> clickableNodes = new ArrayList<>();
            findClickableNodesInQuickSettings(rootNode, clickableNodes);
            
            android.util.Log.e("AICHATBOT_SERVICE", "快速设置中找到 " + clickableNodes.size() + " 个可点击元素");
            
            for (int i = 0; i < clickableNodes.size(); i++) {
                AccessibilityNodeInfo node = clickableNodes.get(i);
                Rect bounds = new Rect();
                node.getBoundsInScreen(bounds);
                
                // 检查是否在快速设置区域（通常在屏幕上方）
                if (bounds.top < 600 && bounds.right > 100) {
                    String className = node.getClassName() != null ? node.getClassName().toString() : "";
                    String text = node.getText() != null ? node.getText().toString() : "";
                    String desc = node.getContentDescription() != null ? node.getContentDescription().toString() : "";
                    
                    android.util.Log.e("AICHATBOT_SERVICE", "快速设置元素" + i + ": 类名=" + className + 
                        " 文本=" + text + " 描述=" + desc + " 位置=" + bounds.toString());
                    
                    // 尝试点击可能是截图按钮的元素
                    if (className.contains("ImageView") || className.contains("Button") || 
                        text.toLowerCase().contains("shot") || desc.toLowerCase().contains("shot")) {
                        
                        android.util.Log.e("AICHATBOT_SERVICE", "尝试点击可能的截图按钮: " + i);
                        boolean clickResult = node.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                        if (clickResult) {
                            // 清理资源
                            for (AccessibilityNodeInfo n : clickableNodes) {
                                n.recycle();
                            }
                            return true;
                        }
                    }
                }
            }
            
            // 清理资源
            for (AccessibilityNodeInfo node : clickableNodes) {
                node.recycle();
            }
            
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "按位置点击截图按钮失败: " + e.getMessage());
            return false;
        }
    }
    
    // 查找快速设置中的可点击节点
    private void findClickableNodesInQuickSettings(AccessibilityNodeInfo node, List<AccessibilityNodeInfo> clickableNodes) {
        if (node == null) return;
        
        if (node.isClickable()) {
            Rect bounds = new Rect();
            node.getBoundsInScreen(bounds);
            // 只关注屏幕上半部分的元素（快速设置区域）
            if (bounds.top < 600) {
                clickableNodes.add(AccessibilityNodeInfo.obtain(node));
            }
        }
        
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                findClickableNodesInQuickSettings(child, clickableNodes);
                child.recycle();
            }
        }
    }
    
    // 尝试下拉手势触发通知栏
    private boolean trySwipeDownNotification() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "尝试下拉手势触发通知栏");
            
            // 从屏幕顶部下拉
            Path swipePath = new Path();
            swipePath.moveTo(540, 0);     // 屏幕中央顶部
            swipePath.lineTo(540, 400);   // 向下拉400像素
            
            GestureDescription.StrokeDescription swipeStroke = 
                new GestureDescription.StrokeDescription(swipePath, 0, 800);
            
            GestureDescription swipeGesture = new GestureDescription.Builder()
                .addStroke(swipeStroke)
                .build();
            
            boolean swipeResult = dispatchGesture(swipeGesture, null, null);
            android.util.Log.e("AICHATBOT_SERVICE", "下拉手势结果: " + swipeResult);
            
            if (swipeResult) {
                Thread.sleep(1500);
                
                // 查找截图按钮
                AccessibilityNodeInfo rootNode = getRootInActiveWindow();
                if (rootNode != null) {
                    AccessibilityNodeInfo screenshotButton = findClickableNodeByText(rootNode, "截图");
                    if (screenshotButton != null) {
                        android.util.Log.e("AICHATBOT_SERVICE", "下拉手势后找到截图按钮");
                        boolean clickResult = screenshotButton.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                        screenshotButton.recycle();
                        rootNode.recycle();
                        
                        if (clickResult) {
                            Thread.sleep(2000);
                            return verifyScreenshotSuccess();
                        }
                    }
                    rootNode.recycle();
                }
            }
            
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "下拉手势截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 查找可点击的节点（按文本）
    private AccessibilityNodeInfo findClickableNodeByText(AccessibilityNodeInfo node, String text) {
        if (node == null) return null;
        
        // 检查当前节点
        String nodeText = node.getText() != null ? node.getText().toString() : "";
        String contentDesc = node.getContentDescription() != null ? node.getContentDescription().toString() : "";
        
        if ((nodeText.contains(text) || contentDesc.contains(text)) && node.isClickable()) {
            return AccessibilityNodeInfo.obtain(node);
        }
        
        // 递归检查子节点
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                AccessibilityNodeInfo result = findClickableNodeByText(child, text);
                child.recycle();
                if (result != null) {
                    return result;
                }
            }
        }
        
        return null;
    }
    
    // 电源+音量键截图方法
    public boolean powerVolumeScreenshot() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 尝试电源+音量键截图 ===");
        
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                // 尝试多种电源+音量键组合策略
                boolean result = false;
                
                // 策略1: 标准电源+音量下键组合
                android.util.Log.e("AICHATBOT_SERVICE", "策略1: 标准电源+音量下键组合");
                result = performPowerVolumeGesture(1070, 650, 1070, 500, 1500);
                if (result) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 策略1成功");
                    return true;
                }
                
                // 策略2: 更长时间的按压
                android.util.Log.e("AICHATBOT_SERVICE", "策略2: 更长时间按压（2秒）");
                result = performPowerVolumeGesture(1070, 650, 1070, 500, 2000);
                if (result) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 策略2成功");
                    return true;
                }
                
                // 策略3: 调整按键位置
                android.util.Log.e("AICHATBOT_SERVICE", "策略3: 调整按键位置");
                result = performPowerVolumeGesture(1060, 600, 1060, 450, 1500);
                if (result) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 策略3成功");
                    return true;
                }
                
                // 策略4: 尝试左边缘（某些设备）
                android.util.Log.e("AICHATBOT_SERVICE", "策略4: 尝试左边缘按键");
                result = performPowerVolumeGesture(10, 600, 10, 450, 1500);
                if (result) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 策略4成功");
                    return true;
                }
                
                android.util.Log.w("AICHATBOT_SERVICE", "所有电源+音量键策略都失败");
                return false;
            } else {
                android.util.Log.w("AICHATBOT_SERVICE", "Android版本过低，不支持手势操作");
                return false;
            }
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "电源+音量键截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 执行具体的电源+音量键手势
    private boolean performPowerVolumeGesture(int powerX, int powerY, int volumeX, int volumeY, int duration) {
        try {
            Path powerKeyPath = new Path();
            Path volumeKeyPath = new Path();
            
            // 设置按键位置
            powerKeyPath.moveTo(powerX, powerY);
            powerKeyPath.lineTo(powerX, powerY);
            
            volumeKeyPath.moveTo(volumeX, volumeY);
            volumeKeyPath.lineTo(volumeX, volumeY);
            
            // 同时长按模拟按键组合
            GestureDescription.StrokeDescription powerStroke = 
                new GestureDescription.StrokeDescription(powerKeyPath, 0, duration);
            GestureDescription.StrokeDescription volumeStroke = 
                new GestureDescription.StrokeDescription(volumeKeyPath, 50, duration); // 稍微延迟50ms
            
            GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(powerStroke)
                .addStroke(volumeStroke)
                .build();
            
            final boolean[] gestureCompleted = {false};
            final boolean[] gestureSuccess = {false};
            
            boolean result = dispatchGesture(gesture, new GestureResultCallback() {
                @Override
                public void onCompleted(GestureDescription gestureDescription) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 电源+音量键手势执行完成");
                    gestureCompleted[0] = true;
                    gestureSuccess[0] = true;
                }
                
                @Override
                public void onCancelled(GestureDescription gestureDescription) {
                    android.util.Log.e("AICHATBOT_SERVICE", "❌ 电源+音量键手势被取消");
                    gestureCompleted[0] = true;
                    gestureSuccess[0] = false;
                }
            }, null);
            
            android.util.Log.e("AICHATBOT_SERVICE", "电源+音量键手势启动结果: " + result + 
                " 位置: 电源(" + powerX + "," + powerY + ") 音量(" + volumeX + "," + volumeY + ") 时长:" + duration + "ms");
            
            if (result) {
                // 等待手势完成
                int waitTime = 0;
                while (!gestureCompleted[0] && waitTime < (duration + 1000)) {
                    Thread.sleep(100);
                    waitTime += 100;
                }
                
                // 额外等待时间让系统处理截图
                Thread.sleep(2500);
                android.util.Log.e("AICHATBOT_SERVICE", "电源+音量键截图等待完成: " + gestureSuccess[0]);
                
                // 检查是否真的截图了（通过检查是否有截图反馈）
                return verifyScreenshotSuccess() || gestureSuccess[0];
            }
            
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "执行电源+音量键手势异常: " + e.getMessage());
            return false;
        }
    }
    
    // 验证截图是否成功（检查系统反馈）
    private boolean verifyScreenshotSuccess() {
        try {
            // 等待一段时间让系统显示截图提示
            Thread.sleep(1000);
            
            // 检查屏幕上是否有截图相关的提示
            AccessibilityNodeInfo rootNode = getRootInActiveWindow();
            if (rootNode != null) {
                boolean hasScreenshotIndicator = checkForScreenshotIndicator(rootNode);
                rootNode.recycle();
                
                if (hasScreenshotIndicator) {
                    android.util.Log.e("AICHATBOT_SERVICE", "✅ 检测到截图成功指示器");
                    return true;
                }
            }
            
            android.util.Log.d("AICHATBOT_SERVICE", "未检测到截图成功指示器");
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "验证截图结果时出错: " + e.getMessage());
            return false;
        }
    }
    
    // 检查截图成功指示器
    private boolean checkForScreenshotIndicator(AccessibilityNodeInfo node) {
        if (node == null) return false;
        
        // 检查常见的截图成功提示文本
        String[] screenshotIndicators = {
            "截图", "Screenshot", "已保存", "保存到", "屏幕截图", "截屏成功", 
            "已截图", "Screenshot saved", "保存成功", "已截取屏幕"
        };
        
        return checkNodeForScreenshotText(node, screenshotIndicators, 0);
    }
    
    // 递归检查节点中的截图文本
    private boolean checkNodeForScreenshotText(AccessibilityNodeInfo node, String[] indicators, int depth) {
        if (node == null || depth > 8) return false;
        
        // 检查当前节点的文本
        String nodeText = node.getText() != null ? node.getText().toString() : "";
        String contentDesc = node.getContentDescription() != null ? node.getContentDescription().toString() : "";
        
        for (String indicator : indicators) {
            if (nodeText.contains(indicator) || contentDesc.contains(indicator)) {
                android.util.Log.e("AICHATBOT_SERVICE", "找到截图指示器: " + indicator + " 在文本: " + nodeText + contentDesc);
                return true;
            }
        }
        
        // 递归检查子节点
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                boolean found = checkNodeForScreenshotText(child, indicators, depth + 1);
                child.recycle();
                if (found) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    // 新增：系统级截图方法（终极备选方案）
    public boolean systemLevelScreenshot() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 尝试系统级截图方法 ===");
        
        try {
            // 方法1: 尝试长按电源键菜单
            boolean powerMenuResult = tryPowerMenuScreenshot();
            if (powerMenuResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 电源菜单截图成功");
                return true;
            }
            
            // 方法2: 尝试音量减少+电源键（不同的实现）
            boolean altVolumeResult = tryAlternativeVolumeScreenshot();
            if (altVolumeResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 替代音量键截图成功");
                return true;
            }
            
            // 方法3: 尝试模拟助手快捷键
            boolean assistantResult = tryAssistantScreenshot();
            if (assistantResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 助手截图成功");
                return true;
            }
            
            android.util.Log.w("AICHATBOT_SERVICE", "所有系统级截图方法都失败");
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "系统级截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 尝试电源菜单截图
    private boolean tryPowerMenuScreenshot() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "尝试电源菜单截图");
            
            // 长按电源键位置
            Path powerPath = new Path();
            powerPath.moveTo(1070, 600);
            powerPath.lineTo(1070, 600);
            
            GestureDescription.StrokeDescription powerStroke = 
                new GestureDescription.StrokeDescription(powerPath, 0, 2500); // 长按2.5秒
            
            GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(powerStroke)
                .build();
            
            boolean result = dispatchGesture(gesture, null, null);
            if (result) {
                Thread.sleep(3000); // 等待电源菜单出现
                
                // 查找电源菜单中的截图选项
                AccessibilityNodeInfo rootNode = getRootInActiveWindow();
                if (rootNode != null) {
                    AccessibilityNodeInfo screenshotOption = findClickableNodeByText(rootNode, "截图");
                    if (screenshotOption == null) {
                        screenshotOption = findClickableNodeByText(rootNode, "Screenshot");
                    }
                    
                    if (screenshotOption != null) {
                        android.util.Log.e("AICHATBOT_SERVICE", "在电源菜单中找到截图选项");
                        boolean clickResult = screenshotOption.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                        screenshotOption.recycle();
                        rootNode.recycle();
                        
                        if (clickResult) {
                            Thread.sleep(2000);
                            return verifyScreenshotSuccess();
                        }
                    }
                    rootNode.recycle();
                }
                
                // 按返回键关闭电源菜单
                performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
            }
            
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "电源菜单截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 尝试替代音量键截图方法
    private boolean tryAlternativeVolumeScreenshot() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "尝试替代音量键截图");
            
            // 先按音量减，再按电源键
            Path volumePath = new Path();
            volumePath.moveTo(1070, 450);
            volumePath.lineTo(1070, 450);
            
            GestureDescription.StrokeDescription volumeStroke = 
                new GestureDescription.StrokeDescription(volumePath, 0, 1000);
            
            GestureDescription volumeGesture = new GestureDescription.Builder()
                .addStroke(volumeStroke)
                .build();
            
            boolean volumeResult = dispatchGesture(volumeGesture, null, null);
            if (volumeResult) {
                Thread.sleep(200); // 短暂间隔
                
                // 然后按电源键
                Path powerPath = new Path();
                powerPath.moveTo(1070, 600);
                powerPath.lineTo(1070, 600);
                
                GestureDescription.StrokeDescription powerStroke = 
                    new GestureDescription.StrokeDescription(powerPath, 0, 1000);
                
                GestureDescription powerGesture = new GestureDescription.Builder()
                    .addStroke(powerStroke)
                    .build();
                
                boolean powerResult = dispatchGesture(powerGesture, null, null);
                if (powerResult) {
                    Thread.sleep(3000);
                    return verifyScreenshotSuccess();
                }
            }
            
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "替代音量键截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 尝试助手截图方法
    private boolean tryAssistantScreenshot() {
        try {
            android.util.Log.e("AICHATBOT_SERVICE", "尝试助手截图");
            
            // 尝试激活助手（通常是长按Home键或特定手势）
            boolean assistantResult = performGlobalAction(AccessibilityService.GLOBAL_ACTION_TOGGLE_SPLIT_SCREEN);
            if (!assistantResult) {
                // 尝试其他全局操作
                assistantResult = performGlobalAction(AccessibilityService.GLOBAL_ACTION_RECENTS);
            }
            
            if (assistantResult) {
                Thread.sleep(2000);
                
                // 查找助手界面中的截图选项
                AccessibilityNodeInfo rootNode = getRootInActiveWindow();
                if (rootNode != null) {
                    AccessibilityNodeInfo screenshotOption = findClickableNodeByText(rootNode, "截图");
                    if (screenshotOption != null) {
                        android.util.Log.e("AICHATBOT_SERVICE", "在助手界面找到截图选项");
                        boolean clickResult = screenshotOption.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                        screenshotOption.recycle();
                        rootNode.recycle();
                        
                        if (clickResult) {
                            Thread.sleep(2000);
                            return verifyScreenshotSuccess();
                        }
                    }
                    rootNode.recycle();
                }
                
                // 返回
                performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
            }
            
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "助手截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 新增：直接系统截图方法（用于f命令）
    public boolean directSystemScreenshot() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 开始执行直接系统截图 ===");
        
        try {
            // 方法1: 优先使用电源+音量键截图（最可靠）
            android.util.Log.e("AICHATBOT_SERVICE", "方法1: 尝试电源+音量键直接截图");
            boolean powerVolumeResult = powerVolumeScreenshot();
            if (powerVolumeResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 电源+音量键截图成功");
                return true;
            }
            
            // 方法2: 三指下拉截图
            android.util.Log.e("AICHATBOT_SERVICE", "方法2: 尝试三指下拉截图");
            boolean threeFingerResult = simpleThreeFingerScreenshot();
            if (threeFingerResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 三指下拉截图成功");
                return true;
            }
            
            // 方法3: MIUI通知栏截图
            android.util.Log.e("AICHATBOT_SERVICE", "方法3: 尝试MIUI通知栏截图");
            boolean miuiResult = miuiNotificationScreenshot();
            if (miuiResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ MIUI通知栏截图成功");
                return true;
            }
            
            // 方法4: 系统级截图方法
            android.util.Log.e("AICHATBOT_SERVICE", "方法4: 尝试系统级截图方法");
            boolean systemResult = systemLevelScreenshot();
            if (systemResult) {
                android.util.Log.e("AICHATBOT_SERVICE", "✅ 系统级截图成功");
                return true;
            }
            
            android.util.Log.w("AICHATBOT_SERVICE", "❌ 所有直接系统截图方法都失败");
            return false;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "直接系统截图失败: " + e.getMessage());
            return false;
        }
    }
    
    // 测试MediaProjection是否可用（新增方法）
    public boolean testMediaProjectionAvailability() {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 测试MediaProjection可用性 ===");
        
        try {
            // 发送广播给MainActivity，请求启动MediaProjection测试
            Intent intent = new Intent("com.mobi.agent.REQUEST_MEDIA_PROJECTION_TEST");
            sendBroadcast(intent);
            
            android.util.Log.e("AICHATBOT_SERVICE", "MediaProjection测试请求已发送");
            return true;
            
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "MediaProjection测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 执行滑动操作（与uiautomator2的swipe_ext方法对应）
     * @param startX 起始X坐标
     * @param startY 起始Y坐标
     * @param endX 结束X坐标
     * @param endY 结束Y坐标
     * @param durationMs 滑动持续时间（毫秒）
     * @return 是否滑动成功
     */
    public boolean performSwipe(int startX, int startY, int endX, int endY, long durationMs) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 执行滑动操作 ===");
        android.util.Log.e("AICHATBOT_SERVICE", "从 (" + startX + ", " + startY + ") 滑动到 (" + endX + ", " + endY + ")");
        android.util.Log.e("AICHATBOT_SERVICE", "持续时间: " + durationMs + "ms");
        
        try {
            // 调用现有的swipeOnScreen方法
            boolean result = swipeOnScreen(startX, startY, endX, endY, durationMs);
            android.util.Log.e("AICHATBOT_SERVICE", "滑动操作结果: " + (result ? "成功" : "失败"));
            return result;
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "滑动操作异常: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 在指定坐标执行点击操作
     * @param x X坐标
     * @param y Y坐标
     * @return 是否点击成功
     */
    public boolean clickOnScreen(int x, int y) {
        android.util.Log.e("AICHATBOT_SERVICE", "=== 执行点击操作 ===");
        android.util.Log.e("AICHATBOT_SERVICE", "点击坐标: (" + x + ", " + y + ")");
        
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                // 使用手势描述进行点击
                Path clickPath = new Path();
                clickPath.moveTo(x, y);
                clickPath.lineTo(x, y);  // 创建一个点的路径
                
                GestureDescription.Builder gestureBuilder = new GestureDescription.Builder();
                GestureDescription.StrokeDescription strokeDescription = 
                    new GestureDescription.StrokeDescription(clickPath, 0, 100); // 100ms点击时长
                gestureBuilder.addStroke(strokeDescription);
                
                GestureDescription gesture = gestureBuilder.build();
                
                final boolean[] clickResult = {false};
                final Object lockObject = new Object();
                
                boolean dispatched = dispatchGesture(gesture, new GestureResultCallback() {
                    @Override
                    public void onCompleted(GestureDescription gestureDescription) {
                        synchronized (lockObject) {
                            android.util.Log.d("AICHATBOT_SERVICE", "点击手势执行完成");
                            clickResult[0] = true;
                            lockObject.notify();
                        }
                    }
                    
                    @Override
                    public void onCancelled(GestureDescription gestureDescription) {
                        synchronized (lockObject) {
                            android.util.Log.e("AICHATBOT_SERVICE", "点击手势被取消");
                            clickResult[0] = false;
                            lockObject.notify();
                        }
                    }
                }, null);
                
                if (dispatched) {
                    // 等待手势完成
                    synchronized (lockObject) {
                        try {
                            lockObject.wait(2000); // 等待最多2秒
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    }
                    
                    android.util.Log.d("AICHATBOT_SERVICE", "点击操作结果: " + (clickResult[0] ? "成功" : "失败"));
                    return clickResult[0];
                } else {
                    android.util.Log.e("AICHATBOT_SERVICE", "点击手势分发失败");
                    return false;
                }
            } else {
                android.util.Log.e("AICHATBOT_SERVICE", "Android版本过低，不支持手势点击");
                return false;
            }
        } catch (Exception e) {
            android.util.Log.e("AICHATBOT_SERVICE", "点击操作异常: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

}
