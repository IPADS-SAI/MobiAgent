package com.mobi.agent;

import android.content.Context;
import android.widget.Toast;
import android.util.Log;

/**
 * 通用工具类 - 统一处理常用功能，消除代码重复
 */
public class CommonUtils {
    
    /**
     * 统一的Toast显示方法
     * @param context 上下文
     * @param message 消息内容
     * @param duration Toast显示时长
     */
    public static void showToast(Context context, String message, int duration) {
        if (context != null && message != null) {
            Toast.makeText(context, message, duration).show();
        }
    }
    
    /**
     * 显示短时Toast
     * @param context 上下文
     * @param message 消息内容
     */
    public static void showShortToast(Context context, String message) {
        showToast(context, message, Toast.LENGTH_SHORT);
    }
    
    /**
     * 显示长时Toast
     * @param context 上下文
     * @param message 消息内容
     */
    public static void showLongToast(Context context, String message) {
        showToast(context, message, Toast.LENGTH_LONG);
    }
    
    /**
     * 统一的日志打印方法
     * @param tag 日志标签
     * @param message 日志消息
     * @param level 日志级别 (d, i, w, e)
     */
    public static void log(String tag, String message, String level) {
        if (tag == null || message == null) return;
        
        switch (level.toLowerCase()) {
            case "d":
                Log.d(tag, message);
                break;
            case "i":
                Log.i(tag, message);
                break;
            case "w":
                Log.w(tag, message);
                break;
            case "e":
                Log.e(tag, message);
                break;
            default:
                Log.d(tag, message);
        }
    }
    
    /**
     * Debug日志
     */
    public static void logd(String tag, String message) {
        log(tag, message, "d");
    }
    
    /**
     * Info日志
     */
    public static void logi(String tag, String message) {
        log(tag, message, "i");
    }
    
    /**
     * Warning日志
     */
    public static void logw(String tag, String message) {
        log(tag, message, "w");
    }
    
    /**
     * Error日志
     */
    public static void loge(String tag, String message) {
        log(tag, message, "e");
    }
    
    /**
     * 安全的线程睡眠方法
     * @param milliseconds 睡眠时间（毫秒）
     */
    public static void safeSleep(long milliseconds) {
        try {
            Thread.sleep(milliseconds);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            loge("CommonUtils", "睡眠被中断: " + e.getMessage());
        }
    }
    
    /**
     * 检查字符串是否为空或null
     * @param str 要检查的字符串
     * @return 如果为空或null返回true
     */
    public static boolean isEmpty(String str) {
        return str == null || str.trim().isEmpty();
    }
    
    /**
     * 安全的字符串转换
     * @param obj 要转换的对象
     * @return 字符串表示，如果为null返回"unknown"
     */
    public static String safeToString(Object obj) {
        return obj != null ? obj.toString() : "unknown";
    }
    
    /**
     * 格式化坐标字符串
     * @param x x坐标
     * @param y y坐标
     * @return 格式化的坐标字符串
     */
    public static String formatCoordinates(float x, float y) {
        return "(" + x + ", " + y + ")";
    }
    
    /**
     * 格式化边界字符串
     * @param x1 左上角x
     * @param y1 左上角y
     * @param x2 右下角x
     * @param y2 右下角y
     * @return 格式化的边界字符串
     */
    public static String formatBounds(int x1, int y1, int x2, int y2) {
        return "[" + x1 + ", " + y1 + ", " + x2 + ", " + y2 + "]";
    }
}
