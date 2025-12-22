package com.mobi.agent;

import android.graphics.Path;
import android.accessibilityservice.GestureDescription;
import android.os.Build;

/**
 * 手势操作工具类 - 统一处理手势相关操作
 */
public class GestureUtils {
    
    private static final String TAG = "GestureUtils";
    
    /**
     * 创建点击手势
     * @param x 点击x坐标
     * @param y 点击y坐标
     * @return GestureDescription对象
     */
    public static GestureDescription createClickGesture(float x, float y) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            Path clickPath = new Path();
            clickPath.moveTo(x, y);
            
            GestureDescription.StrokeDescription stroke = 
                new GestureDescription.StrokeDescription(clickPath, 0, 100);
            
            GestureDescription.Builder builder = new GestureDescription.Builder();
            builder.addStroke(stroke);
            
            return builder.build();
        }
        return null;
    }
    
    /**
     * 创建滑动手势
     * @param startX 起始x坐标
     * @param startY 起始y坐标
     * @param endX 结束x坐标
     * @param endY 结束y坐标
     * @param duration 手势持续时间
     * @return GestureDescription对象
     */
    public static GestureDescription createSwipeGesture(float startX, float startY, 
                                                       float endX, float endY, long duration) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            Path swipePath = new Path();
            swipePath.moveTo(startX, startY);
            swipePath.lineTo(endX, endY);
            
            GestureDescription.StrokeDescription stroke = 
                new GestureDescription.StrokeDescription(swipePath, 0, duration);
            
            GestureDescription.Builder builder = new GestureDescription.Builder();
            builder.addStroke(stroke);
            
            return builder.build();
        }
        return null;
    }
    
    /**
     * 格式化手势操作日志
     * @param operation 操作类型
     * @param startX 起始x坐标
     * @param startY 起始y坐标
     * @param endX 结束x坐标（可选）
     * @param endY 结束y坐标（可选）
     * @return 格式化的日志字符串
     */
    public static String formatGestureLog(String operation, float startX, float startY, Float endX, Float endY) {
        StringBuilder log = new StringBuilder("=== " + operation.toUpperCase() + " ATTEMPT: ");
        log.append(CommonUtils.formatCoordinates(startX, startY));
        
        if (endX != null && endY != null) {
            log.append(" to ").append(CommonUtils.formatCoordinates(endX, endY));
        }
        
        log.append(" ===");
        return log.toString();
    }
    
    /**
     * 创建手势结果回调
     * @param operation 操作名称
     * @param x 坐标x（用于日志）
     * @param y 坐标y（用于日志）
     * @return GestureResultCallback实例
     */
    public static android.accessibilityservice.AccessibilityService.GestureResultCallback 
            createGestureCallback(String operation, float x, float y) {
        
        return new android.accessibilityservice.AccessibilityService.GestureResultCallback() {
            @Override
            public void onCompleted(GestureDescription gestureDescription) {
                super.onCompleted(gestureDescription);
                CommonUtils.loge(TAG, operation + "手势完成: " + CommonUtils.formatCoordinates(x, y));
            }
            
            @Override
            public void onCancelled(GestureDescription gestureDescription) {
                super.onCancelled(gestureDescription);
                CommonUtils.loge(TAG, operation + "手势被取消");
            }
        };
    }
}
