package com.mobi.agent;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.media.projection.MediaProjectionManager;
import android.os.IBinder;
import android.util.Log;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

/**
 * 屏幕截图管理器
 * 负责MediaProjection权限管理和ScreenCaptureService的协调
 */
public class ScreenshotManager {
    private static final String TAG = "ScreenshotManager";
    
    private Context context;
    private MediaProjectionManager mediaProjectionManager;
    private ActivityResultLauncher<Intent> screenCaptureLauncher;
    
    // 服务连接相关
    private ScreenCaptureService screenCaptureService;
    private boolean serviceBound = false;
    private ScreenshotCallback pendingCallback;

    /**
     * 截图回调接口
     */
    public interface ScreenshotCallback {
        void onSuccess(String filePath);
        void onError(String error);
        void onPermissionRequired();
    }

    /**
     * 服务连接回调
     */
    private ServiceConnection serviceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName className, IBinder service) {
            Log.d(TAG, "ScreenCaptureService连接成功");
            ScreenCaptureService.ScreenCaptureBinder binder = 
                (ScreenCaptureService.ScreenCaptureBinder) service;
            screenCaptureService = binder.getService();
            serviceBound = true;
            
            // 如果有待处理的截图请求，立即执行
            if (pendingCallback != null) {
                Log.d(TAG, "执行待处理的截图请求");
                executeScreenshot(pendingCallback);
                pendingCallback = null;
            }
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            Log.d(TAG, "ScreenCaptureService连接断开");
            serviceBound = false;
            screenCaptureService = null;
        }
    };

    /**
     * 构造函数
     */
    public ScreenshotManager(AppCompatActivity activity) {
        this.context = activity;
        this.mediaProjectionManager = 
            (MediaProjectionManager) context.getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        
        // 注册权限请求结果回调
        screenCaptureLauncher = activity.registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    Log.d(TAG, "用户授权MediaProjection权限成功");
                    startScreenCaptureService(result.getResultCode(), result.getData());
                } else {
                    Log.w(TAG, "用户拒绝MediaProjection权限");
                    if (pendingCallback != null) {
                        pendingCallback.onError("用户拒绝屏幕截图权限");
                        pendingCallback = null;
                    }
                }
            });
    }

    /**
     * 请求屏幕截图权限
     */
    public void requestScreenCapturePermission() {
        Log.d(TAG, "请求MediaProjection屏幕截图权限");
        Intent captureIntent = mediaProjectionManager.createScreenCaptureIntent();
        screenCaptureLauncher.launch(captureIntent);
    }

    /**
     * 启动屏幕截图服务
     */
    private void startScreenCaptureService(int resultCode, Intent data) {
        Log.d(TAG, "启动ScreenCaptureService前台服务");
        
        // 创建服务Intent并传递权限数据
        Intent serviceIntent = new Intent(context, ScreenCaptureService.class);
        serviceIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_CODE, resultCode);
        serviceIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_DATA, data);
        
        // 启动前台服务
        context.startForegroundService(serviceIntent);
        
        // 如果之前已绑定，先解绑
        if (serviceBound) {
            try {
                context.unbindService(serviceConnection);
                serviceBound = false;
            } catch (Exception e) {
                Log.w(TAG, "解绑旧服务连接失败: " + e.getMessage());
            }
        }
        
        // 绑定到服务以便后续通信
        bindToScreenCaptureService();
    }
    
    /**
     * 绑定到屏幕截图服务
     */
    private void bindToScreenCaptureService() {
        Log.d(TAG, "绑定到ScreenCaptureService");
        Intent serviceIntent = new Intent(context, ScreenCaptureService.class);
        boolean bindResult = context.bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE);
        
        if (!bindResult) {
            Log.e(TAG, "绑定ScreenCaptureService失败");
        }
    }

    /**
     * 执行截图
     */
    public void takeScreenshot(ScreenshotCallback callback) {
        Log.d(TAG, "请求执行截图");
        
        // 检查服务连接状态
        if (!serviceBound || screenCaptureService == null) {
            Log.w(TAG, "ScreenCaptureService未连接，需要先请求权限");
            this.pendingCallback = callback;
            callback.onPermissionRequired();
            return;
        }
        
        // 检查服务是否准备就绪
        if (!screenCaptureService.isReady()) {
            Log.w(TAG, "ScreenCaptureService未准备就绪，权限可能已过期");
            this.pendingCallback = callback;
            callback.onPermissionRequired();
            return;
        }

        // 执行截图
        executeScreenshot(callback);
    }
    
    /**
     * 实际执行截图操作
     */
    private void executeScreenshot(ScreenshotCallback callback) {
        Log.d(TAG, "开始执行截图操作");
        
        screenCaptureService.captureScreen(new ScreenCaptureService.ScreenCaptureCallback() {
            @Override
            public void onSuccess(String filePath) {
                Log.d(TAG, "截图成功，文件路径: " + filePath);
                callback.onSuccess(filePath);
            }

            @Override
            public void onError(String error) {
                Log.e(TAG, "截图失败: " + error);
                
                // 分析错误类型，判断是否需要重新授权
                if (isPermissionError(error)) {
                    Log.w(TAG, "检测到权限相关错误，先清理MediaProjection状态");
                    // 保存回调以便重新授权后重试
                    pendingCallback = callback;
                    // 强制清理当前的MediaProjection状态
                    forceReauthorization();
                    callback.onPermissionRequired();
                } else {
                    callback.onError(error);
                }
            }
        });
    }
    
    /**
     * 判断是否为权限相关错误
     */
    private boolean isPermissionError(String error) {
        return error.contains("MediaProjection未初始化") || 
               error.contains("re-use the resultData") ||
               error.contains("timed out") ||
               error.contains("token that has timed out") ||
               error.contains("SecurityException");
    }

    /**
     * 检查是否有有效的截图权限
     */
    public boolean hasPermission() {
        return serviceBound && 
               screenCaptureService != null && 
               screenCaptureService.isReady();
    }

    /**
     * 强制重新授权（清理当前权限状态）
     */
    public void forceReauthorization() {
        Log.d(TAG, "强制重新授权MediaProjection权限");
        
        // 清理当前服务连接
        if (serviceBound && screenCaptureService != null) {
            try {
                // 停止服务中的MediaProjection
                screenCaptureService.stopProjection();
                
                // 解绑服务
                context.unbindService(serviceConnection);
                serviceBound = false;
                screenCaptureService = null;
                
                // 停止前台服务
                Intent serviceIntent = new Intent(context, ScreenCaptureService.class);
                context.stopService(serviceIntent);
                
            } catch (Exception e) {
                Log.w(TAG, "清理服务状态时出错: " + e.getMessage());
            }
        }
        
        // 重新请求权限
        requestScreenCapturePermission();
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        Log.d(TAG, "清理ScreenshotManager资源");
        
        // 清除待处理的回调
        pendingCallback = null;
        
        // 解绑服务连接
        if (serviceBound) {
            try {
                context.unbindService(serviceConnection);
            } catch (Exception e) {
                Log.w(TAG, "清理时解绑服务失败: " + e.getMessage());
            }
            serviceBound = false;
            screenCaptureService = null;
        }
        
        // 停止前台服务
        try {
            Intent serviceIntent = new Intent(context, ScreenCaptureService.class);
            context.stopService(serviceIntent);
        } catch (Exception e) {
            Log.w(TAG, "清理时停止服务失败: " + e.getMessage());
        }
    }
}
