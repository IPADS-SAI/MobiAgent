package com.mobi.agent;

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.WindowManager;
import androidx.core.app.NotificationCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class ScreenCaptureService extends Service {
    private static final String TAG = "ScreenCaptureService";
    private static final String NOTIFICATION_CHANNEL_ID = "ScreenCaptureChannel";
    private static final int NOTIFICATION_ID = 1;
    public static final String ACTION_CAPTURE = "ACTION_CAPTURE";
    public static final String EXTRA_RESULT_CODE = "EXTRA_RESULT_CODE";
    public static final String EXTRA_RESULT_DATA = "EXTRA_RESULT_DATA";

    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;
    private Handler backgroundHandler;
    private HandlerThread backgroundThread;
    private int screenWidth;
    private int screenHeight;
    private int screenDensity;
    
    // 截图状态控制
    private volatile boolean isCapturingScreenshot = false; // 控制是否正在进行截图

    // 回调接口
    public interface ScreenCaptureCallback {
        void onSuccess(String filePath);
        void onError(String error);
    }

    private ScreenCaptureCallback callback;

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "ScreenCaptureService onCreate");
        
        // 创建后台线程处理图像数据
        backgroundThread = new HandlerThread("ScreenCaptureBackground");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "ScreenCaptureService onStartCommand");
        
        if (intent != null) {
            if (ACTION_CAPTURE.equals(intent.getAction())) {
                // 如果是截图命令，并且mediaProjection已经初始化
                if (mediaProjection != null) {
                    Log.d(TAG, "执行截图命令");
                    startCapture();
                } else {
                    Log.w(TAG, "MediaProjection未初始化，无法截图");
                }
            } else {
                // 首次启动服务，从Activity获取授权数据
                int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, -1);
                Intent resultData = intent.getParcelableExtra(EXTRA_RESULT_DATA);
                
                if (resultCode == Activity.RESULT_OK && resultData != null) {
                    Log.d(TAG, "初始化MediaProjection");
                    initializeMediaProjection(resultCode, resultData);
                } else {
                    Log.e(TAG, "无效的授权数据");
                    stopSelf();
                }
            }
        }
        
        return START_STICKY; // 保持服务运行
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "屏幕截图服务",
                NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("用于后台屏幕截图");
            
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
    }

    private void startForegroundService() {
        Log.d(TAG, "启动前台服务");
        
        Notification notification = new NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("MobiAgent")
            .setContentText("屏幕截图服务正在运行")
            .setSmallIcon(R.drawable.baseline_rocket_launch_24)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            // Android 14+ 需要指定前台服务类型
            startForeground(NOTIFICATION_ID, notification, 
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
    }

    private void initializeMediaProjection(int resultCode, Intent resultData) {
        try {
            // 启动前台服务
            startForegroundService();
            
            // 获取MediaProjection
            MediaProjectionManager projectionManager = 
                (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
            
            if (projectionManager != null) {
                mediaProjection = projectionManager.getMediaProjection(resultCode, resultData);
                
                if (mediaProjection != null) {
                    Log.d(TAG, "MediaProjection创建成功");
                    
                    // 注册回调，监听会话状态
                    mediaProjection.registerCallback(new MediaProjection.Callback() {
                        @Override
                        public void onStop() {
                            Log.d(TAG, "MediaProjection会话停止");
                            super.onStop();
                            // 清理资源但不停止服务，等待重新授权
                            stopCapture();
                            // 将mediaProjection设为null，表示需要重新授权
                            mediaProjection = null;
                            Log.w(TAG, "MediaProjection已失效，需要重新授权");
                        }
                    }, backgroundHandler);
                    
                    // 获取屏幕信息
                    getScreenInfo();
                } else {
                    Log.e(TAG, "MediaProjection创建失败");
                    stopSelf();
                }
            } else {
                Log.e(TAG, "MediaProjectionManager获取失败");
                stopSelf();
            }
        } catch (Exception e) {
            Log.e(TAG, "初始化MediaProjection失败: " + e.getMessage());
            e.printStackTrace();
            stopSelf();
        }
    }

    private void getScreenInfo() {
        WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
        DisplayMetrics metrics = new DisplayMetrics();
        windowManager.getDefaultDisplay().getRealMetrics(metrics);
        
        screenWidth = metrics.widthPixels;
        screenHeight = metrics.heightPixels;
        screenDensity = metrics.densityDpi;
        
        Log.d(TAG, "屏幕信息 - 宽度: " + screenWidth + ", 高度: " + screenHeight + ", 密度: " + screenDensity);
    }

    private void startCapture() {
        if (mediaProjection == null) {
            Log.e(TAG, "MediaProjection为null，无法开始截图");
            return;
        }

        try {
            Log.d(TAG, "开始截图");
            
            // 创建ImageReader
            imageReader = ImageReader.newInstance(screenWidth, screenHeight, PixelFormat.RGBA_8888, 2);
            
            // 创建VirtualDisplay
            virtualDisplay = mediaProjection.createVirtualDisplay(
                "ScreenCapture",
                screenWidth,
                screenHeight,
                screenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(),
                null,
                backgroundHandler
            );

            if (virtualDisplay != null) {
                Log.d(TAG, "VirtualDisplay创建成功");
                
                // 设置图像监听器 - 只有在请求截图时才处理图像
                imageReader.setOnImageAvailableListener(reader -> {
                    // 检查是否正在进行截图请求
                    if (!isCapturingScreenshot) {
                        // 如果没有截图请求，丢弃图像数据
                        Image image = null;
                        try {
                            image = reader.acquireLatestImage();
                        } catch (Exception e) {
                            // 忽略
                        } finally {
                            if (image != null) {
                                image.close();
                            }
                        }
                        return;
                    }
                    
                    Image image = null;
                    try {
                        image = reader.acquireLatestImage();
                        if (image != null) {
                            Log.d(TAG, "获取到图像，开始处理");
                            processImage(image);
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "处理图像失败: " + e.getMessage());
                        e.printStackTrace();
                        
                        // 重置截图状态
                        isCapturingScreenshot = false;
                        
                        if (callback != null) {
                            callback.onError("处理图像失败: " + e.getMessage());
                        }
                    } finally {
                        if (image != null) {
                            image.close();
                        }
                        // 保持VirtualDisplay活跃状态，支持后续截图
                        Log.d(TAG, "图像处理完成，VirtualDisplay保持活跃");
                    }
                }, backgroundHandler);
            } else {
                Log.e(TAG, "VirtualDisplay创建失败");
            }
        } catch (Exception e) {
            Log.e(TAG, "开始截图失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void processImage(Image image) {
        try {
            Log.d(TAG, "处理图像");
            
            Image.Plane[] planes = image.getPlanes();
            ByteBuffer buffer = planes[0].getBuffer();
            int pixelStride = planes[0].getPixelStride();
            int rowStride = planes[0].getRowStride();
            int rowPadding = rowStride - pixelStride * screenWidth;

            // 创建Bitmap
            Bitmap bitmap = Bitmap.createBitmap(
                screenWidth + rowPadding / pixelStride,
                screenHeight,
                Bitmap.Config.ARGB_8888
            );
            bitmap.copyPixelsFromBuffer(buffer);

            // 裁剪到正确尺寸
            Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, 0, 0, screenWidth, screenHeight);

            // 保存截图
            saveScreenshot(croppedBitmap);

            // 清理资源
            if (bitmap != null) {
                bitmap.recycle();
            }
            if (croppedBitmap != null) {
                croppedBitmap.recycle();
            }
        } catch (Exception e) {
            Log.e(TAG, "处理图像异常: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void saveScreenshot(Bitmap bitmap) {
        try {
            // 生成文件名
            String fileName = "screenshot_" + new SimpleDateFormat("yyyyMMdd_HHmmss", 
                Locale.getDefault()).format(new Date()) + ".png";
            
            // 保存到应用私有目录
            File screenshotsDir = new File(getExternalFilesDir(null), "Screenshots");
            if (!screenshotsDir.exists()) {
                screenshotsDir.mkdirs();
            }
            
            File screenshotFile = new File(screenshotsDir, fileName);
            
            FileOutputStream fos = new FileOutputStream(screenshotFile);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();
            fos.close();
            
            String filePath = screenshotFile.getAbsolutePath();
            Log.d(TAG, "截图已保存: " + filePath);
            
            // 重置截图状态
            isCapturingScreenshot = false;
            
            // 回调成功
            if (callback != null) {
                callback.onSuccess(filePath);
            }
        } catch (Exception e) {
            Log.e(TAG, "保存截图失败: " + e.getMessage());
            e.printStackTrace();
            
            // 重置截图状态
            isCapturingScreenshot = false;
            
            // 回调失败
            if (callback != null) {
                callback.onError("保存截图失败: " + e.getMessage());
            }
        }
    }

    private void stopCapture() {
        boolean hasResourcesToClean = virtualDisplay != null || imageReader != null;
        
        if (hasResourcesToClean) {
            Log.d(TAG, "清理截图资源");
        }
        
        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
    }

    // 公共方法：执行截图
    public void captureScreen(ScreenCaptureCallback callback) {
        Log.d(TAG, "收到截图请求");
        this.callback = callback;
        
        // 检查MediaProjection状态
        if (mediaProjection == null) {
            Log.e(TAG, "MediaProjection未初始化");
            if (callback != null) {
                callback.onError("MediaProjection未初始化，请重新授权");
            }
            return;
        }
        
        try {
            // 设置截图状态
            isCapturingScreenshot = true;
            
            // 检查是否已有活跃的截图会话
            if (virtualDisplay != null && imageReader != null) {
                Log.d(TAG, "复用现有的截图会话");
                // 直接触发截图，不重新创建VirtualDisplay
                captureScreenshot();
            } else {
                Log.d(TAG, "创建新的截图会话");
                startCapture();
            }
        } catch (Exception e) {
            Log.e(TAG, "截图失败: " + e.getMessage());
            e.printStackTrace();
            
            // 重置截图状态
            isCapturingScreenshot = false;
            
            if (callback != null) {
                callback.onError("截图失败: " + e.getMessage());
            }
        }
    }

    /**
     * 直接截图（复用现有的VirtualDisplay）
     */
    private void captureScreenshot() {
        Log.d(TAG, "复用现有VirtualDisplay进行截图");
        
        if (virtualDisplay == null || imageReader == null) {
            Log.e(TAG, "VirtualDisplay或ImageReader未初始化");
            
            // 重置截图状态
            isCapturingScreenshot = false;
            
            if (callback != null) {
                callback.onError("截图组件未初始化");
            }
            return;
        }
        
        // VirtualDisplay会持续向ImageReader发送图像数据
        // 由于我们已经设置了 isCapturingScreenshot = true
        // 下一次收到的图像会被处理并保存
        Log.d(TAG, "等待VirtualDisplay发送下一帧图像");
        
        // 可选：添加超时保护，防止长时间等待
        backgroundHandler.postDelayed(() -> {
            if (isCapturingScreenshot) {
                Log.w(TAG, "截图等待超时，重置状态");
                isCapturingScreenshot = false;
                if (callback != null) {
                    callback.onError("截图等待超时");
                }
            }
        }, 10000); // 增加到10秒超时
    }

    // 检查服务是否准备好
    public boolean isReady() {
        boolean ready = mediaProjection != null;
        Log.d(TAG, "服务准备状态: " + ready);
        return ready;
    }

    /**
     * 停止MediaProjection（用于强制重新授权）
     */
    public void stopProjection() {
        Log.d(TAG, "停止MediaProjection");
        
        // 先停止当前的截图操作
        stopCapture();
        
        // 停止MediaProjection
        if (mediaProjection != null) {
            try {
                mediaProjection.stop();
            } catch (Exception e) {
                Log.w(TAG, "停止MediaProjection时出错: " + e.getMessage());
            }
            mediaProjection = null;
        }
        
        Log.d(TAG, "MediaProjection已停止，权限已清理");
    }

    @Override
    public void onDestroy() {
        Log.d(TAG, "ScreenCaptureService onDestroy");
        
        stopCapture();
        
        if (mediaProjection != null) {
            mediaProjection.stop();
            mediaProjection = null;
        }
        
        if (backgroundThread != null) {
            backgroundThread.quitSafely();
            try {
                backgroundThread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            backgroundThread = null;
            backgroundHandler = null;
        }
        
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return new ScreenCaptureBinder();
    }

    public class ScreenCaptureBinder extends android.os.Binder {
        public ScreenCaptureService getService() {
            return ScreenCaptureService.this;
        }
    }
}
