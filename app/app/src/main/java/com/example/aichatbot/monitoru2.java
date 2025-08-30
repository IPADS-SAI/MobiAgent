package com.mobi.agent;

import android.accessibilityservice.AccessibilityService;
import android.app.ActivityManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Base64;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import java.io.ByteArrayOutputStream;
import java.util.List;

/**
 * monitoru2 - 精简版
 * 只包含六个接口函数及其必要依赖：
 * - appStart: 启动指定包名的应用
 * - appWait: 等待应用启动完成
 * - u2Screenshot: 截图并保存到指定路径
 * - u2Click: 在指定坐标点击
 * - u2Input: 输入文本
 * - u2SwipeExt: 滑动操作
 */
public class monitoru2 extends AppCompatActivity {
    
    // 必要的依赖变量
    private SharedPreferences prefs;
    private static final String PREF_NAME = "MediaProjectionPrefs";
    private static final String KEY_MEDIA_PROJECTION_AUTHORIZED = "media_projection_authorized";
    private ScreenCaptureService screenCaptureService;
    private boolean serviceBound = false;
    
    // 截图相关常量
    private static final String SCREENSHOT_PATH = "/storage/emulated/0/screenshot.png";
    private static final double FACTOR = 0.5; // 缩放因子
    
    // Service connection for ScreenCaptureService
    private ServiceConnection serviceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            Log.d("monitoru2", "截图服务连接成功");
            ScreenCaptureService.ScreenCaptureBinder binder = (ScreenCaptureService.ScreenCaptureBinder) service;
            screenCaptureService = binder.getService();
            serviceBound = true;
        }
        
        @Override
        public void onServiceDisconnected(ComponentName name) {
            Log.d("monitoru2", "截图服务连接断开");
            screenCaptureService = null;
            serviceBound = false;
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // 初始化SharedPreferences
        prefs = getSharedPreferences(PREF_NAME, MODE_PRIVATE);
        
        // 绑定截图服务
        bindScreenCaptureService();
    }
    
    /**
     * 绑定截图服务
     */
    private void bindScreenCaptureService() {
        try {
            Intent serviceIntent = new Intent(this, ScreenCaptureService.class);
            boolean bindResult = bindService(serviceIntent, serviceConnection, BIND_AUTO_CREATE);
            Log.d("monitoru2", "截图服务绑定结果: " + bindResult);
        } catch (Exception e) {
            Log.e("monitoru2", "绑定截图服务失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        // 解绑截图服务
        if (serviceBound) {
            try {
                unbindService(serviceConnection);
                serviceBound = false;
                Log.d("monitoru2", "截图服务已解绑");
            } catch (Exception e) {
                Log.e("monitoru2", "解绑截图服务失败: " + e.getMessage());
            }
        }
    }
    
    // ========== 六个接口函数 ==========
    
    /**
     * 检查应用是否已安装
     * @param context 上下文对象
     * @param packageName 应用包名
     * @return 是否已安装
     */
    public static boolean isAppInstalled(Context context, String packageName) {
        try {
            PackageManager packageManager = context.getPackageManager();
            // 使用更简单的方式检查应用是否存在
            packageManager.getApplicationInfo(packageName, 0);
            Log.d("monitoru2", "应用已安装: " + packageName);
            return true;
        } catch (PackageManager.NameNotFoundException e) {
            Log.w("monitoru2", "应用未找到: " + packageName);
            return false;
        } catch (Exception e) {
            Log.e("monitoru2", "检查应用安装状态时出错: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 启动指定包名的应用（增强版 - 处理特殊应用）
     * @param context 上下文对象
     * @param packageName 应用包名
     * @throws Exception 启动失败时抛出异常，成功时返回成功信息
     */
    public static void appStart(Context context, String packageName) throws Exception {
        Log.d("monitoru2", "增强版启动应用: " + packageName);
        
        try {
            PackageManager packageManager = context.getPackageManager();
            
            // 方法2: 直接组件启动（模拟 ADB 方式）
            try {
                String activityName = getMainActivityName(packageName);
                if (activityName != null) {
                    Intent intent = new Intent(Intent.ACTION_MAIN);
                    intent.addCategory(Intent.CATEGORY_LAUNCHER);
                    intent.setComponent(new ComponentName(packageName, activityName));
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_NEW_TASK);
                    context.startActivity(intent);
                    Log.d("monitoru2", "SUCCESS: 直接组件启动成功 - " + packageName);
                    throw new Exception("SUCCESS: 应用启动成功 - " + packageName);
                }
            } catch (Exception e) {
                if (e.getMessage() != null && e.getMessage().startsWith("SUCCESS:")) {
                    throw e;
                }
                Log.d("monitoru2", "直接组件启动失败: " + e.getMessage());
            }
            
            

            // 所有方法都失败
            throw new Exception("❌ 所有启动方式都失败，应用可能不支持外部启动: " + packageName);
            
        } catch (Exception e) {
            if (e.getMessage() != null && e.getMessage().startsWith("SUCCESS:")) {
                throw e;
            }
            
            Log.e("monitoru2", "启动应用失败: " + packageName + ", 错误: " + e.getMessage());
            throw new Exception("启动应用失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取应用的主Activity名称
     * @param packageName 包名
     * @return Activity名称，失败返回null
     */
    public static String getMainActivityName(String packageName) {
        //adb shell dumpsys window | findstr mCurrentFocus
        // 常见应用的主Activity映射
        switch (packageName) {
            case "com.tencent.mm":
                return "com.tencent.mm.ui.LauncherUI";
            case "com.tencent.mobileqq":
                return "com.tencent.mobileqq.activity.SplashActivity";
            case "com.sina.weibo":
                return "com.sina.weibo.SplashActivity";
            case "tv.danmaku.bili":
                return "tv.danmaku.bili.MainActivityV2";
            case "com.mihoyo.hyperion":
                return "com.mihoyo.hyperion.splash.SplashActivity";
            case "com.autonavi.minimap":
                return "com.autonavi.map.activity.SplashActivity";
            case "me.ele":
                return "me.ele.Launcher";
            case "com.sankuai.meituan":
                return "com.meituan.android.pt.homepage.activity.MainActivity";
            case "com.qiyi.video":
                return "com.qiyi.video.WelcomeActivity";
            case "com.taobao.taobao":
                return "com.taobao.tao.welcome.Welcome";
            case "com.taobao.trip":
                return "com.alipay.mobile.quinox.LauncherActivity";
            case "com.tongcheng.android":
                return "com.tongcheng.android.LoadingActivity";
            case "com.tencent.qqmusic":
                return "com.tencent.qqmusic.activity.AppStarterActivity";
            case "com.tencent.qqlive":
                return "com.tencent.qqlive.ona.activity.SplashHomeActivity";
            case "com.xingin.xhs":
                return "com.xingin.xhs.index.v2.IndexActivityV2";
            case "com.zhihu.android":
                return "com.zhihu.android.app.ui.activity.LauncherActivity";
            case "com.jingdong.app.mall":
                return "com.jingdong.app.mall.MainFrameActivity";
            case "ctrip.android.view":
                return "ctrip.business.splash.CtripSplashActivity";
            case "com.netease.cloudmusic":
                return "com.netease.cloudmusic.activity.IconChangeDefaultAlias";
            default:
                Log.d("monitoru2", "未知应用包名，无法获取主Activity: " + packageName);
                return null;
        }
    }
    
    /**
     * 等待应用启动完成 - 使用更可靠的检测方法
     * @param context 上下文对象
     * @param packageName 应用包名
     * @param timeout 超时时间（秒）
     * @return 是否启动成功
     */
    public static boolean appWait(Context context, String packageName, int timeout) {
        Log.d("monitoru2", "等待应用启动: " + packageName + ", 超时: " + timeout + "秒");
        
        long startTime = System.currentTimeMillis();
        long timeoutMs = timeout * 1000;
        int checkCount = 0;
        
        while (System.currentTimeMillis() - startTime < timeoutMs) {
            try {
                checkCount++;
                Log.d("monitoru2", "第 " + checkCount + " 次检查应用状态: " + packageName);
                
                // 方法1：检查应用进程是否运行（前台检测）
                android.app.ActivityManager am = (android.app.ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
                List<android.app.ActivityManager.RunningAppProcessInfo> runningProcesses = am.getRunningAppProcesses();
                
                boolean processRunning = false;
                if (runningProcesses != null) {
                    for (android.app.ActivityManager.RunningAppProcessInfo processInfo : runningProcesses) {
                        if (packageName.equals(processInfo.processName)) {
                            processRunning = true;
                            if (processInfo.importance == android.app.ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND) {
                                Log.d("monitoru2", "应用进程在前台运行: " + packageName + " (检查了" + checkCount + "次)");
                                return true;
                            }
                            Log.d("monitoru2", "应用进程在运行但非前台，重要性: " + processInfo.importance);
                        }
                    }
                }
                
                // 方法2：尝试检查任务列表（可能需要权限）
                try {
                    List<android.app.ActivityManager.RunningTaskInfo> tasks = am.getRunningTasks(1);
                    if (!tasks.isEmpty()) {
                        String topPackage = tasks.get(0).topActivity.getPackageName();
                        Log.d("monitoru2", "当前前台应用: " + topPackage + ", 目标应用: " + packageName);
                        
                        if (packageName.equals(topPackage)) {
                            Log.d("monitoru2", "应用已启动并在前台 (通过任务检查): " + packageName + " (检查了" + checkCount + "次)");
                            return true;
                        }
                    }
                } catch (SecurityException e) {
                    // getRunningTasks需要特殊权限，忽略这个错误
                    Log.w("monitoru2", "无法获取运行任务列表，权限不足: " + e.getMessage());
                }
                
                // 如果检查超过20次（10秒）
                if (checkCount >= 20) {
                    if (processRunning) {
                        Log.d("monitoru2", "进程在运行且等待时间够长，假设应用已启动: " + packageName);
                        return true;
                    } else {
                        Log.w("monitoru2", "检查次数已达20次但进程未运行: " + packageName);
                    }
                }
                
                // 等待500ms后重试
                Thread.sleep(500);
                
            } catch (Exception e) {
                Log.w("monitoru2", "检查应用状态时出错: " + e.getMessage());
            }
        }
        
        Log.w("monitoru2", "等待应用启动超时: " + packageName + " (检查了" + checkCount + "次)");
        // 超时时返回false，表示检测失败
        return false;
    }
    
    /**
     * 截图并保存到指定路径（静态方法版本）
     * @param context 上下文对象
     * @param path 保存路径
     * @throws Exception 截图失败时抛出异常
     */
    public static void u2Screenshot(Context context, String path) throws Exception {
        Log.d("monitoru2", "静态截图保存到: " + path);
        
        try {
            // 静态方法暂时不支持，需要使用实例方法
            throw new Exception("静态截图方法暂不支持，请使用实例方法或MainActivity中的截图功能");
            
        } catch (Exception e) {
            Log.e("monitoru2", "静态截图失败: " + e.getMessage());
            throw new Exception("截图失败: " + e.getMessage());
        }
    }
    
    /**
     * 截图并保存到指定路径（实例方法版本）
     * @param path 保存路径
     * @throws Exception 截图失败时抛出异常
     */
    public void u2Screenshot(String path) throws Exception {
        Log.d("monitoru2", "截图保存到: " + path);
        
        try {
            // 检查prefs是否已初始化
            if (prefs == null) {
                throw new Exception("SharedPreferences未初始化，请确保调用了onCreate方法");
            }
            
            // 检查是否有已授权的截图权限
            boolean isAlreadyAuthorized = prefs.getBoolean(KEY_MEDIA_PROJECTION_AUTHORIZED, false);
            
            if (!isAlreadyAuthorized) {
                throw new Exception("截图权限未授权，请先授权截图功能");
            }
            
            // 使用已有的截图服务进行同步截图
            if (serviceBound && screenCaptureService != null) {
                // 使用回调方式进行截图
                final Object lockObject = new Object();
                final Exception[] captureException = {null};
                final boolean[] captureCompleted = {false};
                
                screenCaptureService.captureScreen(new ScreenCaptureService.ScreenCaptureCallback() {
                    @Override
                    public void onSuccess(String filePath) {
                        synchronized (lockObject) {
                            Log.d("monitoru2", "截图成功: " + filePath);
                            captureCompleted[0] = true;
                            lockObject.notify();
                        }
                    }

                    @Override
                    public void onError(String error) {
                        synchronized (lockObject) {
                            Log.e("monitoru2", "截图失败: " + error);
                            captureException[0] = new Exception("截图失败: " + error);
                            captureCompleted[0] = true;
                            lockObject.notify();
                        }
                    }
                });
                
                // 等待截图完成
                synchronized (lockObject) {
                    while (!captureCompleted[0]) {
                        try {
                            lockObject.wait(10000); // 最多等待10秒
                            if (!captureCompleted[0]) {
                                throw new Exception("截图操作超时");
                            }
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new Exception("截图操作被中断");
                        }
                    }
                }
                
                // 检查是否有错误
                if (captureException[0] != null) {
                    throw captureException[0];
                }
                
                Log.d("monitoru2", "截图完成: " + path);
            } else {
                throw new Exception("截图服务未就绪");
            }
            
        } catch (Exception e) {
            Log.e("monitoru2", "截图失败: " + e.getMessage());
            throw new Exception("截图失败: " + e.getMessage());
        }
    }
    
    /**
     * 拿到截图同时保存到screenshot_path的路径，同时用base64编码加载，缩放因子factor，返回编码（静态方法版本）
     * @param context 上下文对象
     * @return base64编码的截图字符串
     * @throws Exception 截图失败时抛出异常
     */
    public static String getScreenshot(Context context) throws Exception {
        Log.d("monitoru2", "获取截图并转换为base64编码（静态方法）");
        
        try {
            // 静态方法暂时不支持，需要使用实例方法
            throw new Exception("静态截图base64编码方法暂不支持，请使用MainActivity中的截图功能");
            
        } catch (Exception e) {
            Log.e("monitoru2", "获取截图base64编码失败: " + e.getMessage());
            throw new Exception("获取截图base64编码失败: " + e.getMessage());
        }
    }
    
    /**
     * 在指定坐标点击
     * @param x X坐标
     * @param y Y坐标
     * @throws Exception 点击失败时抛出异常
     */
    public static void u2Click(int x, int y) throws Exception {
        Log.d("monitoru2", "点击坐标: (" + x + ", " + y + ")");
        
        try {
            MyAccessibilityService service = MyAccessibilityService.getInstance();
            if (service != null) {
                Log.d("monitoru2", "无障碍服务实例获取成功，准备执行点击");
                boolean success = service.clickOnScreen((float)x, (float)y);
                if (success) {
                    Log.d("monitoru2", "点击成功: (" + x + ", " + y + ")");
                } else {
                    Log.e("monitoru2", "点击失败: 无障碍服务clickOnScreen返回false");
                    throw new Exception("无障碍服务点击失败");
                }
            } else {
                Log.e("monitoru2", "无障碍服务实例为null，服务可能未启用");
                throw new Exception("无障碍服务未启用");
            }
        } catch (Exception e) {
            Log.e("monitoru2", "点击操作异常: " + e.getMessage());
            throw new Exception("点击失败: " + e.getMessage());
        }
    }
    

    
    /**
     * 输入文本
     * @param text 要输入的文本
     * @throws Exception 输入失败时抛出异常
     */
    public static void u2Input(String text) throws Exception {
        Log.d("monitoru2", "输入文本: " + text);
        
        try {
            MyAccessibilityService service = MyAccessibilityService.getInstance();
            if (service != null) {
                Log.d("monitoru2", "无障碍服务实例获取成功，准备执行输入操作");
                
                // 首先尝试传统的无障碍服务输入（避免系统剪贴板Toast）
                Log.d("monitoru2", "尝试使用传统方法输入文本");
                long startTime = System.currentTimeMillis();
                boolean success = service.inputTextSafely(text);
                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                
                Log.d("monitoru2", "inputTextSafely执行完成，耗时: " + duration + "ms，结果: " + success);
                
                if (success) {
                    Log.d("monitoru2", "传统方法文本输入成功: " + text);
                    return;
                } else {
                    Log.w("monitoru2", "传统方法失败，尝试剪贴板方法");
                }
                
                // 备用方法：使用剪贴板粘贴方式（可能触发系统Toast）
                Log.d("monitoru2", "尝试使用剪贴板粘贴方式输入文本");
                boolean clipboardSuccess = service.inputTextByClipboard(text);
                
                if (clipboardSuccess) {
                    Log.d("monitoru2", "剪贴板粘贴输入成功: " + text);
                } else {
                    Log.e("monitoru2", "所有输入方法都失败了");
                    throw new Exception("文本输入失败：无障碍服务和剪贴板方法都失败");
                }
            } else {
                Log.e("monitoru2", "无障碍服务实例为null，服务可能未启用");
                throw new Exception("无障碍服务未启用");
            }
        } catch (Exception e) {
            Log.e("monitoru2", "输入操作异常: " + e.getMessage());
            throw new Exception("文本输入失败: " + e.getMessage());
        }
        
        Log.d("monitoru2", "u2Input方法执行完成");
    }
    
    /**
     * 滑动操作
     * @param direction 滑动方向 (up/down/left/right)
     * @param scale 滑动比例
     * @throws Exception 滑动失败时抛出异常
     */
    public static void u2SwipeExt(String direction, double scale) throws Exception {
        Log.d("monitoru2", "滑动操作: " + direction + ", 比例: " + scale);
        
        try {
            MyAccessibilityService service = MyAccessibilityService.getInstance();
            if (service != null) {
                Log.d("monitoru2", "无障碍服务实例获取成功，准备执行滑动操作");
                boolean success = false;
                
                switch (direction.toLowerCase()) {
                    case "up":
                        success = service.swipeUp();
                        break;
                    case "down":
                        success = service.swipeDown();
                        break;
                    case "left":
                        success = service.swipeLeft();
                        break;
                    case "right":
                        success = service.swipeRight();
                        break;
                    default:
                        Log.e("monitoru2", "不支持的滑动方向: " + direction);
                        throw new Exception("不支持的滑动方向: " + direction);
                }
                
                if (success) {
                    Log.d("monitoru2", "滑动成功: " + direction);
                } else {
                    Log.e("monitoru2", "滑动失败: 无障碍服务返回false");
                    throw new Exception("无障碍服务滑动失败");
                }
            } else {
                Log.e("monitoru2", "无障碍服务实例为null，服务可能未启用");
                throw new Exception("无障碍服务未启用");
            }
        } catch (Exception e) {
            Log.e("monitoru2", "滑动操作异常: " + e.getMessage());
            throw new Exception("滑动失败: " + e.getMessage());
        }
    }
    
    /**
     * 检查指定应用是否正在运行
     * @param context 上下文对象
     * @param packageName 应用包名
     * @return 是否正在运行
     */
    public static boolean isAppRunning(Context context, String packageName) {
        try {
            android.app.ActivityManager am = (android.app.ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
            List<android.app.ActivityManager.RunningAppProcessInfo> runningProcesses = am.getRunningAppProcesses();
            
            if (runningProcesses != null) {
                for (android.app.ActivityManager.RunningAppProcessInfo processInfo : runningProcesses) {
                    if (packageName.equals(processInfo.processName)) {
                        Log.d("monitoru2", "应用正在运行: " + packageName + ", 重要性: " + processInfo.importance);
                        return true;
                    }
                }
            }
            
            Log.d("monitoru2", "应用未在运行: " + packageName);
            return false;
        } catch (Exception e) {
            Log.e("monitoru2", "检查应用运行状态失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 强制结束指定应用进程
     * @param packageName 要结束的应用包名
     * @throws Exception 操作失败时抛出异常
     */
    public static void appEnd(String packageName) throws Exception {
        Log.d("monitoru2", "使用ActivityManager强制结束应用进程: " + packageName);
        
        try {
            // 需要Context来获取ActivityManager，这里需要从外部传入
            throw new Exception("appEnd(String packageName) 需要Context参数，请使用 appEnd(Context context, String packageName)");
        } catch (Exception e) {
            Log.e("monitoru2", "结束应用进程异常: " + e.getMessage());
            throw new Exception("结束应用进程失败: " + e.getMessage());
        }
    }
    
    /**
     * 强制结束指定应用进程（带Context参数）
     * @param context 上下文对象
     * @param packageName 要结束的应用包名
     * @throws Exception 操作失败时抛出异常
     */
    public static void appEnd(Context context, String packageName) throws Exception {
        Log.d("monitoru2", "使用ActivityManager强制结束应用进程: " + packageName);
        
        try {
            // 获取ActivityManager系统服务
            android.app.ActivityManager manager = (android.app.ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
            
            if (manager != null) {
                // 使用killBackgroundProcesses方法终止指定应用的后台进程
                manager.killBackgroundProcesses(packageName);
                Log.d("monitoru2", "✅ 成功调用killBackgroundProcesses: " + packageName);
                
                // 验证应用是否已被终止
                Thread.sleep(500); // 等待一下让系统处理
                boolean stillRunning = isAppRunning(context, packageName);
                
                if (stillRunning) {
                    Log.w("monitoru2", "⚠️ 应用进程可能仍在运行，可能需要前台权限或应用有保护机制: " + packageName);
                    // 不抛异常，因为killBackgroundProcesses调用成功了
                } else {
                    Log.d("monitoru2", "✅ 应用进程已成功终止: " + packageName);
                }
                
            } else {
                Log.e("monitoru2", "❌ 无法获取ActivityManager服务");
                throw new Exception("无法获取ActivityManager服务");
            }
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.e("monitoru2", "等待过程被中断: " + e.getMessage());
            throw new Exception("操作被中断: " + e.getMessage());
        } catch (Exception e) {
            Log.e("monitoru2", "结束应用进程异常: " + e.getMessage());
            throw new Exception("结束应用进程失败: " + e.getMessage());
        }
    }

    /**
     * 强制结束当前应用进程
     * @throws Exception 操作失败时抛出异常
     */
    public static void appEnd() throws Exception {
        Log.d("monitoru2", "强制结束当前应用进程");
        
        try {
            MyAccessibilityService service = MyAccessibilityService.getInstance();
            if (service != null) {
                Log.d("monitoru2", "无障碍服务实例获取成功，准备强制关闭当前应用");
                boolean success = service.forceCloseCurrentApp();
                if (success) {
                    Log.d("monitoru2", "强制关闭当前应用成功");
                } else {
                    Log.e("monitoru2", "强制关闭当前应用失败");
                    throw new Exception("无法强制关闭当前应用");
                }
            } else {
                Log.e("monitoru2", "无障碍服务实例为null，服务可能未启用");
                throw new Exception("无障碍服务未启用");
            }
        } catch (Exception e) {
            Log.e("monitoru2", "结束应用进程异常: " + e.getMessage());
            throw new Exception("结束应用进程失败: " + e.getMessage());
        }
    }
}
