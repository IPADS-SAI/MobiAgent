/**
 * MobiAgent移动代理主活动类
 * 
 * 功能说明：
 * 1. 提供聊天界面，支持文本输入
 * 2. 从本地JSON文件循环读取预设响应数据
 * 3. 支持复制、分享等功能
 * 4. 使用RecyclerView显示聊天记录
 * 
 * @author MobiAgent-AndroidStudio
 * @version 1.0
 */
package com.mobi.agent;

import android.Manifest;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Rect;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Vibrator;
import android.os.VibrationEffect;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONArray;

import java.io.IOException;
import java.io.InputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.ByteArrayOutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import android.util.Base64;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.core.type.TypeReference;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.core.JsonProcessingException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.ConnectionSpec;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * 主活动类 - AI聊天机器人界面
 * 
 * 该类负责：
 * - 管理聊天界面UI组件
 * - 处理用户输入（文本）
 * - 从JSON文件循环读取预设响应
 * - 提供复制、分享功能
 */
public class MainActivity extends AppCompatActivity {

    // ==================== UI组件定义 ====================
    RecyclerView recyclerView;              // 聊天记录列表
    TextView welcomeTextView;               // 欢迎文本
    EditText messageEditText;               // 消息输入框
    Button sendButton;                      // 发送按钮
    Button clearButton;                     // 清空聊天按钮
    Button copyButton, shareButton;         // 终止按钮和详情按钮

    // ==================== 数据和适配器 ====================
    List<Message> messageList;              // 聊天消息列表
    MessageAdapter messageAdapter;          // RecyclerView适配器
    JSONArray historyList = new JSONArray();
    String functionName;                     // 当前执行的命令名称
    JSONObject currentItem;                  // 当前执行的命令项（包含reasoning和function）
    JSONObject currentAction;               // 当前执行的动作数据（包含坐标等信息）
    
    // ==================== 进程控制相关 ====================
    private volatile boolean isTaskInterrupted = false;  // 任务中断标志
    private Thread currentApiThread = null;              // 当前API执行线程
    private final List<Thread> activeThreads = new ArrayList<>(); // 活跃线程列表
    private final Object threadLock = new Object();     // 线程列表同步锁
    
    // ==================== 网络和媒体相关 ====================
    public static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    OkHttpClient client = createHttpClient(); // HTTP客户端（用于原始API调用）
    ClipboardManager clipboard;             // 剪贴板管理器
    
    // ==================== 截图相关 ====================
    private ScreenshotManager screenshotManager; // 截图管理器
    private String lastScreenshotPath = null;    // 最新截图的文件路径
    
    // ==================== 常量定义 ====================
    private static final int REQUEST_CODE_SCREENSHOT_PERMISSION = 100; // 截图权限请求码
    private static final int REQUEST_CODE_MEDIA_PROJECTION = 1001; // MediaProjection权限请求码
    
    // 跟踪权限申请次数，避免重复申请
    private static boolean hasRequestedScreenshotPermission = false;
    private static boolean hasRequestedMediaProjectionPermission = false; // 跟踪MediaProjection权限申请
    
    // ==================== JSON数据相关 ====================
    private JSONArray reactData;            // 存储从react.json加载的数据
    private JSONArray actionData;           // 存储从actions.json加载的数据

    private static OkHttpClient createHttpClient() {
        try {
            OkHttpClient.Builder builder = new OkHttpClient.Builder();
            
            // 设置超时时间
            builder.connectTimeout(30, TimeUnit.SECONDS);
            builder.readTimeout(30, TimeUnit.SECONDS);
            builder.writeTimeout(30, TimeUnit.SECONDS);
            
            // 添加拦截器来支持明文通信
            builder.addInterceptor(chain -> {
                Request originalRequest = chain.request();
                return chain.proceed(originalRequest);
            });
            
            // 配置连接规范以支持明文通信
            List<ConnectionSpec> connectionSpecs = new ArrayList<>();
            connectionSpecs.add(ConnectionSpec.CLEARTEXT);
            connectionSpecs.add(ConnectionSpec.MODERN_TLS);
            builder.connectionSpecs(connectionSpecs);
            
            Log.d("MainActivity", "创建支持明文通信的OkHttpClient成功");
            return builder.build();
            
        } catch (Exception e) {
            Log.e("MainActivity", "创建OkHttpClient失败: " + e.getMessage());
            e.printStackTrace();
            // 如果创建失败，返回默认的OkHttpClient
            return new OkHttpClient();
        }
    }

    /**
     * 活动创建方法
     * 初始化所有UI组件、设置监听器、配置RecyclerView等
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // 初始化消息列表
        messageList = new ArrayList<>();

        // ==================== 初始化UI组件 ====================
        recyclerView = findViewById(R.id.recycler_view);
        welcomeTextView = findViewById(R.id.welcome_text);
        messageEditText = findViewById(R.id.message_edit_text);
        sendButton = findViewById(R.id.send_btn);
        clearButton = findViewById(R.id.clearButton);
        copyButton = findViewById(R.id.copy_button);
        shareButton = findViewById(R.id.shareButton);

        // ==================== 确保终止按钮显示正确 ====================
        // 明确设置终止按钮的文本和样式，确保没有图标
        copyButton.setText("终止");
        copyButton.setTextColor(android.graphics.Color.WHITE); // 确保文字颜色是白色
        copyButton.setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 14); // 设置字体大小
        copyButton.setGravity(android.view.Gravity.CENTER); // 确保文字居中
        
        // 移除所有图标
        copyButton.setCompoundDrawables(null, null, null, null);
        copyButton.setCompoundDrawablePadding(0);
        
        // 额外的图标清除措施
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.JELLY_BEAN_MR1) {
            copyButton.setCompoundDrawablesRelative(null, null, null, null);
        }
        
        // 确保按钮有足够的空间显示文字
        copyButton.setPadding(8, 0, 8, 0);
        copyButton.setMinWidth(0);
        copyButton.setMinHeight(0);
        
        // 强制刷新按钮显示
        copyButton.invalidate();
        copyButton.requestLayout();
        
        // 延迟确保文字显示（处理可能的UI加载时序问题）
        copyButton.post(() -> {
            copyButton.setText("终止");
            copyButton.setVisibility(android.view.View.VISIBLE);
            Log.d("MainActivity", "延迟设置终止按钮文字完成");
        });
        
        Log.d("MainActivity", "终止按钮已设置：文本='终止'，字体大小=14sp，颜色=白色");

        // ==================== 确保发送按钮显示正确 ====================
        // 明确设置发送按钮的文本和样式，确保没有图标，与终止按钮样式一致
        sendButton.setText("发送");
        sendButton.setTextColor(android.graphics.Color.WHITE); // 确保文字颜色是白色
        sendButton.setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 14); // 设置字体大小
        sendButton.setGravity(android.view.Gravity.CENTER); // 确保文字居中
        
        // 移除所有图标
        sendButton.setCompoundDrawables(null, null, null, null);
        sendButton.setCompoundDrawablePadding(0);
        
        // 额外的图标清除措施
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.JELLY_BEAN_MR1) {
            sendButton.setCompoundDrawablesRelative(null, null, null, null);
        }
        
        // 确保按钮有足够的空间显示文字
        sendButton.setPadding(8, 0, 8, 0);
        sendButton.setMinWidth(0);
        sendButton.setMinHeight(0);
        
        // 强制刷新按钮显示
        sendButton.invalidate();
        sendButton.requestLayout();
        
        // 延迟确保文字显示（处理可能的UI加载时序问题）
        sendButton.post(() -> {
            sendButton.setText("发送");
            sendButton.setVisibility(android.view.View.VISIBLE);
            Log.d("MainActivity", "延迟设置发送按钮文字完成");
        });
        
        Log.d("MainActivity", "发送按钮已设置：文本='发送'，字体大小=14sp，颜色=白色");

        // ==================== 设置随机提示文本 ====================
        setRandomHint();

        // ==================== 设置RecyclerView ====================
        messageAdapter = new MessageAdapter(messageList);
        recyclerView.setAdapter(messageAdapter);
        LinearLayoutManager llm = new LinearLayoutManager(this);
        llm.setStackFromEnd(true);  // 让消息从底部开始显示
        recyclerView.setLayoutManager(llm);

        // 初始化剪贴板管理器
        clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);

        // ==================== 设置按钮监听器 ====================
        
        // 发送按钮监听器
        sendButton.setOnClickListener((v) -> {
            String question = messageEditText.getText().toString().trim();
            
            // 特殊处理done指令 - 立即中断所有操作并重置状态
            if ("done".equalsIgnoreCase(question)) {
                Log.d("MainActivity", "在发送按钮中检测到done指令，开始执行完全重置");
                
                // 添加消息到聊天
                addToChat(question, Message.SENT_BY_ME);
                messageEditText.setText("");
                
                // 执行完全重置
                resetAllStates();
                
                // 立即显示完成消息
                addResponse("🔄 done指令执行完成\n" +
                          "✅ 所有运行中的线程已停止\n" +
                          "✅ 历史记录已清空\n" +
                          "✅ 所有状态已重置\n" +
                          "✅ 下次操作将从头开始", Message.SENT_BY_BOT);
                showExtendedToast("done指令完成，系统已重置");
                
                return; // 不调用callAPI
            }
            
            addToChat(question, Message.SENT_BY_ME);        // 将用户消息添加到聊天
            messageEditText.setText("");                    // 清空输入框
            callAPI(question);                             // 调用API获取回复
            welcomeTextView.setVisibility(View.GONE);      // 隐藏欢迎文本
        });

        // 清空聊天按钮监听器
        clearButton.setOnClickListener((v) -> {
            clearChat();  // 清空聊天记录
        });

        // 设置输入框焦点并显示键盘
        messageEditText.requestFocus();
        showKeyboard();

        // 复制按钮监听器
        // 终止按钮监听器 - 执行done指令效果
        copyButton.setOnClickListener((v) -> {
            Log.d("MainActivity", "终止按钮被点击，开始执行done指令效果");
            
            // 执行完全重置
            resetAllStates();
            
            // 立即显示完成消息
            addResponse("🛑 任务已终止\n" +
                      "✅ 所有运行中的线程已停止\n" +
                      "✅ 历史记录已清空\n" +
                      "✅ 所有状态已重置\n" +
                      "✅ 下次操作将从头开始", Message.SENT_BY_BOT);
            
            // 显示终止完成Toast
            showEnhancedCompletionToast("🛑 任务已终止，系统已重置");
            
            Log.d("MainActivity", "终止按钮处理完成");
        });

        // 分享按钮监听器
        shareButton.setOnClickListener((v) -> {
            // 显示悬浮窗信息而不是Toast
            String welcomeMessage = "您好！欢迎使用MobiAgent\n请确保启动无障碍与通知权限\n无障碍模式通常在设置中开启\n现在支持的应用包括但不限于:\n qq,B站,高德地图,饿了么,淘宝\n知乎,微博,京东,携程,网易云音乐\n小红书,美团,爱奇艺,QQ音乐\n输入样例参见输入框灰色小字\n app可能会出错，请及时接管\n微信键盘输入有问题，请手动输入\n小米手机有概率不支持键盘的正常输入";
            showFloatingWindow(welcomeMessage);
        });

        RelativeLayout bottomLayout = findViewById(R.id.bottom_layout);
        final ViewTreeObserver.OnGlobalLayoutListener globalLayoutListener = new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                Rect r = new Rect();
                bottomLayout.getWindowVisibleDisplayFrame(r);
                int screenHeight = bottomLayout.getRootView().getHeight();

                int keypadHeight = screenHeight - r.bottom;

                if (keypadHeight > screenHeight * 0.15) { // Adjust the threshold as needed
                    clearButton.setVisibility(View.GONE);
                    shareButton.setVisibility(View.GONE);
                } else {
                    clearButton.setVisibility(View.VISIBLE);
                    shareButton.setVisibility(View.VISIBLE);
                }
            }
        };

        bottomLayout.getViewTreeObserver().addOnGlobalLayoutListener(globalLayoutListener);

        // 初始化截图管理器并在首次启动时主动获取权限
        initializeScreenshotManagerAndRequestPermission();
    }

    /**
     * 调用API方法 - 一次性处理整个JSON数据并执行所有命令
     * 
     * @param question 用户输入的问题
     * 
     * 功能说明：
     * 1. 显示"正在执行"提示
     * 2. 遍历整个reactData数组
     * 3. 解析并执行所有function命令
     * 4. 综合显示所有执行结果
     */
    void callAPI(String question) {
        // ========== 首先检查中断标志 ==========
        if (isTaskInterrupted) {
            Log.d("MainActivity", "检测到任务中断标志，不执行callAPI");
            runOnUiThread(() -> {
                Toast.makeText(MainActivity.this, "任务已被中断", Toast.LENGTH_SHORT).show();
            });
            return;
        }
        
        // 显示"正在执行"指示器
        addResponse("正在执行所有命令...", Message.SENT_BY_BOT);

        //String packageName = "com.taobao.trip";

        // 异步执行所有命令
        currentApiThread = new Thread(() -> {
            // 将当前线程添加到活跃线程列表
            addActiveThread(Thread.currentThread());
            
            try {
                // ========== 新任务开始时的中断检查和重置 ==========
                // 对于用户主动发起的新任务，清除之前的中断标志
                if (isTaskInterrupted) {
                    Log.d("MainActivity", "检测到新任务开始，清除之前的中断标志");
                    isTaskInterrupted = false;
                }
                
                // 只检查当前线程的中断状态
                if (Thread.currentThread().isInterrupted()) {
                    Log.d("MainActivity", "当前线程已被中断，停止执行");
                    return;
                }
            
                StringBuilder responseBuilder = new StringBuilder();
                
                int successCount = 0;
                int failureCount = 0;
                boolean flag = false;

                try {
                        // ========== 最早期检查done指令 ==========
                        if ("done".equalsIgnoreCase(question.trim())) {
                            Log.d("MainActivity", "在线程内部检测到done指令，显示完成Toast");
                            
                            // 确保done命令显示Toast
                            runOnUiThread(() -> {
                                try {
                                    showEnhancedCompletionToast("✅ 任务已完成");
                                    Log.d("MainActivity", "早期done命令Toast已显示");
                                } catch (Exception e) {
                                    Log.e("MainActivity", "早期done命令Toast显示失败", e);
                                }
                            });
                            
                            // 让done命令继续正常流转到API处理
                            Log.d("MainActivity", "done命令将继续正常处理");
                        }
                        
                        // ========== 检查sigint中断指令 ==========
                        if ("sigint".equalsIgnoreCase(question.trim())) {
                            Log.d("MainActivity", "检测到sigint指令，执行快速中断");
                            
                            // 立即设置中断标志
                            isTaskInterrupted = true;
                            
                            // 中断所有活跃线程
                            interruptAllActiveThreads();
                            
                            // 中断当前线程（如果存在）
                            if (currentApiThread != null && currentApiThread != Thread.currentThread()) {
                                currentApiThread.interrupt();
                                Log.d("MainActivity", "已中断当前API线程");
                            }
                            
                            // 清空历史记录
                            historyList = new JSONArray();
                            
                            runOnUiThread(() -> {
                                // Hide typing indicator and show success message
                                messageList.remove(messageList.size() - 1);
                                messageAdapter.notifyDataSetChanged();
                                recyclerView.smoothScrollToPosition(messageAdapter.getItemCount());
                                addResponse("⚡ sigint中断执行完成\n✅ 所有线程已中断\n✅ 历史记录已清空", Message.SENT_BY_BOT);
                                
                                // 弹出系统消息
                                showExtendedToast("sigint中断完成");
                            });
                            
                            // 延迟重置中断标志，让其他线程有时间检测到
                            Thread resetThread = new Thread(() -> {
                                try {
                                    Thread.sleep(500); // 缩短到500ms，快速恢复正常操作
                                    isTaskInterrupted = false;
                                    Log.d("MainActivity", "中断标志已在500ms后重置");
                                } catch (InterruptedException e) {
                                    Log.d("MainActivity", "重置中断标志的线程被中断");
                                }
                            });
                            addActiveThread(resetThread);
                            resetThread.start();
                            
                            // 立即退出当前线程，不再继续处理任何操作
                            Log.d("MainActivity", "sigint指令处理完成，立即退出");
                            return;
                        }
                        
                        boolean commandSuccess = false;

                        // 1. 在动作执行前先截图（带重试机制）
                        Log.d("MainActivity", "步骤1: 动作执行前截图");       
                        responseBuilder.append(" 动作前截图");
                        
                        boolean screenshotSuccess = false;
                        int retryCount = 0;
                        int maxRetries = 5;
                        
                        while (!screenshotSuccess && retryCount < maxRetries) {
                            // 检查是否被中断
                            if (isTaskInterrupted || Thread.currentThread().isInterrupted()) {
                                Log.d("MainActivity", "检测到任务中断，停止截图操作");
                                return;
                            }
                            
                            retryCount++;
                            Log.d("MainActivity", "截图尝试第 " + retryCount + " 次");
                            screenshotSuccess = executeDoneCommand(responseBuilder);
                            
                            if (!screenshotSuccess && retryCount < maxRetries) {
                                Log.w("MainActivity", "截图失败，等待2秒后重试");
                                Thread.sleep(2000); // 等待2秒后重试
                            }
                        }
                        
                        Log.d("MainActivity", "步骤1结果: 动作前截图" + (screenshotSuccess ? "成功" : "失败") + " (尝试次数: " + retryCount + ")");

                        Thread.sleep(1500);
                        Log.d("MainActivity", "截图后等待完成");

                        // 打印用户问题
                        Log.d("MainActivity", "当前处理的用户问题: " + question);

                        // 检查是否被中断
                        if (isTaskInterrupted || Thread.currentThread().isInterrupted()) {
                            Log.d("MainActivity", "检测到任务中断，停止Base64编码操作");
                            return;
                        }

                        // 读取截图并转换为Base64编码
                        String screenshotBase64 = null;
                        if (screenshotSuccess && lastScreenshotPath != null) {
                            screenshotBase64 = convertImageToBase64(lastScreenshotPath);
                            if (screenshotBase64 != null) {
                                Log.d("MainActivity", "截图Base64编码成功，准备发送到API");
                                responseBuilder.append(" 📊 Base64编码完成");
                            } else {
                                Log.w("MainActivity", "截图Base64编码失败");
                                responseBuilder.append(" ⚠️ Base64编码失败");
                            }
                        }

                        // 创建history列表，包含最近的消息
                        
                        // 检查是否有有效的截图，如果没有则重试截图
                        if (screenshotBase64 == null || screenshotBase64.trim().isEmpty()) {
                            Log.w("MainActivity", "截图失败，尝试重新截图");
                            
                            // 尝试重新截图（最多3次）
                            boolean retryScreenshotSuccess = false;
                            int retryScreenshotCount = 0;
                            int maxScreenshotRetries = 3;
                            
                            while (!retryScreenshotSuccess && retryScreenshotCount < maxScreenshotRetries) {
                                // 检查是否被中断
                                if (isTaskInterrupted || Thread.currentThread().isInterrupted()) {
                                    Log.d("MainActivity", "检测到任务中断，停止重试截图操作");
                                    return;
                                }
                                
                                retryScreenshotCount++;
                                Log.d("MainActivity", "重新截图尝试第 " + retryScreenshotCount + " 次");
                                
                                // 等待一段时间后重试
                                try {
                                    Thread.sleep(1000); // 等待1秒
                                } catch (InterruptedException e) {
                                    Log.d("MainActivity", "重试截图等待被中断");
                                    return;
                                }
                                
                                // 重新执行截图
                                StringBuilder retryResponseBuilder = new StringBuilder();
                                retryScreenshotSuccess = executeDoneCommand(retryResponseBuilder);
                                
                                if (retryScreenshotSuccess && lastScreenshotPath != null) {
                                    screenshotBase64 = convertImageToBase64(lastScreenshotPath);
                                    if (screenshotBase64 != null && !screenshotBase64.trim().isEmpty()) {
                                        Log.d("MainActivity", "重新截图成功，Base64编码完成");
                                        responseBuilder.append(" 🔄 重新截图成功");
                                        break;
                                    } else {
                                        Log.w("MainActivity", "重新截图的Base64编码失败");
                                        retryScreenshotSuccess = false;
                                    }
                                } else {
                                    Log.w("MainActivity", "重新截图失败: " + retryResponseBuilder.toString());
                                }
                                
                                if (!retryScreenshotSuccess && retryScreenshotCount < maxScreenshotRetries) {
                                    Log.w("MainActivity", "重新截图失败，等待2秒后再次重试");
                                }
                            }
                            
                            // 如果重试后仍然失败，显示错误消息
                            if (!retryScreenshotSuccess || screenshotBase64 == null || screenshotBase64.trim().isEmpty()) {
                                Log.e("MainActivity", "经过" + retryScreenshotCount + "次重试后截图仍然失败，无法发送API请求");
                                
                                // 创建final变量供lambda使用
                                final int finalRetryCount = retryScreenshotCount;
                                
                                runOnUiThread(() -> {
                                    // Hide typing indicator and show error message
                                    messageList.remove(messageList.size() - 1);
                                    messageAdapter.notifyDataSetChanged();
                                    recyclerView.smoothScrollToPosition(messageAdapter.getItemCount());
                                    addResponse("截图失败（已重试" + finalRetryCount + "次），请检查截图权限或重启应用", Message.SENT_BY_BOT);
                                    
                                    // 提示用户检查权限
                                    showExtendedToast("截图功能异常，请检查权限设置");
                                });
                                return; // 直接返回，不发送请求
                            }
                        }

                        

                        // 检查是否被中断
                        if (isTaskInterrupted || Thread.currentThread().isInterrupted()) {
                            Log.d("MainActivity", "检测到任务中断，停止HTTP请求构建");
                            return;
                        }

                        // OkHttp
                        JSONObject jsonBody = new JSONObject();
                        try {
                            jsonBody.put("task", question);
                            jsonBody.put("image", screenshotBase64);
                            jsonBody.put("history", historyList);
                            
                            // 截图已处理完成，删除临时截图文件以节省存储空间
                            if (lastScreenshotPath != null) {
                                try {
                                    File screenshotFile = new File(lastScreenshotPath);
                                    if (screenshotFile.exists()) {
                                        boolean deleted = screenshotFile.delete();
                                        if (deleted) {
                                            Log.d("MainActivity", "✅ 已删除临时截图文件: " + lastScreenshotPath);
                                        } else {
                                            Log.w("MainActivity", "⚠️ 删除临时截图文件失败: " + lastScreenshotPath);
                                        }
                                    } else {
                                        Log.d("MainActivity", "临时截图文件不存在，无需删除: " + lastScreenshotPath);
                                    }
                                } catch (Exception deleteException) {
                                    Log.e("MainActivity", "删除临时截图文件时发生异常: " + deleteException.getMessage());
                                }
                                // 清空路径引用
                                lastScreenshotPath = null;
                            }
                            
                            // 添加调试日志
                            Log.d("MainActivity", "准备发送请求:");
                            Log.d("MainActivity", "URL: http://123.60.91.241:2333/v1");
                            Log.d("MainActivity", "Content-Type: application/json; charset=utf-8");
                            Log.d("MainActivity", "请求体大小: " + jsonBody.toString().length() + " 字符");
                            Log.d("MainActivity", "task字段: " + question);
                            Log.d("MainActivity", "image字段长度: " + (screenshotBase64 != null ? screenshotBase64.length() : "null"));
                            Log.d("MainActivity", "history字段长度: " + historyList.length());
                            Log.d("MainActivity", "完整请求JSON: " + jsonBody.toString(2));
                            
                        } catch (JSONException e) {
                            Log.e("MainActivity", "构建JSON请求体失败", e);
                            e.printStackTrace();
                        }

                        // 在发送网络请求前最后检查一次中断状态
                        if (isTaskInterrupted || Thread.currentThread().isInterrupted()) {
                            Log.d("MainActivity", "网络请求发送前检测到中断，取消请求");
                            return;
                        }

                        RequestBody body = RequestBody.create(jsonBody.toString(), JSON);
                        Request request = new Request.Builder()
                                .url("http://123.60.91.241:2333/v1")
                                .post(body)
                                .addHeader("Content-Type", "application/json; charset=utf-8")
                                .build();

                        client.newCall(request).enqueue(new Callback() {
                            @Override
                            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                                runOnUiThread(() -> {
                                    // 创建失败响应的JSON结构
                                    JSONObject failureResponse = new JSONObject();
                                    try {
                                        failureResponse.put("status", "error");
                                        failureResponse.put("error_type", "network_failure");
                                        failureResponse.put("error_message", e.getMessage());
                                        failureResponse.put("timestamp", System.currentTimeMillis());
                                        
                                        Log.d("MainActivity", "网络请求失败响应JSON: " + failureResponse.toString());
                                    } catch (JSONException jsonException) {
                                        Log.e("MainActivity", "创建失败响应JSON时出错", jsonException);
                                    }
                                    
                                    // Hide typing indicator and show error message
                                    messageList.remove(messageList.size() - 1);
                                    messageAdapter.notifyDataSetChanged();
                                    recyclerView.smoothScrollToPosition(messageAdapter.getItemCount());
                                    addResponse("Failed to load due to " + e.getMessage(), Message.SENT_BY_BOT);
                                });
                            }

                            @Override
                            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                                // 立即检查中断状态，如果被中断则直接返回，不处理响应
                                if (isTaskInterrupted || Thread.currentThread().isInterrupted()) {
                                    Log.d("MainActivity", "响应处理时检测到中断，忽略响应");
                                    if (response.body() != null) {
                                        response.body().close();
                                    }
                                    return;
                                }
                                
                                // 将响应体序列化为JSON
                                JSONObject responseJson = new JSONObject();
                                String result = null;
                                
                                try {
                                    // 先读取响应体内容（确保在finally中关闭）
                                    if (response.body() != null) {
                                        result = response.body().string();
                                    } else {
                                        result = "";
                                    }
                                    
                                    // ========== 响应格式日志打印 ==========
                                    Log.d("MainActivity", "========== 响应详细信息 ==========");
                                    Log.d("MainActivity", "HTTP状态码: " + response.code());
                                    Log.d("MainActivity", "HTTP状态消息: " + response.message());
                                    Log.d("MainActivity", "请求是否成功: " + response.isSuccessful());
                                    Log.d("MainActivity", "原始响应体内容: " + (result != null ? result : "null"));
                                    Log.d("MainActivity", "响应体长度: " + (result != null ? result.length() : 0) + " 字符");
                                    Log.d("MainActivity", "响应体是否为空: " + (result == null || result.trim().isEmpty()));
                                    
                                    // 添加响应基本信息
                                    responseJson.put("status_code", response.code());
                                    responseJson.put("status_message", response.message());
                                    responseJson.put("is_successful", response.isSuccessful());
                                    responseJson.put("timestamp", System.currentTimeMillis());
                                    
                                    // 添加响应头信息
                                    JSONObject headersJson = new JSONObject();
                                    for (String headerName : response.headers().names()) {
                                        headersJson.put(headerName, response.header(headerName));
                                    }
                                    responseJson.put("headers", headersJson);
                                    
                                    // 添加响应体内容
                                    responseJson.put("body_raw", result != null ? result : "");
                                    
                                    if (response.isSuccessful()) {
                                        if (result != null && !result.trim().isEmpty()) {
                                            try {
                                                // 解析新的JSON格式: reasoning, action, parameters
                                                JSONObject bodyJson = new JSONObject(result);
                                                responseJson.put("body_parsed", bodyJson);
                                                responseJson.put("parse_status", "success");
                                                
                                                // ========== 解析成功后的格式日志 ==========
                                                Log.d("MainActivity", "========== JSON解析成功 ==========");
                                                Log.d("MainActivity", "解析后的JSON结构: " + bodyJson.toString(2));
                                                
                                                // 提取AI操作指令的各个字段
                                                String reasoning = bodyJson.optString("reasoning", "");
                                                String action = bodyJson.optString("action", "");
                                                JSONObject parameters = bodyJson.optJSONObject("parameters");
                                                
                                                // ========== 字段提取日志 ==========
                                                Log.d("MainActivity", "提取的字段信息:");
                                                Log.d("MainActivity", "  • reasoning字段: " + (reasoning.isEmpty() ? "空" : "'" + reasoning + "'"));
                                                Log.d("MainActivity", "  • action字段: " + (action.isEmpty() ? "空" : "'" + action + "'"));
                                                Log.d("MainActivity", "  • parameters字段: " + (parameters != null ? parameters.toString() : "null"));
                                                Log.d("MainActivity", "准备执行action: " + action);
                                                
                                                if (parameters != null) {
                                                    Log.d("MainActivity", "parameters详细内容:");
                                                    for (java.util.Iterator<String> keys = parameters.keys(); keys.hasNext(); ) {
                                                        String key = keys.next();
                                                        Object value = parameters.opt(key);
                                                        Log.d("MainActivity", "    - " + key + ": " + value);
                                                    }
                                                }
                                                
                                                // 添加解析后的字段到响应JSON
                                                responseJson.put("ai_reasoning", reasoning);
                                                responseJson.put("ai_action", action);
                                                responseJson.put("ai_parameters", parameters);
                                                
                                                Log.d("MainActivity", "完整响应JSON: " + responseJson.toString(2));
                                                Log.d("MainActivity", "AI推理: " + reasoning);
                                                Log.d("MainActivity", "AI动作: " + action);
                                                Log.d("MainActivity", "AI参数: " + (parameters != null ? parameters.toString() : "null"));
                                                
                                                // ========== 更新全局变量 ==========
                                                // 根据API响应更新全局变量，供后续命令执行使用
                                                try {
                                                    // 更新functionName
                                                    functionName = action;
                                                    
                                                    // 更新currentItem
                                                    currentItem = new JSONObject();
                                                    currentItem.put("reasoning", reasoning);
                                                    JSONObject function = new JSONObject();
                                                    function.put("name", action);
                                                    if (parameters != null) {
                                                        function.put("parameters", parameters);
                                                    } else {
                                                        function.put("parameters", new JSONObject());
                                                    }
                                                    currentItem.put("function", function);
                                                    
                                                    // 更新currentAction
                                                    currentAction = new JSONObject();
                                                    if (parameters != null) {
                                                        // 处理坐标信息
                                                        if (parameters.has("x")) {
                                                            currentAction.put("position_x", 2 * parameters.getInt("x"));
                                                        }
                                                        if (parameters.has("y")) {
                                                            currentAction.put("position_y", 2 * parameters.getInt("y"));
                                                        }
                                                        if (parameters.has("package_name")) {
                                                            currentAction.put("packageName", parameters.getString("package_name"));
                                                        }
                                                        // 处理其他参数
                                                        if (parameters.has("text")) {
                                                            currentAction.put("text", parameters.getString("text"));
                                                        }
                                                        if (parameters.has("direction")) {
                                                            currentAction.put("direction", parameters.getString("direction"));
                                                        }
                                                        if (parameters.has("target_element")) {
                                                            currentAction.put("target_element", parameters.getString("target_element"));
                                                        }
                                                    }
                                                    
                                                    Log.d("MainActivity", "已更新全局变量:");
                                                    Log.d("MainActivity", "functionName: " + functionName);
                                                    Log.d("MainActivity", "currentItem: " + currentItem.toString());
                                                    Log.d("MainActivity", "currentAction: " + currentAction.toString());
                                                    
                                                    // ========== 执行API指令 ==========
                                                    // 在成功解析API响应并更新全局变量后，立即执行相应的命令
                                                    Log.d("MainActivity", "开始执行API指令: " + functionName);
                                                    
                                                    // 检查是否被中断（只对连续命令序列有效，独立命令可以继续执行）
                                                    if (isTaskInterrupted && Thread.currentThread().isInterrupted()) {
                                                        Log.d("MainActivity", "检测到严重中断信号，停止命令执行");
                                                        return;
                                                    } else if (isTaskInterrupted) {
                                                        Log.d("MainActivity", "检测到中断标志，但这可能是新的独立命令，继续执行");
                                                    }
                                                    
                                                    boolean actionSuccess = false;
                                                    StringBuilder commandResponseBuilder = new StringBuilder();
                                                    
                                                    try {
                                                        // 执行click命令
                                                        if ("click".equals(functionName)) {
                                                            Log.d("MainActivity", "执行click命令");
                                                            
                                                            // 在UI线程中执行点击操作和Toast显示
                                                            runOnUiThread(() -> {
                                                                Log.d("MainActivity", "UI线程：准备显示点击Toast");
                                                                
                                                                // 检查中断标志，但只对连续执行的命令序列有效
                                                                // 如果这是一个新的独立命令，则忽略之前的中断标志
                                                                if (isTaskInterrupted) {
                                                                    Log.d("MainActivity", "检测到中断标志，但这是新的独立点击命令，继续执行");
                                                                }
                                                                
                                                                try {
                                                                    String targetElement = parameters != null ? parameters.optString("target_element", "") : "";
                                                                    
                                                                    // 检查是否包含"始终允许"，如果是则立即执行异常终止
                                                                    if (targetElement.contains("始终允许")) {
                                                                        Log.w("MainActivity", "检测到点击目标包含'始终允许'，执行异常终止");
                                                                        showExtendedToast("检测到权限操作，任务异常终止");
                                                                        
                                                                        // 立即执行terminate等价操作
                                                                        historyList = new JSONArray();
                                                                        resetAllStates();
                                                                        
                                                                        // 显示terminate完成Toast
                                                                        showEnhancedCompletionToast("任务异常终止");
                                                                        
                                                                        // 添加终止消息到聊天界面
                                                                        addResponse("🚨 任务异常终止\n" +
                                                                                  "✅ 所有运行中的线程已停止\n" +
                                                                                  "✅ 历史记录已清空\n" +
                                                                                  "✅ 所有状态已重置\n" +
                                                                                  "🔄 系统已恢复到初始状态", Message.SENT_BY_BOT);
                                                                        
                                                                        return; // 不继续执行点击操作
                                                                    }
                                                                    
                                                                    String toastMessage = targetElement.isEmpty() ? "开始点击" : "开始点击: " + targetElement;
                                                                    showExtendedToast(toastMessage);
                                                                    Log.d("MainActivity", "UI线程：点击Toast已调用");
                                                                } catch (Exception e) {
                                                                    Log.e("MainActivity", "UI线程：Toast显示失败", e);
                                                                }
                                                                
                                                                boolean clickSuccess = executeClickCommand(currentItem, currentAction, commandResponseBuilder);
                                                                Log.d("MainActivity", "UI线程中点击执行结果: " + (clickSuccess ? "成功" : "失败"));
                                                            });
                                                            
                                                            // 临时设置为true，实际结果在UI线程中处理
                                                            actionSuccess = true;
                                                        }
                                                        // 执行input命令
                                                        else if ("input".equals(functionName)) {
                                                            Log.d("MainActivity", "执行input命令");
                                                            
                                                            // 在UI线程中执行输入操作和Toast显示
                                                            runOnUiThread(() -> {
                                                                Log.d("MainActivity", "UI线程：准备显示输入Toast");
                                                                
                                                                // 检查中断标志，但只对连续执行的命令序列有效
                                                                if (isTaskInterrupted) {
                                                                    Log.d("MainActivity", "检测到中断标志，但这是新的独立输入命令，继续执行");
                                                                }
                                                                
                                                                try {
                                                                    String targetElement = parameters != null ? parameters.optString("text", "") : "";
                                                                    String packageName = parameters != null ? parameters.optString("package_name", "") : "";
                                                                    
                                                                    String toastMessage;
                                                                    if ("com.tencent.mm".equals(packageName)) {
                                                                        // 如果是微信，显示特殊提示
                                                                        String targetElementForToast = parameters != null ? parameters.optString("target_element", "") : "";
                                                                        toastMessage = "请手动输入" + (targetElementForToast.isEmpty() ? "" : "+" + targetElementForToast);
                                                                    } else {
                                                                        // 其他应用使用原有逻辑
                                                                        toastMessage = targetElement.isEmpty() ? "输入文字" : "输入文字: " + targetElement;
                                                                    }
                                                                    
                                                                    showExtendedToast(toastMessage);
                                                                    Log.d("MainActivity", "UI线程：输入Toast已调用，package_name: " + packageName + ", toast: " + toastMessage);
                                                                } catch (Exception e) {
                                                                    Log.e("MainActivity", "UI线程：Toast显示失败", e);
                                                                }
                                                                
                                                                boolean inputSuccess = executeInputCommand(currentItem, currentAction, commandResponseBuilder);
                                                                Log.d("MainActivity", "UI线程中输入执行结果: " + (inputSuccess ? "成功" : "失败"));
                                                            });
                                                            
                                                            // 临时设置为true，实际结果在UI线程中处理
                                                            actionSuccess = true;
                                                        }
                                                        // 执行swipe命令
                                                        else if ("swipe".equals(functionName)) {
                                                            Log.d("MainActivity", "执行swipe命令");
                                                            
                                                            // 在UI线程中执行滑动操作和Toast显示
                                                            runOnUiThread(() -> {
                                                                Log.d("MainActivity", "UI线程：准备显示滑动Toast");
                                                                
                                                                // 检查中断标志，但只对连续执行的命令序列有效
                                                                if (isTaskInterrupted) {
                                                                    Log.d("MainActivity", "检测到中断标志，但这是新的独立滑动命令，继续执行");
                                                                }
                                                                
                                                                try {
                                                                    String targetElement = parameters != null ? parameters.optString("direction", "") : "";
                                                                    String toastMessage = targetElement.isEmpty() ? "开始滑动" : "开始滑动: " + targetElement;
                                                                    showExtendedToast(toastMessage);
                                                                    Log.d("MainActivity", "UI线程：滑动Toast已调用");
                                                                } catch (Exception e) {
                                                                    Log.e("MainActivity", "UI线程：Toast显示失败", e);
                                                                }
                                                                
                                                                boolean swipeSuccess = executeSwipeCommand(currentItem, currentAction, commandResponseBuilder);
                                                                Log.d("MainActivity", "UI线程中滑动执行结果: " + (swipeSuccess ? "成功" : "失败"));
                                                            });
                                                            
                                                            // 临时设置为true，实际结果在UI线程中处理
                                                            actionSuccess = true;
                                                        }
                                                        // 执行appStart命令
                                                        else if ("open_app".equals(functionName)) {
                                                            Log.d("MainActivity", "执行appStart命令");
                                                            
                                                            // 在UI线程中执行应用启动操作和Toast显示
                                                            runOnUiThread(() -> {
                                                                // 检查中断标志，但只对连续执行的命令序列有效
                                                                if (isTaskInterrupted) {
                                                                    Log.d("MainActivity", "检测到中断标志，但这是新的独立应用启动命令，继续执行");
                                                                }
                                                                
                                                                String targetElement = parameters != null ? parameters.optString("package_name", "") : "";
                                                                String toastMessage = targetElement.isEmpty() ? "启动应用" : "启动应用: " + targetElement;
                                                                showExtendedToast(toastMessage);
                                                                
                                                                boolean appStartSuccess = executeAppStartCommand(currentItem, currentAction, commandResponseBuilder);
                                                                Log.d("MainActivity", "UI线程中应用启动执行结果: " + (appStartSuccess ? "成功" : "失败"));
                                                                
                                                                // 如果启动失败，显示失败Toast
                                                                if (!appStartSuccess) {
                                                                    showExtendedToast("app启动失败！");
                                                                    Log.d("MainActivity", "已显示app启动失败Toast");
                                                                } else {
                                                                    // 如果是微信应用启动成功，显示特殊提示
                                                                    if ("com.tencent.mm".equals(targetElement)) {
                                                                        showExtendedToast("文字输入功能有误，请手动输入");
                                                                        Log.d("MainActivity", "微信启动成功，已显示手动输入提示Toast");
                                                                    }
                                                                }
                                                            });
                                                            
                                                            // 临时设置为true，实际结果在UI线程中处理
                                                            actionSuccess = true;
                                                        }
                                                        // done命令 - 只需记录
                                                        else if ("done".equalsIgnoreCase(functionName != null ? functionName.trim() : "")) {
                                                            historyList = new JSONArray();
                                                            Log.d("MainActivity", "执行done命令，functionName值: '" + functionName + "'");
                                                            commandResponseBuilder.append("任务已完成");
                                                            actionSuccess = true;
                                                            

                                                            
                                                            // 在系统屏幕上弹出任务完成消息
                                                            runOnUiThread(() -> {
                                                                try {
                                                                    Log.d("MainActivity", "准备显示done命令完成Toast");
                                                                    String targetElement = currentAction.optJSONObject("parameters") != null ? 
                                                                        currentAction.optJSONObject("parameters").optString("target_element", "") : "";
                                                                    String message = !targetElement.isEmpty() ? 
                                                                        "✅ 任务已完成: " + targetElement : "✅ 任务已完成";
                                                                    
                                                                    Log.d("MainActivity", "done命令Toast消息: " + message);
                                                                    
                                                                    // 显示更明显的Toast
                                                                    showEnhancedCompletionToast(message);
                                                                    Log.d("MainActivity", "done命令完成Toast已启动");
                                                                } catch (Exception e) {
                                                                    Log.e("MainActivity", "显示done命令Toast失败", e);
                                                                }
                                                            });
                                                        }
                                                        else if ("terminate".equalsIgnoreCase(functionName != null ? functionName.trim() : "")) {
                                                            historyList = new JSONArray();
                                                            Log.d("MainActivity", "执行terminate命令，functionName值: '" + functionName + "'");
                                                            commandResponseBuilder.append("任务异常终止");
                                                            actionSuccess = true;
                                                            
                                                            // 在系统屏幕上弹出任务异常终止消息
                                                            runOnUiThread(() -> {
                                                                try {
                                                                    Log.d("MainActivity", "准备显示terminate命令完成Toast");
                                                                    String targetElement = currentAction.optJSONObject("parameters") != null ? 
                                                                        currentAction.optJSONObject("parameters").optString("target_element", "") : "";
                                                                    String message = !targetElement.isEmpty() ? 
                                                                        "⚠️ 任务异常终止: " + targetElement : "⚠️ 任务异常终止";
                                                                    
                                                                    Log.d("MainActivity", "terminate命令Toast消息: " + message);
                                                                    
                                                                    // 显示更明显的Toast
                                                                    showEnhancedCompletionToast(message);
                                                                    Log.d("MainActivity", "terminate命令完成Toast已启动");
                                                                } catch (Exception e) {
                                                                    Log.e("MainActivity", "显示terminate命令Toast失败", e);
                                                                }
                                                            });
                                                        }
                                                        else if ("wait".equals(functionName)) {
                                                            Log.d("MainActivity", "执行wait命令");
                                                            commandResponseBuilder.append("任务等待中");
                                                            actionSuccess = true;
                                                            try {
                                                                Thread.sleep(1500);
                                                            } catch (InterruptedException e) {
                                                                e.printStackTrace();
                                                            }

                                                            // 在系统屏幕上弹出任务等待中消息
                                                            runOnUiThread(() -> {
                                                                String targetElement = currentAction.optJSONObject("parameters") != null ? 
                                                                    currentAction.optJSONObject("parameters").optString("target_element", "") : "";
                                                                String message = !targetElement.isEmpty() ? 
                                                                    "任务等待中: " + targetElement : "任务等待中";
                                                                showExtendedToast(message);
                                                            });
                                                        }
                                                        else {
                                                            Log.w("MainActivity", "未知的功能名称: " + functionName);
                                                            commandResponseBuilder.append("未知命令: ").append(functionName);
                                                            actionSuccess = false;
                                                            runOnUiThread(() -> {
                                                                String targetElement = currentAction.optJSONObject("parameters") != null ? 
                                                                    currentAction.optJSONObject("parameters").optString("target_element", "") : "";
                                                                String message = !targetElement.isEmpty() ? 
                                                                    "异常界面，任务终止: " + targetElement : "异常界面，任务终止";
                                                                showExtendedToast(message);
                                                            });
                                                        }
                                                        
                                                        Log.d("MainActivity", "命令执行结果: " + (actionSuccess ? "成功" : "失败"));
                                                        Log.d("MainActivity", "命令执行详情: " + commandResponseBuilder.toString());
                                                        
                                                    } catch (Exception commandException) {
                                                        Log.e("MainActivity", "执行API命令时发生异常", commandException);
                                                        commandResponseBuilder.append("命令执行异常: ").append(commandException.getMessage());
                                                        actionSuccess = false;
                                                    }
                                                    
                                                } catch (JSONException e) {
                                                    Log.e("MainActivity", "更新全局变量失败", e);
                                                }
                                                
                                                // 保存原始functionName，因为某些命令（如terminate）会重置functionName
                                                String originalFunctionName = functionName;
                                                
                                                // 将原始响应添加到history中（done和terminate命令除外）
                                                String currentFunctionName = originalFunctionName != null ? originalFunctionName.trim() : "";
                                                boolean isTerminalCommand = "done".equalsIgnoreCase(currentFunctionName) || "terminate".equalsIgnoreCase(currentFunctionName);
                                                
                                                if (result != null && !result.trim().isEmpty() && !isTerminalCommand) {
                                                    historyList.put(result);
                                                    Log.d("MainActivity", "已将响应添加到history，当前history长度: " + historyList.length());
                                                } else if (isTerminalCommand) {
                                                    Log.d("MainActivity", "检测到终端命令 (" + currentFunctionName + ")，不添加响应到history，当前history长度: " + historyList.length());
                                                }

                                                // 构建显示给用户的回复信息
                                                StringBuilder displayMessage = new StringBuilder();
                                                displayMessage.append("AI分析: ").append(reasoning).append("\n\n");
                                                displayMessage.append("执行动作: ").append(action).append("\n");
                                                
                                                if (parameters != null) {
                                                    displayMessage.append("操作参数:\n");
                                                    
                                                    // 解析target_element参数
                                                    String targetElement = parameters.optString("target_element", "");
                                                    if (!targetElement.isEmpty()) {
                                                        displayMessage.append("  • 目标元素: ").append(targetElement).append("\n");
                                                    }
                                                    
                                                    // 解析bbox参数
                                                    String bbox = parameters.optString("bbox", "");
                                                    if (!bbox.isEmpty()) {
                                                        displayMessage.append("  • 坐标范围: ").append(bbox).append("\n");
                                                    }
                                                    
                                                    // 解析其他参数
                                                    if (parameters.has("text")) {
                                                        displayMessage.append("  • 输入文本: ").append(parameters.optString("text")).append("\n");
                                                    }
                                                    if (parameters.has("direction")) {
                                                        displayMessage.append("  • 滑动方向: ").append(parameters.optString("direction")).append("\n");
                                                    }
                                                    if (parameters.has("duration")) {
                                                        displayMessage.append("  • 持续时间: ").append(parameters.optString("duration")).append("\n");
                                                    }
                                                }
                                                
                                                String finalDisplayMessage = displayMessage.toString();
                                                
                                                runOnUiThread(() -> {
                                                    // Hide typing indicator and display the response
                                                    messageList.remove(messageList.size() - 1);
                                                    messageAdapter.notifyDataSetChanged();
                                                    recyclerView.smoothScrollToPosition(messageAdapter.getItemCount());
                                                    addResponse(finalDisplayMessage, Message.SENT_BY_BOT);
                                                });
                                                
                                                // ========== 检查是否需要继续执行 ==========
                                                // 如果不是done或terminate命令，等待一段时间后继续执行下一步
                                                // 使用原始的functionName，因为terminate命令会重置functionName为null
                                                String trimmedFunctionName = originalFunctionName != null ? originalFunctionName.trim() : "";
                                                if (!"done".equalsIgnoreCase(trimmedFunctionName) && !"terminate".equalsIgnoreCase(trimmedFunctionName)) {
                                                    Log.d("MainActivity", "当前命令不是done或terminate ('" + trimmedFunctionName + "')，准备继续执行下一步");
                                                    currentApiThread = new Thread(() -> {
                                                        // 将递归线程添加到活跃线程列表
                                                        addActiveThread(Thread.currentThread());
                                                        
                                                        try {
                                                            // 在等待期间检查中断
                                                            if (isTaskInterrupted || Thread.currentThread().isInterrupted()) {
                                                                Log.d("MainActivity", "检测到任务中断，停止继续执行");
                                                                return;
                                                            }
                                                            
                                                            // 分段等待，每500ms检查一次中断
                                                            for (int i = 0; i < 4; i++) { // 总共等待2秒（4 * 500ms）
                                                                if (isTaskInterrupted || Thread.currentThread().isInterrupted()) {
                                                                    Log.d("MainActivity", "等待期间检测到任务中断");
                                                                    return;
                                                                }
                                                                Thread.sleep(500);
                                                            }
                                                            
                                                            Log.d("MainActivity", "开始继续执行下一步");
                                                            
                                                            // 最终检查中断
                                                            if (isTaskInterrupted || Thread.currentThread().isInterrupted()) {
                                                                Log.d("MainActivity", "检测到任务中断，停止递归调用");
                                                                return;
                                                            }
                                                            
                                                            // 递归调用API来继续处理
                                                            callAPI(question);
                                                        } catch (InterruptedException e) {
                                                            Log.e("MainActivity", "等待下一步执行时被中断", e);
                                                        } finally {
                                                            // 从活跃线程列表移除递归线程
                                                            removeActiveThread(Thread.currentThread());
                                                        }
                                                    });
                                                    currentApiThread.start();
                                                } else {
                                                    // 对于done和terminate命令，确保不继续执行下一步
                                                    // terminate命令已经在执行时清空了historyList，这里不需要重复清空
                                                    if ("done".equalsIgnoreCase(trimmedFunctionName)) {
                                                        historyList = new JSONArray();
                                                    }
                                                    Log.d("MainActivity", "检测到done或terminate命令 ('" + trimmedFunctionName + "')，停止继续执行");
                                                    // 注意：done和terminate命令的Toast已经在前面的命令处理逻辑中显示过了，这里不再重复显示
                                                }
                                                
                                            } catch (JSONException parseException) {
                                                // 响应体不是有效的JSON
                                                responseJson.put("parse_status", "failed");
                                                responseJson.put("parse_error", parseException.getMessage());
                                                
                                                // ========== JSON解析失败日志 ==========
                                                Log.w("MainActivity", "========== JSON解析失败 ==========");
                                                Log.w("MainActivity", "原始响应内容: " + result);
                                                Log.w("MainActivity", "解析错误信息: " + parseException.getMessage());
                                                Log.w("MainActivity", "响应不是有效的JSON格式");
                                                
                                                // 即使解析失败，也将原始响应添加到history中
                                                if (result != null && !result.trim().isEmpty()) {
                                                    historyList.put(result);
                                                    Log.d("MainActivity", "已将解析失败的响应添加到history，当前history长度: " + historyList.length());
                                                }
                                                
                                                Log.w("MainActivity", "响应体JSON解析失败: " + parseException.getMessage());
                                                Log.d("MainActivity", "响应序列化JSON: " + responseJson.toString(2));
                                                
                                                // 创建final变量用于lambda表达式
                                                final String finalResult = result;
                                                runOnUiThread(() -> {
                                                    messageList.remove(messageList.size() - 1);
                                                    messageAdapter.notifyDataSetChanged();
                                                    recyclerView.smoothScrollToPosition(messageAdapter.getItemCount());
                                                    addResponse("响应格式解析失败: " + parseException.getMessage() + "\n原始响应: " + finalResult, Message.SENT_BY_BOT);
                                                });
                                            }
                                        } else {
                                            // 成功但是响应体为空
                                            responseJson.put("parse_status", "empty_body");
                                            Log.w("MainActivity", "服务器返回成功但响应体为空");
                                            
                                            runOnUiThread(() -> {
                                                messageList.remove(messageList.size() - 1);
                                                messageAdapter.notifyDataSetChanged();
                                                recyclerView.smoothScrollToPosition(messageAdapter.getItemCount());
                                                addResponse("服务器返回成功但没有内容", Message.SENT_BY_BOT);
                                            });
                                        }
                                    } else {
                                        // HTTP请求失败（包括422错误）
                                        responseJson.put("error_description", "HTTP请求返回非成功状态码");
                                        
                                        // ========== HTTP错误响应日志 ==========
                                        Log.e("MainActivity", "========== HTTP请求失败 ==========");
                                        Log.e("MainActivity", "HTTP状态码: " + response.code());
                                        Log.e("MainActivity", "HTTP状态消息: " + response.message());
                                        Log.e("MainActivity", "错误响应体: " + (result != null ? result : "null"));
                                    }
                                    
                                    // ========== 完整响应格式总结 ==========
                                    Log.d("MainActivity", "========== 完整响应处理总结 ==========");
                                    Log.d("MainActivity", "最终构建的响应JSON结构:");
                                    Log.d("MainActivity", responseJson.toString(2));
                                    Log.d("MainActivity", "==========================================");
                                    
                                } catch (JSONException e) {
                                    Log.e("MainActivity", "序列化响应体为JSON时出错", e);
                                } finally {
                                    // 确保response被正确关闭，防止连接泄漏
                                    if (response != null) {
                                        response.close();
                                    }
                                }
                            }
                        });

            } catch (Exception e) {
                responseBuilder.append("\n❌ 批量执行过程中发生严重错误: ").append(e.getMessage());
                Log.e("MainActivity", "批量执行命令时出错: " + e.getMessage());
                e.printStackTrace();
            } finally {
                // 从活跃线程列表移除当前线程
                removeActiveThread(Thread.currentThread());
                Log.d("MainActivity", "API线程执行完毕，已从活跃线程列表移除");
            }
        } catch (Exception e) {
            Log.e("MainActivity", "线程执行过程中发生错误: " + e.getMessage());
            e.printStackTrace();
            // 确保线程被移除
            removeActiveThread(Thread.currentThread());
        }
        });
        
        // 启动线程
        currentApiThread.start();
    }
    
    /**
     * 执行点击命令
     */
    private boolean executeClickCommand(JSONObject currentItem, JSONObject currentAction, StringBuilder responseBuilder) {
        try {
            int position_x = -1;
            int position_y = -1;
            
            // 调试日志：显示currentAction的状态
            Log.d("MainActivity", "执行点击命令 - currentAction: " + (currentAction != null ? currentAction.toString() : "null"));
            
            // 添加坐标信息到响应
            if (currentAction != null) {
                if (currentAction.has("position_x")) {
                    position_x = currentAction.getInt("position_x");
                    Log.d("MainActivity", "从currentAction获取position_x: " + position_x);
                }
                if (currentAction.has("position_y")) {
                    position_y = currentAction.getInt("position_y");
                    Log.d("MainActivity", "从currentAction获取position_y: " + position_y);
                }
                responseBuilder.append(" (").append(position_x).append(", ").append(position_y).append(")");
            } else {
                Log.w("MainActivity", "currentAction为null，无法获取坐标信息");
                responseBuilder.append(" (currentAction为null)");
            }
            
            if (position_x != -1 && position_y != -1) {
                if (!isAccessibilityServiceEnabled()) {
                    responseBuilder.append(" ❌ 无障碍服务未启用");
                    return false;
                } else {
                    Log.d("MainActivity", "准备执行点击操作: (" + position_x + ", " + position_y + ")");
                    try {
                        monitoru2.u2Click(position_x, position_y);
                        responseBuilder.append(" ✅ 点击成功");
                        return true;
                    } catch (Exception e) {
                        Log.e("MainActivity", "点击失败: " + e.getMessage());
                        responseBuilder.append(" ❌ 点击失败: ").append(e.getMessage());
                        
                        // 如果是无障碍服务问题，提示用户重新启用
                        if (e.getMessage().contains("无障碍服务未启用") || e.getMessage().contains("服务实例为null")) {
                            responseBuilder.append("\n💡 请检查无障碍服务是否正常运行");
                        }
                        return false;
                    }
                }
            } else {
                Log.e("MainActivity", "点击失败：坐标信息无效 - x: " + position_x + ", y: " + position_y);
                responseBuilder.append(" ❌ 缺少坐标信息");
                return false;
            }
        } catch (Exception e) {
            Log.e("MainActivity", "执行点击命令异常", e);
            responseBuilder.append(" ❌ 执行失败: ").append(e.getMessage());
            return false;
        }
    }
    
    /**
     * 执行输入命令
     */
    private boolean executeInputCommand(JSONObject currentItem, JSONObject currentAction, StringBuilder responseBuilder) {
        try {
            String inputText = "";
            int position_x = -1;
            int position_y = -1;
            
            // 从react.json中获取text参数
            if (currentItem.has("function") && 
                currentItem.getJSONObject("function").has("parameters") &&
                currentItem.getJSONObject("function").getJSONObject("parameters").has("text")) {
                inputText = currentItem.getJSONObject("function").getJSONObject("parameters").getString("text");
                Log.d("MainActivity", "从react.json获取输入文本: " + inputText);
                
                // 同时获取坐标信息
                JSONObject parameters = currentItem.getJSONObject("function").getJSONObject("parameters");
                if (parameters.has("x")) {
                    position_x = 2 * parameters.getInt("x");
                }
                if (parameters.has("y")) {
                    position_y = 2 * parameters.getInt("y");
                }
            }
            
            responseBuilder.append(" \"").append(inputText).append("\"");
            
            if (!inputText.isEmpty()) {
                if (!isAccessibilityServiceEnabled()) {
                    responseBuilder.append(" ❌ 无障碍服务未启用");
                    Log.e("MainActivity", "输入失败：无障碍服务未启用");
                    return false;
                } else {
                    Log.d("MainActivity", "准备执行文本输入: " + inputText);
                    Log.d("MainActivity", "输入框坐标: (" + position_x + ", " + position_y + ")");
                    
                    try {
                        // 执行文本输入
                        monitoru2.u2Input(inputText);
                        responseBuilder.append(" ✅ 输入成功");
                        Log.d("MainActivity", "文本输入完成");
                        return true;
                    } catch (Exception inputException) {
                        Log.e("MainActivity", "输入操作执行异常", inputException);
                        responseBuilder.append(" ❌ 输入执行异常: ").append(inputException.getMessage());
                        return false;
                    }
                }
            } else {
                responseBuilder.append(" ❌ 缺少输入文本");
                Log.e("MainActivity", "输入失败：缺少输入文本");
                return false;
            }
        } catch (Exception e) {
            Log.e("MainActivity", "执行输入命令异常", e);
            responseBuilder.append(" ❌ 执行失败: ").append(e.getMessage());
            return false;
        }
    }
    
    /**
     * 执行滑动命令
     */
    private boolean executeSwipeCommand(JSONObject currentItem, JSONObject currentAction, StringBuilder responseBuilder) {
        try {
            String direction = "";
            
            // 从react.json中获取direction参数
            if (currentItem.has("function") && 
                currentItem.getJSONObject("function").has("parameters") &&
                currentItem.getJSONObject("function").getJSONObject("parameters").has("direction")) {
                direction = currentItem.getJSONObject("function").getJSONObject("parameters").getString("direction");
            }
            
            // 如果react.json中没有direction，尝试从action数据中获取
            if (direction.isEmpty() && currentAction != null && currentAction.has("direction")) {
                direction = currentAction.getString("direction");
            }
            
            responseBuilder.append(" ").append(direction);
            
            if (!direction.isEmpty()) {
                if (!isAccessibilityServiceEnabled()) {
                    responseBuilder.append(" ❌ 无障碍服务未启用");
                    return false;
                } else {
                    monitoru2.u2SwipeExt(direction, 1.0);
                    responseBuilder.append(" ✅ 滑动成功");
                    return true;
                }
            } else {
                responseBuilder.append(" ❌ 缺少滑动方向");
                return false;
            }
        } catch (Exception e) {
            responseBuilder.append(" ❌ 执行失败: ").append(e.getMessage());
            return false;
        }
    }
    
    /**
     * 执行动作前截图命令 - 使用新的MediaProjection前台服务架构
     * 在每次动作执行前截取一张当前屏幕状态图
     */
    private boolean executeDoneCommand(StringBuilder responseBuilder) {
        Log.d("MainActivity", "=== 执行截图命令 ===");
        
        try {
            if (screenshotManager == null) {
                Log.e("MainActivity", "ScreenshotManager未初始化，尝试重新初始化");
                responseBuilder.append("❌ 截图功能未初始化，正在重新初始化...");
                
                // 尝试重新初始化
                try {
                    screenshotManager = new ScreenshotManager(this);
                    Log.d("MainActivity", "ScreenshotManager重新初始化成功");
                    responseBuilder.append("\n✅ 重新初始化成功");
                } catch (Exception e) {
                    Log.e("MainActivity", "重新初始化ScreenshotManager失败: " + e.getMessage());
                    responseBuilder.append("\n❌ 重新初始化失败，请重启应用");
                    return false;
                }
            }
            
            // 检查权限状态
            if (!screenshotManager.hasPermission()) {
                Log.w("MainActivity", "没有截图权限，提示用户重新授权");
                responseBuilder.append("❌ 截图权限不足，正在请求权限...");
                
                // 自动申请权限
                screenshotManager.requestScreenCapturePermission();
                return false;
            }
            
            // 执行截图
            final boolean[] success = {false};
            final CountDownLatch latch = new CountDownLatch(1);
            final String[] errorMsg = {null};
            
            screenshotManager.takeScreenshot(new ScreenshotManager.ScreenshotCallback() {
                @Override
                public void onSuccess(String filePath) {
                    Log.d("MainActivity", "截图成功: " + filePath);
                    success[0] = true;
                    
                    // 保存最新截图路径
                    lastScreenshotPath = filePath;
                    
                    responseBuilder.append("✅ 截图已完成!\n")
                                  .append("保存位置: ").append(filePath);
                    latch.countDown();
                }

                @Override
                public void onError(String error) {
                    Log.e("MainActivity", "截图失败: " + error);
                    errorMsg[0] = error;
                    responseBuilder.append("❌ 截图失败: ").append(error);
                    latch.countDown();
                }

                @Override
                public void onPermissionRequired() {
                    Log.w("MainActivity", "检测到权限过期，自动重新申请权限");
                    responseBuilder.append("⚠️ 检测到MediaProjection权限过期，正在重新获取权限...\n");
                    responseBuilder.append("请在弹出的权限对话框中点击'立即开始'来重新授权");
                    
                    // 自动重新请求权限
                    screenshotManager.requestScreenCapturePermission();
                    
                    // 先标记为需要权限，不立即结束等待
                    // ScreenshotManager会在权限获取后自动重试
                    success[0] = false;
                    latch.countDown();
                }
            });
            
            // 等待截图完成（最多等待10秒）
            try {
                boolean finished = latch.await(10, TimeUnit.SECONDS);
                if (!finished) {
                    Log.e("MainActivity", "截图操作超时");
                    responseBuilder.append("❌ 截图操作超时");
                    return false;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                Log.e("MainActivity", "截图操作被中断");
                responseBuilder.append("❌ 截图操作被中断");
                return false;
            }
            
            // 检查截图结果和错误信息
            if (!success[0]) {
                Log.e("MainActivity", "截图失败，错误信息: " + (errorMsg[0] != null ? errorMsg[0] : "未知错误"));
                if (errorMsg[0] != null) {
                    responseBuilder.append("\n详细错误: ").append(errorMsg[0]);
                }
                
                // 检查截图文件是否真的存在
                if (lastScreenshotPath != null) {
                    File screenshotFile = new File(lastScreenshotPath);
                    if (screenshotFile.exists() && screenshotFile.length() > 0) {
                        Log.w("MainActivity", "虽然截图回调显示失败，但文件实际存在且不为空，尝试使用该文件");
                        responseBuilder.append("\n⚠️ 检测到截图文件存在，尝试继续使用");
                        return true; // 尝试继续使用该文件
                    } else {
                        Log.e("MainActivity", "截图文件不存在或为空: " + lastScreenshotPath);
                        responseBuilder.append("\n❌ 截图文件不存在或为空");
                    }
                }
            } else {
                // 成功情况下也要验证文件
                if (lastScreenshotPath != null) {
                    File screenshotFile = new File(lastScreenshotPath);
                    if (!screenshotFile.exists() || screenshotFile.length() == 0) {
                        Log.e("MainActivity", "截图回调显示成功，但文件不存在或为空: " + lastScreenshotPath);
                        responseBuilder.append("\n❌ 截图文件验证失败");
                        return false;
                    } else {
                        Log.d("MainActivity", "截图文件验证成功，大小: " + screenshotFile.length() + " 字节");
                    }
                }
            }
            
            return success[0];
            
        } catch (Exception e) {
            Log.e("MainActivity", "执行截图命令时发生异常: " + e.getMessage());
            e.printStackTrace();
            responseBuilder.append("❌ 截图命令执行异常: ").append(e.getMessage());
            return false;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // 处理MediaProjection权限申请结果
        if (requestCode == REQUEST_CODE_MEDIA_PROJECTION) {
            handleMediaProjectionPermissionResult(resultCode, data, "MainActivity权限申请");
        }
        // 处理ScreenshotManager的权限申请结果（如果使用的是ScreenshotManager.SCREENSHOT_REQUEST_CODE）
        else if (requestCode == 100) { // ScreenshotManager.SCREENSHOT_REQUEST_CODE
            handleMediaProjectionPermissionResult(resultCode, data, "ScreenshotManager权限申请");
        }
        
    }

    /**
     * 统一处理MediaProjection权限申请结果
     * @param resultCode 结果码
     * @param data 结果数据
     * @param source 权限申请来源（用于日志）
     */
    private void handleMediaProjectionPermissionResult(int resultCode, Intent data, String source) {
        if (resultCode == RESULT_OK && data != null) {
            Log.d("MainActivity", source + " - MediaProjection权限授权成功");
            
            // 将权限数据传递给ScreenshotManager
            try {
                if (screenshotManager != null) {
                    // ScreenshotManager使用ActivityResultLauncher自动处理权限结果
                    // 无需手动调用onActivityResult
                    
                    // 更新内部状态
                    hasRequestedMediaProjectionPermission = true;
                    
                    Log.d("MainActivity", "权限授权成功，ScreenshotManager将自动处理");
                } else {
                    Log.e("MainActivity", "ScreenshotManager为null");
                }
                
                // 显示授权成功提示
                android.widget.Toast.makeText(this, 
                    "✅ 截图权限已永久授权！将不再需要重复申请权限", 
                    android.widget.Toast.LENGTH_LONG).show();
                    
                Log.d("MainActivity", source + " - MediaProjection权限已永久授权，今后无需重复申请");
                
                // 立即尝试保持MediaProjection活跃状态
                tryKeepMediaProjectionAlive();
                
            } catch (Exception e) {
                Log.e("MainActivity", "保存MediaProjection权限数据失败: " + e.getMessage());
                android.widget.Toast.makeText(this, 
                    "权限保存失败，请重试", 
                    android.widget.Toast.LENGTH_SHORT).show();
            }
        } else {
            Log.w("MainActivity", source + " - MediaProjection权限被拒绝，resultCode: " + resultCode);
            
            // 权限被拒绝，重置标记
            hasRequestedMediaProjectionPermission = false;
            
            android.widget.Toast.makeText(this, 
                "❌ 截图权限被拒绝，请重启应用重新授权", 
                android.widget.Toast.LENGTH_LONG).show();
        }
    }

    /**
     * 添加消息到聊天列表
     * @param message 消息内容
     * @param sentBy 发送者类型（Message.SENT_BY_ME 或 Message.SENT_BY_BOT）
     */
    void addToChat(String message, String sentBy) {
        runOnUiThread(() -> {
            messageList.add(new Message(message, sentBy));      // 添加消息到列表
            messageAdapter.notifyDataSetChanged();              // 通知适配器数据改变
            recyclerView.smoothScrollToPosition(messageAdapter.getItemCount()); // 滚动到最新消息
        });
    }

    /**
     * 添加响应到聊天列表（与addToChat功能相同，保持兼容性）
     * @param response 响应内容
     * @param sentBy 发送者类型
     */
    void addResponse(String response, String sentBy) {
        runOnUiThread(() -> {
            messageList.add(new Message(response, sentBy));     // 添加响应到列表
            messageAdapter.notifyDataSetChanged();              // 通知适配器数据改变
            recyclerView.smoothScrollToPosition(messageAdapter.getItemCount()); // 滚动到最新消息
        });
    }

    /**
     * 清空聊天记录
     */
    void clearChat() {
        messageList.clear();                    // 清空消息列表
        messageAdapter.notifyDataSetChanged();  // 通知适配器数据改变
    }

    /**
     * 获取最后一条机器人回复
     * @return 最后一条机器人消息，如果没有则返回空字符串
     */
    private String getLastResponse() {
        // 从后往前遍历消息列表
        for (int i = messageList.size() - 1; i >= 0; i--) {
            Message message = messageList.get(i);
            if (message.getSentBy().equals(Message.SENT_BY_BOT)) {
                return message.getMessage();  // 返回找到的第一条机器人消息
            }
        }
        return "";  // 如果没有找到机器人消息，返回空字符串
    }

    /**
     * 复制文本到剪贴板
     * @param text 要复制的文本内容
     */
    private void copyToClipboard(String text) {
        ClipData clip = ClipData.newPlainText("ChatBotResponse", text);
        clipboard.setPrimaryClip(clip);  // 设置剪贴板内容
        Toast.makeText(this, "Response copied to clipboard", Toast.LENGTH_SHORT).show();
    }

    /**
     * 分享响应内容
     * @param text 要分享的文本内容
     */
    private void shareResponse(String text) {
        Intent shareIntent = new Intent(Intent.ACTION_SEND);
        shareIntent.setType("text/plain");                     // 设置分享类型为纯文本
        shareIntent.putExtra(Intent.EXTRA_TEXT, text);         // 设置分享的文本内容
        startActivity(Intent.createChooser(shareIntent, "Share response")); // 启动系统分享选择器
    }

    /**
     * 显示软键盘
     */
    private void showKeyboard() {
        InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.toggleSoftInput(InputMethodManager.SHOW_FORCED, InputMethodManager.HIDE_IMPLICIT_ONLY);
    }

    /**
     * 初始化截图管理器并在首次启动时主动获取权限
     * 使用新的MediaProjection前台服务架构
     */
    private void initializeScreenshotManagerAndRequestPermission() {
        try {
            Log.d("MainActivity", "=== 初始化ScreenshotManager ===");
            
            // 初始化ScreenshotManager
            screenshotManager = new ScreenshotManager(this);
            
            // 检查是否已有权限
            if (screenshotManager.hasPermission()) {
                Log.d("MainActivity", "✅ 截图权限已准备就绪");
                hasRequestedMediaProjectionPermission = true;
                
                Toast.makeText(this, 
                    "截图功能已就绪", 
                    Toast.LENGTH_SHORT).show();
            } else {
                Log.d("MainActivity", "🔐 需要申请截图权限");
                requestInitialScreenshotPermission();
            }
            
        } catch (Exception e) {
            Log.e("MainActivity", "初始化截图管理器失败: " + e.getMessage());
            e.printStackTrace();
            
            hasRequestedMediaProjectionPermission = false;
            Toast.makeText(this, 
                "截图功能初始化失败", 
                Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * 主动申请截图权限（首次启动时调用）
     */
    private void requestInitialScreenshotPermission() {
        Log.d("MainActivity", "=== 开始申请截图权限 ===");
        
        try {
            // 显示友好的申请提示
            Toast.makeText(this, 
                "首次使用需要授权截图功能，请点击\"立即开始\"", 
                Toast.LENGTH_LONG).show();
            
            // 延迟1秒后发起权限请求，让Toast有时间显示
            new Handler(Looper.getMainLooper()).postDelayed(() -> {
                screenshotManager.requestScreenCapturePermission();
            }, 1000);
            
        } catch (Exception e) {
            Log.e("MainActivity", "申请截图权限失败: " + e.getMessage());
            e.printStackTrace();
            
            Toast.makeText(this, 
                "申请截图权限失败: " + e.getMessage(), 
                Toast.LENGTH_LONG).show();
        }
    }

    /**
     * 检查无障碍服务是否已启用
     * @return true如果已启用，false如果未启用
     */
    private boolean isAccessibilityServiceEnabled() {
        try {
            MyAccessibilityService service = MyAccessibilityService.getInstance();
            if (service == null) {
                // 在系统屏幕上弹出无障碍权限提示
                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this, "请开启无障碍权限", Toast.LENGTH_LONG).show();
                });
                Log.w("MainActivity", "无障碍服务未启用，已提示用户开启权限");
                
                // 执行异常终止处理
                executeTerminateCheatCode();
                
                return false;
            }
            return true;
        } catch (Exception e) {
            Log.e("MainActivity", "检查无障碍服务状态失败: " + e.getMessage());
            // 在系统屏幕上弹出无障碍权限提示
            runOnUiThread(() -> {
                Toast.makeText(MainActivity.this, "请开启无障碍权限", Toast.LENGTH_LONG).show();
            });
            
            // 执行异常终止处理
            executeTerminateCheatCode();
            
            return false;
        }
    }

    /**
     * 执行done作弊代码 - 在无障碍权限未启用时自动完成任务
     */
    /**
     * 执行terminate异常终止代码 - 在无障碍权限未启用等非正常情况时调用
     */
    private void executeTerminateCheatCode() {
        Log.d("MainActivity", "执行terminate异常终止代码 - 无障碍权限未启用");
        
        // 在新线程中执行，避免阻塞UI
        new Thread(() -> {
            try {
                // 等待一小段时间让Toast显示
                Thread.sleep(1000);
                
                // 执行完全重置
                resetAllStates();
                
                // 清空历史记录
                historyList = new JSONArray();
                
                runOnUiThread(() -> {
                    // 添加terminate异常终止消息到聊天界面
                    addResponse("� 任务异常终止（无障碍权限未启用）\n" +
                              "✅ 所有运行中的线程已停止\n" +
                              "✅ 历史记录已清空\n" +
                              "✅ 所有状态已重置\n" +
                              "⚠️ 请启用无障碍权限以获得完整功能\n" +
                              "🔄 系统已恢复到初始状态", Message.SENT_BY_BOT);
                    
                    // 弹出异常终止提示
                    showEnhancedCompletionToast("任务异常终止");
                });
                
                Log.d("MainActivity", "terminate异常终止代码执行完成");
                
            } catch (Exception e) {
                Log.e("MainActivity", "执行terminate异常终止代码失败", e);
            }
        }).start();
    }

    /**
     * 显示延长时间的Toast（约3.5秒）
     */
    private void showExtendedToast(String message) {
        runOnUiThread(() -> {
            try {
                // 显示Toast（3.5秒）
                Toast.makeText(MainActivity.this, message, Toast.LENGTH_LONG).show();
                
            } catch (Exception e) {
                Log.e("MainActivity", "显示延长Toast失败: " + message, e);
            }
        });
    }

    // 显示增强版任务完成Toast
    private void showEnhancedCompletionToast(String message) {
        runOnUiThread(() -> {
            try {
                // 添加振动反馈
                try {
                    Vibrator vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
                    if (vibrator != null && vibrator.hasVibrator()) {
                        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                            // 创建振动模式：短-长-短 (表示完成)
                            long[] pattern = {0, 200, 100, 400, 100, 200};
                            VibrationEffect effect = VibrationEffect.createWaveform(pattern, -1);
                            vibrator.vibrate(effect);
                        } else {
                            // 兼容老版本
                            long[] pattern = {0, 200, 100, 400, 100, 200};
                            vibrator.vibrate(pattern, -1);
                        }
                        Log.d("MainActivity", "任务完成振动反馈已触发");
                    }
                } catch (Exception e) {
                    Log.e("MainActivity", "振动反馈失败", e);
                }
                
                // 连续显示多个Toast确保用户看到
                Toast completionToast = Toast.makeText(MainActivity.this, message, Toast.LENGTH_LONG);
                completionToast.show();
                Log.d("MainActivity", "增强版任务完成Toast已启动: " + message);
                
            } catch (Exception e) {
                Log.e("MainActivity", "显示增强版任务完成Toast失败: " + message, e);
            }
        });
    }

    /**
     * 尝试保持MediaProjection实例活跃，延长权限有效期
     */
    private void tryKeepMediaProjectionAlive() {
        Log.d("MainActivity", "尝试保持MediaProjection实例活跃状态");
        
        new Thread(() -> {
            try {
                // 延迟一点时间确保服务初始化完成
                Thread.sleep(1500);
                
                if (screenshotManager != null && screenshotManager.hasPermission()) {
                    Log.d("MainActivity", "MediaProjection权限活跃状态已保持");
                } else {
                    Log.w("MainActivity", "ScreenshotManager未就绪，无法保持MediaProjection活跃");
                }
                
            } catch (Exception e) {
                Log.e("MainActivity", "保持MediaProjection活跃状态失败: " + e.getMessage());
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        Log.d("MainActivity", "MainActivity正在销毁，开始清理资源");
        
        // 清理所有线程
        interruptAllActiveThreads();
        
        // 释放ScreenshotManager资源
        if (screenshotManager != null) {
            try {
                screenshotManager.cleanup();
            } catch (Exception e) {
                Log.e("MainActivity", "释放ScreenshotManager资源失败: " + e.getMessage());
            }
        }
        
        Log.d("MainActivity", "MainActivity资源清理完成");
    }

    /**
     * 将图片文件转换为Base64编码字符串
     * @param imagePath 图片文件路径
     * @return Base64编码的字符串，如果转换失败返回null
     */
    private String convertImageToBase64(String imagePath) {
        if (imagePath == null || imagePath.isEmpty()) {
            Log.e("MainActivity", "图片路径为空");
            return null;
        }
        
        try {
            File imageFile = new File(imagePath);
            if (!imageFile.exists()) {
                Log.e("MainActivity", "图片文件不存在: " + imagePath);
                return null;
            }
            
            if (imageFile.length() == 0) {
                Log.e("MainActivity", "图片文件为空: " + imagePath);
                return null;
            }
            
            Log.d("MainActivity", "开始处理图片文件: " + imagePath + "，大小: " + imageFile.length() + " 字节");
            
            // 读取图片为Bitmap
            Bitmap originalBitmap = BitmapFactory.decodeFile(imagePath);
            if (originalBitmap == null) {
                Log.e("MainActivity", "无法解码图片文件: " + imagePath);
                
                // 尝试直接读取文件并转换为Base64（不缩放）
                try {
                    byte[] fileBytes = java.nio.file.Files.readAllBytes(imageFile.toPath());
                    if (fileBytes.length > 0) {
                        String base64 = Base64.encodeToString(fileBytes, Base64.DEFAULT);
                        Log.d("MainActivity", "直接读取文件转换Base64成功，长度: " + base64.length());
                        return base64;
                    }
                } catch (Exception e) {
                    Log.e("MainActivity", "直接读取文件也失败: " + e.getMessage());
                }
                
                return null;
            }
            
            // 获取原始尺寸
            int originalWidth = originalBitmap.getWidth();
            int originalHeight = originalBitmap.getHeight();
            Log.d("MainActivity", "原始图片尺寸: " + originalWidth + "x" + originalHeight);
            
            if (originalWidth <= 0 || originalHeight <= 0) {
                Log.e("MainActivity", "图片尺寸无效: " + originalWidth + "x" + originalHeight);
                originalBitmap.recycle();
                return null;
            }
            
            // 计算缩放后的尺寸（一半）
            int scaledWidth = Math.max(1, originalWidth / 2);
            int scaledHeight = Math.max(1, originalHeight / 2);
            Log.d("MainActivity", "缩放后图片尺寸: " + scaledWidth + "x" + scaledHeight);
            
            // 缩放图片
            Bitmap scaledBitmap = null;
            try {
                scaledBitmap = Bitmap.createScaledBitmap(originalBitmap, scaledWidth, scaledHeight, true);
                if (scaledBitmap == null) {
                    Log.e("MainActivity", "创建缩放Bitmap失败");
                    originalBitmap.recycle();
                    return null;
                }
            } catch (OutOfMemoryError e) {
                Log.e("MainActivity", "缩放图片时内存不足，尝试更小尺寸", e);
                originalBitmap.recycle();
                
                // 尝试更小的尺寸
                int smallerWidth = Math.max(1, originalWidth / 4);
                int smallerHeight = Math.max(1, originalHeight / 4);
                try {
                    Bitmap newOriginal = BitmapFactory.decodeFile(imagePath);
                    if (newOriginal != null) {
                        scaledBitmap = Bitmap.createScaledBitmap(newOriginal, smallerWidth, smallerHeight, true);
                        newOriginal.recycle();
                        Log.d("MainActivity", "使用更小尺寸成功: " + smallerWidth + "x" + smallerHeight);
                    }
                } catch (Exception e2) {
                    Log.e("MainActivity", "更小尺寸也失败", e2);
                    return null;
                }
                
                if (scaledBitmap == null) {
                    return null;
                }
            }
            
            // 释放原始Bitmap
            originalBitmap.recycle();
            
            // 将缩放后的Bitmap转换为字节数组
            ByteArrayOutputStream baos = null;
            try {
                baos = new ByteArrayOutputStream();
                boolean compressResult = scaledBitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);
                if (!compressResult) {
                    Log.e("MainActivity", "Bitmap压缩失败");
                    return null;
                }
                
                byte[] imageBytes = baos.toByteArray();
                if (imageBytes.length == 0) {
                    Log.e("MainActivity", "压缩后的图片数据为空");
                    return null;
                }
                
                // 转换为Base64
                String base64String = Base64.encodeToString(imageBytes, Base64.DEFAULT);
                Log.d("MainActivity", "图片缩放并转换为Base64成功，压缩后大小: " + imageBytes.length + " 字节，Base64长度: " + base64String.length());
                
                return base64String;
                
            } catch (Exception e) {
                Log.e("MainActivity", "Bitmap压缩或Base64转换失败", e);
                return null;
            } finally {
                // 释放缩放后的Bitmap
                if (scaledBitmap != null) {
                    scaledBitmap.recycle();
                }
                // 关闭ByteArrayOutputStream
                if (baos != null) {
                    try {
                        baos.close();
                    } catch (IOException e) {
                        Log.e("MainActivity", "关闭ByteArrayOutputStream失败", e);
                    }
                }
            }
            
        } catch (OutOfMemoryError e) {
            Log.e("MainActivity", "内存不足，无法处理图片: " + imagePath, e);
            return null;
        } catch (Exception e) {
            Log.e("MainActivity", "图片缩放转Base64失败: " + imagePath, e);
            return null;
        }
    }

    /**
     * 执行应用启动命令
     * @param currentItem 当前指令项
     * @param currentAction 当前动作信息
     * @param responseBuilder 响应构建器
     * @return 是否执行成功
     */
    private boolean executeAppStartCommand(JSONObject currentItem, JSONObject currentAction, StringBuilder responseBuilder) {
        try {
            String packageName = null;
            
            // 调试日志：显示currentAction的状态
            Log.d("MainActivity", "执行应用启动命令 - currentAction: " + (currentAction != null ? currentAction.toString() : "null"));
            
            // 从parameters中获取包名
            if (currentItem != null && currentItem.has("function")) {
                JSONObject function = currentItem.getJSONObject("function");
                if (function.has("parameters")) {
                    JSONObject parameters = function.getJSONObject("parameters");
                    packageName = parameters.optString("package_name", null);
                    Log.d("MainActivity", "从parameters获取应用包名: " + packageName);
                }
            }
            
            
            try {
                // 调用monitoru2的appStart方法
                monitoru2.appStart(this, packageName);
                Log.d("MainActivity", "成功调用monitoru2.appStart: " + packageName);
                responseBuilder.append(" ✅ 启动应用: ").append(packageName);
                return true;
            } catch (Exception e) {
                String errorMsg = e.getMessage();
                if (errorMsg != null && errorMsg.startsWith("SUCCESS:")) {
                    // 这是成功消息，不是真正的异常
                    Log.d("MainActivity", "应用启动成功: " + errorMsg);
                    responseBuilder.append(" ✅ ").append(errorMsg.substring(8)); // 去掉"SUCCESS:"前缀
                    return true;
                } else {
                    Log.e("MainActivity", "启动应用失败: " + packageName, e);
                    responseBuilder.append(" ❌ 启动失败: ").append(errorMsg != null ? errorMsg : "未知错误");
                    return false;
                }
            }
            
        } catch (JSONException e) {
            Log.e("MainActivity", "解析appStart命令参数失败", e);
            responseBuilder.append(" ❌ 参数解析失败");
            return false;
        } catch (Exception e) {
            Log.e("MainActivity", "executeAppStartCommand异常", e);
            responseBuilder.append(" ❌ 执行异常: ").append(e.getMessage());
            return false;
        }
    }

    // ==================== 线程管理方法 ====================
    
    /**
     * 添加线程到活跃线程列表
     */
    private void addActiveThread(Thread thread) {
        synchronized (threadLock) {
            activeThreads.add(thread);
            Log.d("MainActivity", "添加活跃线程，当前活跃线程数: " + activeThreads.size());
        }
    }
    
    /**
     * 从活跃线程列表移除线程
     */
    private void removeActiveThread(Thread thread) {
        synchronized (threadLock) {
            activeThreads.remove(thread);
            Log.d("MainActivity", "移除活跃线程，当前活跃线程数: " + activeThreads.size());
        }
    }
    
    /**
     * 中断所有活跃线程
     */
    private void interruptAllActiveThreads() {
        synchronized (threadLock) {
            Log.d("MainActivity", "开始中断所有活跃线程，总数: " + activeThreads.size());
            
            for (Thread thread : new ArrayList<>(activeThreads)) {
                if (thread != null && thread.isAlive()) {
                    try {
                        thread.interrupt();
                        Log.d("MainActivity", "已中断线程: " + thread.getName());
                    } catch (Exception e) {
                        Log.e("MainActivity", "中断线程失败: " + thread.getName(), e);
                    }
                }
            }
            
            // 清空线程列表
            activeThreads.clear();
            Log.d("MainActivity", "所有活跃线程已清空");
        }
    }
    
    /**
     * 清理并重置所有状态 - 为done指令使用
     */
    private void resetAllStates() {
        Log.d("MainActivity", "=== 开始重置所有状态 ===");
        
        // 1. 设置强制中断标志
        isTaskInterrupted = true;
        
        // 2. 中断所有线程
        interruptAllActiveThreads();
        
        // 3. 重置API线程
        if (currentApiThread != null) {
            currentApiThread.interrupt();
            currentApiThread = null;
        }
        
        // 4. 清空历史记录
        historyList = new JSONArray();
        
        // 5. 重置命令相关变量
        functionName = null;
        currentItem = null;
        currentAction = null;
        
        // 6. 重置截图状态
        lastScreenshotPath = null;
        
        // 7. 重置TTS状态
        // 8. 立即重置中断标志，允许新任务执行
        // 注意：这里不再延迟重置，因为所有必要的清理已经完成
        isTaskInterrupted = false;
        Log.d("MainActivity", "中断标志已立即重置，系统可以接受新任务");
        
        Log.d("MainActivity", "=== 所有状态重置完成 ===");
    }

    /**
     * 显示悬浮窗信息
     * @param message 要显示的消息内容
     */
    private void showFloatingWindow(String message) {
        // 创建一个自定义的Dialog作为悬浮窗
        android.app.Dialog dialog = new android.app.Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen);
        dialog.setCancelable(true);
        dialog.setCanceledOnTouchOutside(true);
        
        // 创建悬浮窗的布局
        android.widget.RelativeLayout layout = new android.widget.RelativeLayout(this);
        layout.setBackgroundColor(android.graphics.Color.parseColor("#80000000")); // 半透明黑色背景
        
        // 创建内容容器
        android.widget.LinearLayout contentLayout = new android.widget.LinearLayout(this);
        contentLayout.setOrientation(android.widget.LinearLayout.VERTICAL);
        contentLayout.setBackgroundResource(android.R.drawable.dialog_frame);
        contentLayout.setPadding(40, 40, 40, 40);
        
        // 创建文本视图
        android.widget.TextView textView = new android.widget.TextView(this);
        textView.setText(message);
        textView.setTextColor(android.graphics.Color.WHITE);
        textView.setTextSize(18);
        textView.setGravity(android.view.Gravity.CENTER);
        textView.setLineSpacing(8, 1.2f); // 设置行间距
        
        // 创建关闭按钮
        android.widget.Button closeButton = new android.widget.Button(this);
        closeButton.setText("关闭");
        closeButton.setTextColor(android.graphics.Color.WHITE);
        closeButton.setBackgroundColor(android.graphics.Color.parseColor("#6200EA"));
        closeButton.setPadding(20, 10, 20, 10);
        closeButton.setOnClickListener(v -> dialog.dismiss());
        
        // 添加视图到容器
        contentLayout.addView(textView);
        
        // 为关闭按钮添加布局参数
        android.widget.LinearLayout.LayoutParams buttonParams = new android.widget.LinearLayout.LayoutParams(
            android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
            android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
        );
        buttonParams.gravity = android.view.Gravity.CENTER;
        buttonParams.topMargin = 30;
        contentLayout.addView(closeButton, buttonParams);
        
        // 设置内容容器的布局参数
        android.widget.RelativeLayout.LayoutParams contentParams = new android.widget.RelativeLayout.LayoutParams(
            android.widget.RelativeLayout.LayoutParams.MATCH_PARENT,
            android.widget.RelativeLayout.LayoutParams.WRAP_CONTENT
        );
        contentParams.addRule(android.widget.RelativeLayout.CENTER_IN_PARENT);
        contentParams.setMargins(50, 100, 50, 100);
        
        layout.addView(contentLayout, contentParams);
        dialog.setContentView(layout);
        
        // 显示悬浮窗
        dialog.show();
    }

    // ==================== 设置随机提示文本方法 ====================
    private void setRandomHint() {
        String[] hints = {
            "例: 用饿了么点一杯糯香柠檬茶",
            "例: 用B站播放波士顿圆脸的第一个视频",
            "例: 用携程在汉庭酒店订一间大床房",
            "例: 用携程在上海交通大学附近的全季酒店订一间大床房",
            "例: 用B站搜索让子弹飞",
            "例: 用饿了么搜索蜜雪冰城并点一杯冰鲜柠檬水",
            "例: 用网易云音乐搜索并播放《容易受伤的女人》",
            "例: 用网易云音乐搜索并播放 Speed of Light",
            "例: 用网易云音乐搜索并播放杯中明月",
            "例: 用高德导航导航到白金汉爵大酒店"
        };
        
        // 生成随机索引
        int randomIndex = (int) (Math.random() * hints.length);
        
        // 设置随机hint
        messageEditText.setHint(hints[randomIndex]);
    }

}