"""
设备抽象层
提供 Android 和 HarmonyOS 设备的统一接口
"""
import time
import base64
from abc import ABC, abstractmethod
import uiautomator2 as u2
from hmdriver2.driver import Driver


class Device(ABC):
    """设备抽象基类"""
    
    @abstractmethod
    def start_app(self, app):
        """启动应用（通过应用名）"""
        pass
    
    @abstractmethod
    def app_start(self, package_name):
        """启动应用（通过包名）"""
        pass
    
    @abstractmethod
    def app_stop(self, package_name):
        """停止应用"""
        pass

    @abstractmethod
    def screenshot(self, path):
        """截图"""
        pass

    @abstractmethod
    def click(self, x, y):
        """点击坐标"""
        pass

    @abstractmethod
    def long_click(self, x, y):
        """长按坐标"""
        pass

    @abstractmethod
    def double_click(self, x, y):
        """双击坐标"""
        pass

    @abstractmethod
    def input(self, text):
        """输入文本"""
        pass

    @abstractmethod
    def swipe(self, direction):
        """滑动（方向：up/down/left/right）"""
        pass

    @abstractmethod
    def swipe_with_coords(self, start_x, start_y, end_x, end_y):
        """使用坐标滑动"""
        pass

    @abstractmethod
    def keyevent(self, key):
        """按键事件"""
        pass

    @abstractmethod
    def dump_hierarchy(self):
        """获取UI层级"""
        pass


class AndroidDevice(Device):
    """Android 设备实现"""
    
    def __init__(self, adb_endpoint=None):
        super().__init__()
        if adb_endpoint:
            self.d = u2.connect(adb_endpoint)
        else:
            # 默认连接第一个设备
            self.d = u2.connect()
        
        # 常用应用的包名映射
        self.app_package_names = {
            "携程": "ctrip.android.view",
            "同城": "com.tongcheng.android",
            "飞猪": "com.taobao.trip",
            "去哪儿": "com.Qunar",
            "华住会": "com.htinns",
            "饿了么": "me.ele",
            "支付宝": "com.eg.android.AlipayGphone",
            "淘宝": "com.taobao.taobao",
            "京东": "com.jingdong.app.mall",
            "美团": "com.sankuai.meituan",
            "滴滴出行": "com.sdu.didi.psnger",
            "微信": "com.tencent.mm",
            "微博": "com.sina.weibo",
            "华为商城": "com.vmall.client",
            "华为视频": "com.huawei.himovie",
            "华为音乐": "com.huawei.music",
            "华为应用市场": "com.huawei.appmarket",
            "拼多多": "com.xunmeng.pinduoduo",
            "大众点评": "com.dianping.v1",
            "小红书": "com.xingin.xhs",
            "浏览器": "com.microsoft.emmx"
        }

    def start_app(self, app):
        """通过应用名启动应用"""
        package_name = self.app_package_names.get(app)
        if not package_name:
            raise ValueError(f"App '{app}' is not registered with a package name.")
        self.d.app_start(package_name, stop=True)
        time.sleep(3)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start app '{app}' with package '{package_name}'")
    
    def app_start(self, package_name):
        """通过包名启动应用"""
        self.d.app_start(package_name, stop=True)
        time.sleep(1)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start package '{package_name}'")

    def app_stop(self, package_name):
        """停止应用"""
        self.d.app_stop(package_name)

    def screenshot(self, path):
        """截图"""
        self.d.screenshot(path)

    def click(self, x, y):
        """点击坐标"""
        self.d.click(x, y)
        time.sleep(0.5)

    def long_click(self, x, y):
        """长按坐标"""
        self.d.long_click(x, y)
        time.sleep(0.5)

    def double_click(self, x, y):
        """双击坐标"""
        self.d.double_click(x, y)
        time.sleep(0.5)

    def clear_input(self):
        """清除输入框内容"""
        self.d.shell(['input', 'keyevent', 'KEYCODE_MOVE_END'])
        self.d.shell(['input', 'keyevent', 'KEYCODE_MOVE_HOME'])
        self.d.shell(['input', 'keyevent', 'KEYCODE_DEL'])

    def input(self, text):
        """输入文本（使用 ADB Keyboard）"""
        current_ime = self.d.current_ime()
        # 切换到 ADB Keyboard
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', 'com.android.adbkeyboard/.AdbIME'])
        time.sleep(0.5)
        # 清除现有文本
        self.d.shell(['am', 'broadcast', '-a', 'ADB_CLEAR_TEXT'])
        time.sleep(0.2)
        # 输入文本（使用 base64 编码支持中文）
        charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
        time.sleep(0.5)
        # 恢复原输入法
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
        time.sleep(0.5)

    def swipe(self, direction, scale=0.5):
        """滑动（方向：up/down/left/right）"""
        if direction.lower() == "up":
            self.d.swipe(0.5, 0.7, 0.5, 0.3,duration=0.5)
        elif direction.lower() == "down":
            self.d.swipe(0.5, 0.3, 0.5, 0.7,duration=0.5)
        elif direction.lower() == "left":
            self.d.swipe(0.7, 0.5, 0.3, 0.5,duration=0.5)
        elif direction.lower() == "right":
            self.d.swipe(0.3, 0.5, 0.7, 0.5,duration=0.5)

    def swipe_with_coords(self, start_x, start_y, end_x, end_y):
        """使用绝对坐标滑动"""
        self.d.swipe(start_x, start_y, end_x, end_y,duration=0.5)

    def keyevent(self, key):
        """按键事件"""
        self.d.keyevent(key)

    def dump_hierarchy(self):
        """获取UI层级（XML格式）"""
        return self.d.dump_hierarchy()


class HarmonyDevice(Device):
    """HarmonyOS 设备实现"""
    
    def __init__(self):
        super().__init__()
        self.d = Driver()
        
        # 常用应用的包名映射
        self.app_package_names = {
            "携程": "com.ctrip.harmonynext",
            "飞猪": "com.fliggy.hmos",
            "IntelliOS": "ohos.hongmeng.intellios",
            "同城": "com.tongcheng.hmos",
            "携程旅行": "com.ctrip.harmonynext",
            "饿了么": "me.ele.eleme",
            "知乎": "com.zhihu.hmos",
            "哔哩哔哩": "yylx.danmaku.bili",
            "微信": "com.tencent.wechat",
            "小红书": "com.xingin.xhs_hos",
            "QQ音乐": "com.tencent.hm.qqmusic",
            "高德地图": "com.amap.hmapp",
            "淘宝": "com.taobao.taobao4hmos",
            "微博": "com.sina.weibo.stage",
            "京东": "com.jd.hm.mall",
            "飞猪旅行": "com.fliggy.hmos",
            "天气": "com.huawei.hmsapp.totemweather",
            "什么值得买": "com.smzdm.client.hmos",
            "闲鱼": "com.taobao.idlefish4ohos",
            "慧通差旅": "com.smartcom.itravelhm",
            "PowerAgent": "com.example.osagent",
            "航旅纵横": "com.umetrip.hm.app",
            "滴滴出行": "com.sdu.didi.hmos.psnger",
            "电子邮件": "com.huawei.hmos.email",
            "图库": "com.huawei.hmos.photos",
            "日历": "com.huawei.hmos.calendar",
            "心声社区": "com.huawei.it.hmxinsheng",
            "信息": "com.ohos.mms",
            "文件管理": "com.huawei.hmos.files",
            "运动健康": "com.huawei.hmos.health",
            "智慧生活": "com.huawei.hmos.ailife",
            "豆包": "com.larus.nova.hm",
            "WeLink": "com.huawei.it.welink",
            "设置": "com.huawei.hmos.settings",
            "懂车帝": "com.ss.dcar.auto",
            "美团外卖": "com.meituan.takeaway",
            "大众点评": "com.sankuai.dianping",
            "美团": "com.sankuai.hmeituan",
            "浏览器": "com.huawei.hmos.browser",
            "拼多多": "com.xunmeng.pinduoduo.hos"
        }

    def start_app(self, app):
        """通过应用名启动应用"""
        package_name = self.app_package_names.get(app)
        if not package_name:
            raise ValueError(f"App '{app}' is not registered with a package name.")
        self.d.start_app(package_name)
        time.sleep(2)

    def app_start(self, package_name):
        """通过包名启动应用（强制启动）"""
        self.d.force_start_app(package_name)
        time.sleep(1.5)

    def app_stop(self, package_name):
        """停止应用"""
        self.d.stop_app(package_name)

    def screenshot(self, path):
        """截图"""
        self.d.screenshot(path)

    def click(self, x, y):
        """点击坐标"""
        self.d.click(x, y)
        time.sleep(0.5)

    def long_click(self, x, y):
        """长按坐标"""
        self.d.long_click(x, y)
        time.sleep(0.5)

    def double_click(self, x, y):
        """双击坐标"""
        self.d.double_click(x, y)
        time.sleep(0.5)

    def input(self, text):
        """输入文本"""
        # 触发键盘
        self.d.shell("uitest uiInput keyEvent 2072 2017")
        # 清除旧内容
        self.d.press_key(2071)
        # 输入新文本
        self.d.input_text(text)

    def swipe(self, direction, scale=0.5):
        """滑动（方向：up/down/left/right）"""
        if direction.lower() == "up":
            self.d.swipe(0.5, 0.7, 0.5, 0.3, speed=2000)
        elif direction.lower() == "down":
            self.d.swipe(0.5, 0.3, 0.5, 0.7, speed=2000)
        elif direction.lower() == "left":
            self.d.swipe(0.7, 0.5, 0.3, 0.5, speed=2000)
        elif direction.lower() == "right":
            self.d.swipe(0.3, 0.5, 0.7, 0.5, speed=2000)

    def swipe_with_coords(self, start_x, start_y, end_x, end_y):
        """使用绝对坐标滑动"""
        self.d.swipe(start_x, start_y, end_x, end_y, speed=2000)

    def keyevent(self, key):
        from hmdriver2.proto import KeyCode
        """按键事件"""
        if key == "HOME":
            self.d.go_home()
        elif key == "BACK":
            self.d.go_back()
        elif key == "RECENTS":
            self.d.press_key(KeyCode.ENTER)


        # self.d.press_key(key)

    def dump_hierarchy(self):
        """获取UI层级（JSON格式）"""
        return self.d.dump_hierarchy()


def create_device(device_type, adb_endpoint=None):
    """
    创建设备实例的工厂方法
    
    Args:
        device_type: "Android" 或 "Harmony"
        adb_endpoint: Android 设备的 ADB 端点（可选）
        
    Returns:
        Device 实例
    """
    if device_type.lower() == "android":
        return AndroidDevice(adb_endpoint)
    elif device_type.lower() == "harmony":
        return HarmonyDevice()
    else:
        raise ValueError(f"Unsupported device type: {device_type}")
