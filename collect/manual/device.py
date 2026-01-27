"""
Device abstraction layer for manual data collection.
Supports both Android and Harmony devices.
"""

import time
import base64
import logging
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Device(ABC):
    """Abstract base class for device operations"""
    
    @abstractmethod
    def start_app(self, app):
        """Start an application by app name"""
        pass
    
    @abstractmethod
    def app_stop(self, package_name):
        """Stop an application by package name"""
        pass

    @abstractmethod
    def screenshot(self, path):
        """Take a screenshot and save to path"""
        pass

    @abstractmethod
    def click(self, x, y):
        """Click at coordinates (x, y)"""
        pass

    @abstractmethod
    def input(self, text):
        """Input text to the device"""
        pass

    @abstractmethod
    def swipe(self, start_x, start_y, end_x, end_y, duration=0.1):
        """Swipe from (start_x, start_y) to (end_x, end_y)"""
        pass

    @abstractmethod
    def keyevent(self, key):
        """Send a key event"""
        pass

    @abstractmethod
    def dump_hierarchy(self):
        """Dump UI hierarchy as XML string"""
        pass


class AndroidDevice(Device):
    """Android device implementation using uiautomator2"""
    
    def __init__(self, adb_endpoint=None):
        """Initialize Android device connection
        
        Args:
            adb_endpoint: ADB endpoint URL (default: auto-detect)
        """
        import uiautomator2 as u2
        
        try:
            if adb_endpoint:
                self.d = u2.connect(adb_endpoint)
            else:
                self.d = u2.connect()
            logger.info("Android device connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect Android device: {e}")
            raise
        
        self.device_type = "Android"
        
        # App package name mappings
        self.app_package_names = {
            "微信": "com.tencent.mm",
            "QQ": "com.tencent.mobileqq",
            "微博": "com.sina.weibo",
            
            "饿了么": "me.ele",
            "美团": "com.sankuai.meituan",

            "bilibili": "tv.danmaku.bili",
            "爱奇艺": "com.qiyi.video",
            "腾讯视频": "com.tencent.qqlive",
            "优酷": "com.youku.phone",

            "淘宝": "com.taobao.taobao",
            "京东": "com.jingdong.app.mall",

            "携程": "ctrip.android.view",
            "同城": "com.tongcheng.android",
            "飞猪": "com.taobao.trip",
            "去哪儿": "com.Qunar",
            "华住会": "com.htinns",

            "知乎": "com.zhihu.android",
            "小红书": "com.xingin.xhs",

            "QQ音乐": "com.tencent.qqmusic",
            "网易云音乐": "com.netease.cloudmusic",
            "酷狗音乐": "com.kugou.android",

            "高德地图": "com.autonavi.minimap",
            "华为商城": "com.vmall.client"
        }

    def start_app(self, app):
        """Start an application by app name"""
        package_name = self.app_package_names.get(app)
        if not package_name:
            logger.warning(f"App '{app}' not registered, trying direct package name")
            package_name = app
        
        self.d.app_start(package_name, stop=True)
        time.sleep(1)
        if not self.d.app_wait(package_name, timeout=10):
            logger.warning(f"App '{app}' may not have started correctly")

    def unlock(self):
        """Unlock the Android device"""
        try:
            logger.info("Attempting to unlock Android device")
            self.d.unlock()
            time.sleep(1)
            logger.info("Android device unlocked successfully")
        except Exception as e:
            logger.error(f"Failed to unlock Android device: {e}")
            raise

    def get_deviceinfo(self):
        """Get Android device information"""

        return self.d.device_info

    def app_stop(self, package_name):
        """Stop an application"""
        self.d.app_stop(package_name)

    def screenshot(self, path):
        """Take a screenshot"""
        self.d.screenshot(path)

    def click(self, x, y):
        """Click at coordinates"""
        self.d.click(x, y)
        time.sleep(0.5)

    def input(self, text):
        """Input text to the device using ADB IME"""
        try:
            logger.info(f"Inputting text to Android device: '{text}'")
            
            # Try to use current IME if available
            current_ime = None
            try:
                current_ime = self.d.current_ime()
                logger.debug(f"Current IME: {current_ime}")
            except (AttributeError, Exception) as e:
                # If current_ime is not available, proceed without saving/restoring
                logger.warning(f"current_ime() not available: {e}, skipping IME preservation")
            
            # Set to ADB keyboard
            try:
                self.d.shell(['settings', 'put', 'secure', 'default_input_method', 'com.android.adbkeyboard/.AdbIME'])
                time.sleep(0.5)
                logger.debug("ADB keyboard set successfully")
            except Exception as e:
                logger.warning(f"Failed to set ADB keyboard: {e}")
                raise
            
            # Encode and send text
            try:
                charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
                self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
                time.sleep(0.5)
                logger.info(f"Text sent successfully: '{text}'")
            except Exception as e:
                logger.error(f"Failed to send text via ADB: {e}")
                raise
            
            # Restore previous IME if we saved it
            if current_ime:
                try:
                    self.d.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
                    time.sleep(0.5)
                    logger.debug(f"IME restored: {current_ime}")
                except Exception as e:
                    logger.warning(f"Failed to restore IME: {e}")
                    
        except Exception as e:
            logger.error(f"Input text operation failed: {e}")
            raise

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.1):
        """Swipe from start to end position"""
        self.d.swipe(start_x, start_y, end_x, end_y, duration=duration)

    def keyevent(self, key):
        """Send a key event"""
        self.d.keyevent(key)

    def dump_hierarchy(self):
        """Dump UI hierarchy"""
        return self.d.dump_hierarchy()


class HarmonyDevice(Device):
    """Harmony device implementation using hmdriver2"""

    def __init__(self, serial = None):
        """Initialize Harmony device connection"""
        from hmdriver2.driver import Driver
        
        try:
            self.d = Driver(serial=serial)
            logger.info("Harmony device connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect Harmony device: {e}")
            raise
        
        self.device_type = "Harmony"
        
        # App package name mappings for Harmony
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
            "微博": "com.sina.weibo.stage",
            "饿了么": "me.ele.eleme",
            "拼多多": "com.xunmeng.pinduoduo.hos",
            "华为商城": "com.huawei.hmos.vmall"
        }

    def start_app(self, app):
        """Start an application by app name"""
        package_name = self.app_package_names.get(app)
        if not package_name:
            logger.warning(f"App '{app}' not registered, trying direct package name")
            package_name = app
        
        # self.d.start_app(package_name)
        self.d.force_start_app(package_name)
        time.sleep(1.5)

    def get_deviceinfo(self):
        """Get Android device information"""
        
        return self.d.device_info
    
    def unlock(self):
        """Unlock the Harmony device"""
        try:
            logger.info("Attempting to unlock Harmony device")
            self.d.unlock()
            time.sleep(1)
            logger.info("Harmony device unlocked successfully")
        except Exception as e:
            logger.error(f"Failed to unlock Harmony device: {e}")
            raise

    def app_stop(self, package_name):
        """Stop an application"""
        self.d.stop_app(package_name)

    def screenshot(self, path):
        """Take a screenshot"""
        self.d.screenshot(path)

    def click(self, x, y):
        """Click at coordinates"""
        self.d.click(x, y)
        time.sleep(0.5)

    def input(self, text):
        """Input text to the device"""
        try:
            logger.info(f"Inputting text to Harmony device: '{text}'")
            
            # Log available methods for debugging
            available_methods = [m for m in dir(self.d) if not m.startswith('_')]
            logger.debug(f"Available driver methods: {available_methods}")
            
            # Try different input methods for Harmony
            methods_tried = []
            
            # Method 1: Try input_text()
            if hasattr(self.d, 'input_text'):
                try:
                    logger.debug("Trying input_text() method")
                    self.d.input_text(text)
                    time.sleep(0.5)
                    logger.info(f"Text successfully sent via input_text(): '{text}'")
                    return
                except Exception as e:
                    logger.warning(f"input_text() failed: {e}")
                    methods_tried.append(f"input_text: {str(e)}")
            
            # Method 2: Try send_text()
            if hasattr(self.d, 'send_text'):
                try:
                    logger.debug("Trying send_text() method")
                    self.d.send_text(text)
                    time.sleep(0.5)
                    logger.info(f"Text successfully sent via send_text(): '{text}'")
                    return
                except Exception as e:
                    logger.warning(f"send_text() failed: {e}")
                    methods_tried.append(f"send_text: {str(e)}")
            
            # Method 3: Try typing()
            if hasattr(self.d, 'typing'):
                try:
                    logger.debug("Trying typing() method")
                    self.d.typing(text)
                    time.sleep(0.5)
                    logger.info(f"Text successfully sent via typing(): '{text}'")
                    return
                except Exception as e:
                    logger.warning(f"typing() failed: {e}")
                    methods_tried.append(f"typing: {str(e)}")
            
            # Method 4: Try press_key for character-by-character input
            if hasattr(self.d, 'press_key'):
                try:
                    logger.debug("Trying press_key() for character-by-character input")
                    # This is slower but more reliable
                    for char in text:
                        self.d.press_key(char)
                        time.sleep(0.1)
                    time.sleep(0.5)
                    logger.info(f"Text successfully sent via press_key(): '{text}'")
                    return
                except Exception as e:
                    logger.warning(f"press_key() failed: {e}")
                    methods_tried.append(f"press_key: {str(e)}")
            
            # If all methods failed
            error_msg = f"All input methods failed. Tried: {'; '.join(methods_tried)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except Exception as e:
            logger.error(f"Input text operation failed on Harmony device: {e}", exc_info=True)
            raise

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.1):
        """Swipe from start to end position"""
        # Harmony device uses a different swipe interface
        # Calculate direction based on coordinates
        dx = end_x - start_x
        dy = end_y - start_y
        
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'
        
        self.d.swipe_ext(direction, scale=0.5)

    def keyevent(self, key):
        """Send a key event"""
        self.d.press_key(key)

    def dump_hierarchy(self):
        """Dump UI hierarchy"""
        return self.d.dump_hierarchy()


def create_device(device_type: str = "Android", adb_endpoint: str = None) -> Device:
    """Factory function to create device instance
    
    Args:
        device_type: "Android" or "Harmony"
        adb_endpoint: ADB endpoint for Android device (optional)
        
    Returns:
        Device instance
        
    Raises:
        ValueError: If device_type is not supported
    """
    if device_type == "Android":
        return AndroidDevice(adb_endpoint)
    elif device_type == "Harmony":
        return HarmonyDevice(adb_endpoint)
    else:
        raise ValueError(f"Unsupported device type: {device_type}")
