import sys
import cv2
import numpy as np
import os
import platform
import qtawesome as qta
from PIL import Image, ImageDraw, ImageFont 
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QStackedWidget, QMessageBox, QFrame, QSizePolicy,
                             QGraphicsDropShadowEffect, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QIcon
from ultralytics import YOLO

# --- 配色方案 ---
COLOR_SIDEBAR_BG = "#FFF085"  
COLOR_ACCENT = "#FCB454"      
COLOR_ACCENT_HOVER = "#FF9B17"
COLOR_DANGER = "#F16767"      
COLOR_BG_MAIN = "#FAFAFA"     
COLOR_TEXT_MAIN = "#333333"   

# --- 字体路径智能查找工具 ---
def get_font_path():
    """智能查找适合 WSL/Linux/Windows 的中文字体路径"""
    system = platform.system()
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

# --- 全局设置 ---
class GlobalSettings:
    model_path = 'best.pt'
    confidence_threshold = 0.5
    
    @staticmethod
    def get_model():
        try:
            return YOLO(GlobalSettings.model_path)
        except Exception:
            print("Warning: model not found, downloading yolov8n.pt...")
            return YOLO("yolov8n.pt")

# --- 中文绘制工具类 ---
class cv2ImgAddText:
    @staticmethod
    def draw_text(img, text, left, top, text_color=(255, 255, 255), text_size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font_path = get_font_path()
        try:
            font = ImageFont.truetype(font_path, text_size) if font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        draw.text((left, top), text, fill=text_color, font=font)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# --- 视频处理线程 ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(int, int)

    def __init__(self, source=0, is_video=False):
        super().__init__()
        self.source = source
        self.running = True
        self.model = GlobalSettings.get_model()
        # self.unique_ids = set()
        self.tracked_objects = {}  # 添加：{id: {'last_seen': frame_num, 'position': (x,y)}}
        self.confirmed_count = 0   # 添加：真实累计数
        self.frame_count = 0        # 添加：帧计数器
        self.is_video = is_video

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Cannot open source: {self.source}")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1  # 添加：增加帧计数
            
            # YOLO Track
            results = self.model.track(
                frame, 
                persist=True, 
                verbose=True,
                conf=GlobalSettings.confidence_threshold,
                tracker='bytetrack.yaml'
            )
            current_count = 0
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xywh.cpu().numpy()  # 添加：获取位置信息
                current_count = len(ids)
                annotated_frame = results[0].plot()
                
                # 智能去重逻辑
                current_ids = set()
                for i, obj_id in enumerate(ids):
                    current_ids.add(obj_id)
                    x, y = boxes[i][0], boxes[i][1]  # 获取中心点坐标
                    
                    # 检查是否是新物体
                    if obj_id not in self.tracked_objects:
                        # 检查附近是否有最近消失的物体（可能是同一个橘子重新获得的ID）
                        is_new = True
                        for old_id, info in list(self.tracked_objects.items()):
                            # 如果某个旧ID在30帧内消失，且位置接近
                            if old_id not in current_ids and self.frame_count - info['last_seen'] < 30:
                                dist = ((x - info['position'][0])**2 + 
                                    (y - info['position'][1])**2)**0.5
                                if dist < 100:  # 距离阈值（根据实际画面大小调整）
                                    is_new = False
                                    break
                        
                        if is_new:
                            self.confirmed_count += 1
                    
                    # 更新追踪信息
                    self.tracked_objects[obj_id] = {
                        'last_seen': self.frame_count,
                        'position': (x, y)
                    }
                
                # 清理长时间未见的ID（避免内存泄漏）
                to_remove = []
                for obj_id, info in self.tracked_objects.items():
                    if self.frame_count - info['last_seen'] > 60:  # 60帧未见则清理
                        to_remove.append(obj_id)
                for obj_id in to_remove:
                    del self.tracked_objects[obj_id]
                    
            else:
                annotated_frame = frame

            # 使用智能累计计数
            cumulative_count = self.confirmed_count
            
            # 使用新的Apple风格dashboard
            if self.is_video:
                annotated_frame = self.draw_apple_dashboard(annotated_frame, current_count, cumulative_count, mode='video')
            else:
                annotated_frame = self.draw_apple_dashboard(annotated_frame, current_count, cumulative_count, mode='camera')

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(qt_img)
            self.stats_signal.emit(current_count, cumulative_count)

        cap.release()
    # def run(self):
    #     cap = cv2.VideoCapture(self.source)
    #     if not cap.isOpened():
    #         print(f"Cannot open source: {self.source}")
    #         self.running = False
    #         return

    #     while self.running:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # YOLO Track - 移除有问题的tracker_conf参数
    #         results = self.model.track(
    #             frame, 
    #             persist=True, 
    #             verbose=True,
    #             conf=GlobalSettings.confidence_threshold,
    #             tracker='bytetrack.yaml'
    #         )
    #         current_count = 0
            
    #         if results[0].boxes is not None and results[0].boxes.id is not None:
    #             ids = results[0].boxes.id.cpu().numpy().astype(int)
    #             current_count = len(ids)
    #             annotated_frame = results[0].plot() 
    #             for obj_id in ids:
    #                 self.unique_ids.add(obj_id)
    #         else:
    #             annotated_frame = frame

    #         cumulative_count = len(self.unique_ids)
            
    #         # 使用新的Apple风格dashboard
    #         if self.is_video:
    #             annotated_frame = self.draw_apple_dashboard(annotated_frame, current_count, cumulative_count, mode='video')
    #         else:
    #             annotated_frame = self.draw_apple_dashboard(annotated_frame, current_count, cumulative_count, mode='camera')

    #         rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    #         h, w, ch = rgb_image.shape
    #         qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
    #         self.change_pixmap_signal.emit(qt_img)
    #         self.stats_signal.emit(current_count, cumulative_count)

    #     cap.release()

    def draw_apple_dashboard(self, img, current, total, mode='camera'):
        """Apple风格的优雅仪表板"""
        h, w, _ = img.shape
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        
        # 背景面板 - 使用半透明玻璃态效果
        panel_w, panel_h = 320, 160
        x_start = w - panel_w - 40
        y_start = 40
        
        # 绘制主面板背景（半透明白色+圆角模拟）
        draw.rectangle(
            [x_start, y_start, x_start + panel_w, y_start + panel_h],
            fill=(255, 255, 255, 200),  # 半透明白色
            outline=(220, 220, 220, 100)
        )
        
        # 添加软阴影效果（通过多层半透明矩形模拟）
        for offset in range(3, 0, -1):
            draw.rectangle(
                [x_start + offset, y_start + offset, 
                 x_start + panel_w + offset, y_start + panel_h + offset],
                fill=(0, 0, 0, 5)
            )
        
        font_path = get_font_path()
        try:
            f_label = ImageFont.truetype(font_path, 24) if font_path else ImageFont.load_default()
            f_number = ImageFont.truetype(font_path, 56) if font_path else ImageFont.load_default()
            f_small = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
        except:
            f_label = ImageFont.load_default()
            f_number = ImageFont.load_default()
            f_small = ImageFont.load_default()
        
        # 左侧：当前计数
        left_x = x_start + 30
        draw.text((left_x, y_start + 20), "当前", font=f_label, fill=(100, 100, 100, 255))
        draw.text((left_x, y_start + 60), str(current), font=f_number, fill=(255, 155, 23, 255))
        
        # 右侧：累计计数
        right_x = x_start + 190
        draw.text((right_x, y_start + 20), "累计", font=f_label, fill=(100, 100, 100, 255))
        draw.text((right_x, y_start + 60), str(total), font=f_number, fill=(76, 144, 234, 255))

        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    def stop(self):
        self.running = False
        self.wait()

# --- UI 组件 (Apple Style + qtawesome) ---
class ModernButton(QPushButton):
    def __init__(self, text, icon_name=None, is_danger=False):
        super().__init__(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(45)
        
        if icon_name:
            self.setIcon(qta.icon(icon_name, color='white'))
            self.setIconSize(QSize(20, 20))

        bg_color = COLOR_DANGER if is_danger else COLOR_ACCENT
        hover_color = "#D64545" if is_danger else COLOR_ACCENT_HOVER
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                border-radius: 12px;
                font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
                font-size: 15px;
                font-weight: bold;
                padding: 0 20px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                margin-top: 2px;
            }}
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

class NavButton(QPushButton):
    def __init__(self, text, icon_name, index, callback):
        super().__init__(text)
        self.index = index
        self.callback = callback
        self.icon_name = icon_name
        
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setFixedHeight(55)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(lambda: self.callback(self.index))
        
        self.update_style(False)

    def update_style(self, active):
        font_family = '"Noto Sans CJK SC", "Microsoft YaHei", sans-serif'
        
        if active:
            icon = qta.icon(self.icon_name, color=COLOR_ACCENT_HOVER)
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: #FFFFFF;
                    color: {COLOR_ACCENT_HOVER};
                    border: none;
                    border-radius: 12px;
                    text-align: left;
                    padding-left: 20px;
                    font-family: {font_family};
                    font-size: 16px;
                    font-weight: bold;
                }}
            """)
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(10)
            shadow.setColor(QColor(0, 0, 0, 20))
            shadow.setOffset(0, 2)
            self.setGraphicsEffect(shadow)
        else:
            icon = qta.icon(self.icon_name, color="#666666")
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: #555555;
                    border: none;
                    border-radius: 12px;
                    text-align: left;
                    padding-left: 20px;
                    font-family: {font_family};
                    font-size: 16px;
                }}
                QPushButton:hover {{
                    background-color: rgba(255, 255, 255, 0.4);
                    color: {COLOR_TEXT_MAIN};
                }}
            """)
            self.setGraphicsEffect(None)
            
        self.setIcon(icon)
        self.setIconSize(QSize(22, 22))

# --- 主程序 ---
class CitrusApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Citrus AI Vision")
        self.resize(1200, 800)
        self.thread = None
        self.nav_btns = []

        self.init_ui()
        self.setStyleSheet(f"background-color: {COLOR_BG_MAIN};")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QFrame()
        self.sidebar.setStyleSheet(f"background-color: {COLOR_SIDEBAR_BG}; border: none;")
        self.sidebar.setFixedWidth(260)
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(20, 50, 20, 30)
        sidebar_layout.setSpacing(15)

        # Title Area
        title_box = QHBoxLayout()
        logo_label = QLabel()
        title_text = QLabel("Citrus\nVision Pro")
        title_text.setStyleSheet(f"""
            font-family: "Noto Sans CJK SC", "Microsoft YaHei"; 
            font-size: 24px; 
            font-weight: 800; 
            color: {COLOR_ACCENT_HOVER};
        """)
        title_box.addWidget(logo_label)
        title_box.addWidget(title_text)
        title_box.addStretch()
        
        sidebar_layout.addLayout(title_box)
        sidebar_layout.addSpacing(20)

        # Navigation Items
        self.create_nav_item("实时监控", "fa5s.camera", 0, sidebar_layout)
        self.create_nav_item("图片识别", "fa5s.image", 1, sidebar_layout)
        self.create_nav_item("视频追踪", "fa5s.video", 2, sidebar_layout)
        
        sidebar_layout.addStretch()
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: rgba(0,0,0,0.05);")
        sidebar_layout.addWidget(line)
        self.create_nav_item("系统设置", "fa5s.cog", 3, sidebar_layout)

        # Content
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(30, 30, 30, 30)
        
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("""
            QStackedWidget {
                background-color: white;
                border-radius: 20px;
                border: 1px solid #EAEAEA;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 10)
        self.stacked_widget.setGraphicsEffect(shadow)

        self.stacked_widget.addWidget(self.create_camera_page())
        self.stacked_widget.addWidget(self.create_image_page())
        self.stacked_widget.addWidget(self.create_video_page())
        self.stacked_widget.addWidget(self.create_settings_page())

        content_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(content_container)
        
        self.nav_btns[0].setChecked(True)
        self.nav_btns[0].update_style(True)

    def create_nav_item(self, text, icon_name, index, layout):
        btn = NavButton(text, icon_name, index, self.switch_page)
        layout.addWidget(btn)
        self.nav_btns.append(btn)

    def switch_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_btns):
            btn.update_style(i == index)
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None
        self.camera_label.clear()
        self.video_label.clear()

    # --- 辅助 UI ---
    def create_header(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            font-family: "Noto Sans CJK SC", "Microsoft YaHei";
            font-size: 24px; font-weight: bold; color: {COLOR_TEXT_MAIN}; margin-bottom: 20px;
        """)
        return lbl

    def create_display_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("""
            background-color: #F5F5F7; border-radius: 15px; color: #AAA; font-size: 16px;
            font-family: "Noto Sans CJK SC", "Microsoft YaHei";
        """)
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return lbl

    # --- 页面定义 ---
    def create_camera_page(self):
        page = QWidget(); page.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(page); layout.setContentsMargins(40,40,40,40)
        layout.addWidget(self.create_header("实时监控中心"))
        self.camera_label = self.create_display_label("摄像头待机中")
        layout.addWidget(self.camera_label)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_start = ModernButton("启动摄像头", "fa5s.play")
        btn_start.clicked.connect(self.start_camera)
        btn_stop = ModernButton("停止监控", "fa5s.stop", is_danger=True)
        btn_stop.clicked.connect(self.stop_thread)
        
        btn_layout.addWidget(btn_start); btn_layout.addSpacing(15); btn_layout.addWidget(btn_stop)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        return page

    def create_image_page(self):
        page = QWidget(); page.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(page); layout.setContentsMargins(40,40,40,40)
        layout.addWidget(self.create_header("智能图片分析"))
        self.image_display = self.create_display_label("拖入图片或点击上传")
        layout.addWidget(self.image_display)
        
        btn_layout = QHBoxLayout(); btn_layout.addStretch()
        btn_upload = ModernButton("上传本地图片", "fa5s.upload")
        btn_upload.clicked.connect(self.upload_image)
        btn_layout.addWidget(btn_upload); btn_layout.addStretch()
        layout.addLayout(btn_layout)
        return page

    def create_video_page(self):
        page = QWidget(); page.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(page); layout.setContentsMargins(40,40,40,40)
        layout.addWidget(self.create_header("视频流追踪分析"))
        self.video_label = self.create_display_label("请导入视频文件")
        layout.addWidget(self.video_label)
        
        ctrl_layout = QHBoxLayout(); ctrl_layout.addStretch()
        btn_load = ModernButton("导入视频", "fa5s.folder-open")
        btn_load.clicked.connect(self.load_video)
        btn_stop = ModernButton("停止追踪", "fa5s.stop", is_danger=True)
        btn_stop.clicked.connect(self.stop_thread)
        ctrl_layout.addWidget(btn_load); ctrl_layout.addSpacing(15); ctrl_layout.addWidget(btn_stop); ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        return page

    def create_settings_page(self):
        page = QWidget(); page.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(page); layout.setContentsMargins(50,50,50,50)
        layout.addWidget(self.create_header("全局设置"))
        
        # 模型配置卡片
        card1 = QFrame()
        card1.setStyleSheet("background: #F9F9F9; border-radius: 15px; padding: 20px;")
        c_layout1 = QVBoxLayout(card1)
        
        lbl_t = QLabel("YOLO 模型权重配置")
        lbl_t.setStyleSheet("font-size:18px; font-weight:bold; color:#333; font-family:'Noto Sans CJK SC';")
        self.lbl_model_path = QLabel(f"当前路径: {GlobalSettings.model_path}")
        self.lbl_model_path.setStyleSheet("color:#666; margin-top:5px; font-family:monospace; font-size:12px;")
        
        btn_change = ModernButton("切换模型文件 (.pt)", "fa5s.file-code")
        btn_change.setFixedWidth(240)
        btn_change.clicked.connect(self.change_model_path)
        
        c_layout1.addWidget(lbl_t); c_layout1.addWidget(self.lbl_model_path)
        c_layout1.addSpacing(15); c_layout1.addWidget(btn_change)
        
        # 阈值配置卡片
        card2 = QFrame()
        card2.setStyleSheet("background: #F9F9F9; border-radius: 15px; padding: 20px;")
        c_layout2 = QVBoxLayout(card2)
        
        lbl_thresh = QLabel("检测阈值配置")
        lbl_thresh.setStyleSheet("font-size:18px; font-weight:bold; color:#333; font-family:'Noto Sans CJK SC';")
        
        conf_layout = QHBoxLayout()
        conf_label = QLabel("检测置信度:")
        conf_label.setStyleSheet("color:#666; font-family:'Noto Sans CJK SC'; min-width: 100px;")
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(0, 100)
        self.conf_spin.setValue(int(GlobalSettings.confidence_threshold * 100))
        self.conf_spin.setSuffix("%")
        self.conf_spin.setStyleSheet("padding: 5px; border-radius: 5px; border: 1px solid #DDD;")
        self.conf_spin.valueChanged.connect(self.update_conf_threshold)
        conf_layout.addWidget(conf_label); conf_layout.addWidget(self.conf_spin); conf_layout.addStretch()
        
        c_layout2.addWidget(lbl_thresh)
        c_layout2.addSpacing(10)
        c_layout2.addLayout(conf_layout)
        
        layout.addWidget(card1)
        layout.addSpacing(20)
        layout.addWidget(card2)
        layout.addStretch()
        return page

    def update_conf_threshold(self, value):
        GlobalSettings.confidence_threshold = value / 100.0

    # --- Actions ---
    def start_camera(self):
        self.stop_thread()
        self.thread = VideoThread(source=0, is_video=False)
        self.thread.change_pixmap_signal.connect(self.update_image_label(self.camera_label))
        self.thread.start()

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.stop_thread()
            self.thread = VideoThread(source=file_path, is_video=True)
            self.thread.change_pixmap_signal.connect(self.update_image_label(self.video_label))
            self.thread.start()

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            model = GlobalSettings.get_model()
            results = model.predict(file_path, conf=GlobalSettings.confidence_threshold)
            res_plotted = results[0].plot()
            
            # 计数
            if results[0].boxes is not None:
                count = len(results[0].boxes)
            else:
                count = 0
            
            # 在图上加计数面板
            final_img = self.draw_image_dashboard(res_plotted, count)
            
            rgb_image = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(self.image_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_display.setPixmap(pixmap)

    def draw_image_dashboard(self, img, count):
        """Apple风格的优雅仪表板 - 图片版（单计数）"""
        h, w, _ = img.shape
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        
        # 背景面板
        panel_w, panel_h = 320, 160
        x_start = w - panel_w - 40
        y_start = 40
        
        # 绘制主面板背景
        draw.rectangle(
            [x_start, y_start, x_start + panel_w, y_start + panel_h],
            fill=(255, 255, 255, 200),
            outline=(220, 220, 220, 100)
        )
        
        # 软阴影效果
        for offset in range(3, 0, -1):
            draw.rectangle(
                [x_start + offset, y_start + offset, 
                 x_start + panel_w + offset, y_start + panel_h + offset],
                fill=(0, 0, 0, 5)
            )
        
        font_path = get_font_path()
        try:
            f_label = ImageFont.truetype(font_path, 56) if font_path else ImageFont.load_default()
            f_number = ImageFont.truetype(font_path, 86) if font_path else ImageFont.load_default()
        except:
            f_label = ImageFont.load_default()
            f_number = ImageFont.load_default()

        # 标签和数字
        draw.text((x_start + 30, y_start + 15), "检测数量", font=f_label, fill=(100, 100, 100, 255))
        draw.text((x_start + 30, y_start + 50), str(count), font=f_number, fill=(255, 155, 23, 255))

        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    def change_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择权重", "", "Models (*.pt)")
        if file_path:
            GlobalSettings.model_path = file_path
            self.lbl_model_path.setText(f"Path: {file_path}")

    def stop_thread(self):
        if self.thread:
            self.thread.stop()
            self.thread = None

    def update_image_label(self, label_widget):
        def func(qt_img):
            label_widget.setPixmap(QPixmap.fromImage(qt_img).scaled(
                label_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        return func

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font_path = get_font_path()
    if font_path:
        font = QFont("Noto Sans CJK SC", 10)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        app.setFont(font)
    
    window = CitrusApp()
    window.show()
    sys.exit(app.exec())