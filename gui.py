import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import os
import threading
from main import capture_face_samples, train_recognizer, recognize_face

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统")
        self.root.geometry("800x600")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建视频显示区域
        self.video_frame = ttk.LabelFrame(self.main_frame, text="视频预览")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 创建控制面板
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 添加功能按钮
        self.add_face_btn = ttk.Button(
            self.control_frame,
            text="录入人脸",
            command=self.start_face_capture
        )
        self.add_face_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(
            self.control_frame,
            text="训练模型",
            command=self.start_training
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.recognize_btn = ttk.Button(
            self.control_frame,
            text="人脸识别",
            command=self.start_recognition
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = ttk.Button(
            self.control_frame,
            text="退出",
            command=self.quit_app
        )
        self.quit_btn.pack(side=tk.RIGHT, padx=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        self.status_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # 视频相关变量
        self.cap = None
        self.running = False
        self.current_mode = None
        
    def start_face_capture(self):
        self.stop_capture()
        self.current_mode = "capture"
        
        user_id = self.get_user_id()
        if user_id:
            self.status_var.set(f"正在录入 {user_id} 的人脸数据...")
            self.cap = cv2.VideoCapture(0)
            self.running = True
            threading.Thread(
                target=capture_face_samples,
                args=(user_id, self.update_video_frame),
                daemon=True
            ).start()
            self.update_video()

    def update_video_frame(self, frame):
        """更新视频帧的回调函数"""
        self.current_frame = frame
            
    def start_training(self):
        self.stop_capture()
        self.current_mode = "train"
        
        self.status_var.set("正在训练模型...")
        threading.Thread(
            target=train_recognizer,
            daemon=True
        ).start()
        self.status_var.set("模型训练完成")
        
    def start_recognition(self):
        self.stop_capture()
        self.current_mode = "recognize"
        
        self.status_var.set("正在识别人脸...")
        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(
            target=self.run_recognition,
            daemon=True
        ).start()
        self.update_video()

    def run_recognition(self):
        """运行人脸识别并显示结果"""
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        
        # 加载ID映射
        id_map = {}
        if os.path.exists('trainer/id_map.txt'):
            with open('trainer/id_map.txt', 'r') as f:
                for line in f:
                    id, name = line.strip().split(':')
                    id_map[int(id)] = name
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x,y,w,h) in faces:
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                
                if confidence < 60:
                    name = id_map.get(id, f"User{id}")
                    confidence = round(100 - confidence)
                    self.status_var.set(f"识别结果: {name} ({confidence}%)")
                    self.current_frame = frame
                    self.running = False
                    messagebox.showinfo("识别结果", f"识别到: {name}\n置信度: {confidence}%")
                    break
                else:
                    self.status_var.set("未识别到已知人脸")
                    self.current_frame = frame
        
    def get_user_id(self):
        user_id = tk.simpledialog.askstring(
            "用户ID",
            "请输入用户ID:",
            parent=self.root
        )
        return user_id
        
    def stop_capture(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.running = False
        
    def quit_app(self):
        self.stop_capture()
        self.root.quit()
        
    def update_video(self):
        if hasattr(self, 'current_frame'):
            # 转换颜色空间
            frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            
            # 调整图像大小以适应窗口
            height, width, _ = frame.shape
            max_size = 600
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # 转换为Tkinter可显示的图像
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # 更新显示
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        if self.running:
            self.root.after(10, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
