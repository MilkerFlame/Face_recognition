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
        self.root.geometry("1000x800")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建视频显示区域
        self.video_frame = ttk.LabelFrame(self.main_frame, text="视频预览")
        
        # 创建菜单栏（放在视频区域之后以确保正确显示）
        self.create_menu()
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

    def load_names_map(self):
        """加载名称映射"""
        name_map = {}
        if os.path.exists('trainer/names.txt'):
            with open('trainer/names.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    id, name = line.strip().split(':')
                    name_map[int(id)] = name
        return name_map

    def run_recognition(self):
        """运行人脸识别并显示结果"""
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        
        # 初始化人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 加载名称映射，优先使用names.txt
        name_map = self.load_names_map()
        if not name_map and os.path.exists('trainer/id_map.txt'):
            with open('trainer/id_map.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    id, name = line.strip().split(':')
                    name_map[int(id)] = name
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x,y,w,h) in faces:
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                
                if confidence < 60:
                    name = name_map.get(id, f"User{id}")
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
        
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="资料库", command=self.show_database)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.quit_app)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        self.root.config(menu=menubar)
        
    def show_database(self):
        """显示资料库界面"""
        # 密码验证
        password = tk.simpledialog.askstring(
            "密码验证",
            "请输入密码:",
            show='*',
            parent=self.root
        )
        
        if password != "123456":  # 默认密码
            messagebox.showerror("错误", "密码错误")
            return
            
        # 创建资料库窗口
        db_window = tk.Toplevel(self.root)
        db_window.title("人脸资料库")
        db_window.geometry("600x400")
        
        # 创建列表框架
        list_frame = ttk.Frame(db_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建滚动条
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建用户列表
        self.user_list = ttk.Treeview(
            list_frame,
            columns=("id", "name"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        self.user_list.heading("id", text="ID")
        self.user_list.heading("name", text="姓名")
        self.user_list.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.user_list.yview)
        
        # 加载用户数据
        self.load_user_data()
        
        # 创建操作按钮
        btn_frame = ttk.Frame(db_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        view_btn = ttk.Button(
            btn_frame,
            text="查看照片",
            command=self.view_user_photos
        )
        view_btn.pack(side=tk.LEFT, padx=5)
        
        delete_btn = ttk.Button(
            btn_frame,
            text="删除用户",
            command=self.delete_user
        )
        delete_btn.pack(side=tk.LEFT, padx=5)
        
    def load_user_data(self):
        """加载用户数据"""
        if os.path.exists('trainer/id_map.txt'):
            with open('trainer/id_map.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    id, name = line.strip().split(':')
                    self.user_list.insert("", "end", values=(id, name))
        
    def view_user_photos(self):
        """查看用户照片"""
        selected = self.user_list.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个用户")
            return
            
        user_id = self.user_list.item(selected[0])['values'][0]
        user_dir = os.path.join('face_recognition', f'user_{user_id}')
        
        if not os.path.exists(user_dir):
            messagebox.showwarning("提示", "该用户没有照片")
            return
            
        # 检查照片文件是否存在
        photo_files = [f for f in os.listdir(user_dir) if f.endswith('.jpg')]
        if not photo_files:
            messagebox.showwarning("提示", "该用户没有照片")
            return
            
        # 创建照片查看窗口
        photo_window = tk.Toplevel(self.root)
        photo_window.title(f"用户 {user_id} 的照片")
        photo_window.geometry("800x600")
        
        # 创建照片显示区域
        photo_frame = ttk.Frame(photo_window)
        photo_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 加载照片
        for img_file in os.listdir(user_dir):
            img_path = os.path.join(user_dir, img_file)
            img = Image.open(img_path)
            img = img.resize((200, 200), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            
            label = ttk.Label(photo_frame, image=img)
            label.image = img  # 保持引用
            label.pack(side=tk.LEFT, padx=5, pady=5)
        
    def delete_user(self):
        """删除用户"""
        selected = self.user_list.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个用户")
            return
            
        user_id = self.user_list.item(selected[0])['values'][0]
        confirm = messagebox.askyesno(
            "确认删除",
            f"确定要删除用户 {user_id} 吗？"
        )
        
        if confirm:
            # 删除用户照片
            user_dir = os.path.join('face_recognition', f'user_{user_id}')
            if os.path.exists(user_dir):
                try:
                    for img_file in os.listdir(user_dir):
                        img_path = os.path.join(user_dir, img_file)
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    if os.path.exists(user_dir):
                        os.rmdir(user_dir)
                except Exception as e:
                    messagebox.showerror("错误", f"删除用户照片失败: {str(e)}")
                    return
            
            # 从id_map.txt中删除用户
            if os.path.exists('trainer/id_map.txt'):
                with open('trainer/id_map.txt', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                with open('trainer/id_map.txt', 'w', encoding='utf-8') as f:
                    for line in lines:
                        if not line.startswith(f"{user_id}:"):
                            f.write(line)
            
            # 从列表中删除
            self.user_list.delete(selected[0])
            messagebox.showinfo("成功", "用户已删除")
        
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
