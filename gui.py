import tkinter as tk
from tkinter import messagebox, simpledialog
from main import capture_face_samples, train_recognizer, recognize_face

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统")
        self.root.geometry("400x300")
        
        # 创建主界面
        self.create_widgets()
        
    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="人脸识别系统", font=("Arial", 20))
        title_label.pack(pady=20)
        
        # 功能按钮
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        self.capture_btn = tk.Button(btn_frame, text="录入人脸", width=15, command=self.capture_face)
        self.capture_btn.pack(pady=5)
        
        self.train_btn = tk.Button(btn_frame, text="训练模型", width=15, command=self.train_model)
        self.train_btn.pack(pady=5)
        
        self.recognize_btn = tk.Button(btn_frame, text="人脸识别", width=15, command=self.recognize)
        self.recognize_btn.pack(pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def capture_face(self):
        self.status_var.set("正在录入人脸...")
        self.root.update()
        
        user_id = self.get_user_id()
        if user_id:
            capture_face_samples(user_id)
            messagebox.showinfo("成功", "人脸录入完成！")
            self.status_var.set("就绪")
        
    def train_model(self):
        self.status_var.set("正在训练模型...")
        self.root.update()
        
        train_recognizer()
        messagebox.showinfo("成功", "模型训练完成！")
        self.status_var.set("就绪")
        
    def recognize(self):
        self.status_var.set("正在识别人脸...")
        self.root.update()
        
        recognize_face()
        self.status_var.set("就绪")
        
    def get_user_id(self):
        user_id = tk.simpledialog.askstring("用户ID", "请输入用户ID：")
        if not user_id:
            messagebox.showwarning("警告", "必须输入用户ID")
            return None
        return user_id

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
