import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from main import capture_faces, recognize_face, detect_face, recognizer, id_to_name

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统")
        self.root.geometry("800x600")
        self.is_recording = False  # 添加录入状态标志
        
        # 初始化摄像头，使用DirectShow后端
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():t
            messagebox.showerror("错误", "无法打开摄像头，请检查摄像头是否连接")
            self.root.destroy()
            return
            
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
            
        self.current_image = None
        self.recognized_name = None
        self.confidence = 0
        
        # 创建主界面布局
        self.create_layout()
        
        # 启动视频更新
        self.update_video()
        
    def create_layout(self):
        # 创建主容器
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 视频显示区域
        self.video_label = tk.Label(main_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 录入按钮
        self.btn_record = tk.Button(
            control_frame,
            text="录入人脸",
            command=self.record_face,
            width=15,
            height=2
        )
        self.btn_record.pack(side=tk.LEFT, padx=5)
        
        # 状态标签
        self.status_label = tk.Label(
            control_frame,
            text="准备就绪",
            fg="green"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        
    def draw_face_info(self, frame):
        """在视频帧上绘制识别结果"""
        if self.recognized_name:
            # 绘制人脸框
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 2)
            
            # 转换为PIL图像以便添加中文
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)
            
            # 设置字体
            try:
                font = ImageFont.truetype("simhei.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            # 绘制识别结果
            text = f"{self.recognized_name} ({self.confidence:.2f})"
            draw.text((10, 10), text, font=font, fill=(255, 0, 0))
            
            return np.array(pil_img)
        return frame
        
    def update_video(self):
        # 如果正在录入，则暂停实时显示
        if self.is_recording:
            self.root.after(30, self.update_video)
            return
            
        # 从摄像头获取帧
        ret, frame = self.cap.read()
        if ret:
            # 实时识别
            try:
                # 转换为灰度图像用于检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 检测人脸
                faces = detect_face(frame)
                
                # 对每个检测到的人脸进行识别
                for (x, y, w, h) in faces:
                    # 使用灰度图像进行预测
                    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if confidence < 50 and id in id_to_name:  # 可信度阈值且ID存在
                        self.recognized_name = id_to_name[id]
                        self.confidence = confidence
                        # 绘制识别结果
                        cv2.putText(frame, f'{self.recognized_name}', (x, y-30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f'{confidence:.2f}%', (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        self.recognized_name = None
                        cv2.putText(frame, 'Unknown', (x, y-30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.putText(frame, 'Unknown', (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 绘制人脸框
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                # 更新状态
                self.status_label.config(
                    text=f"识别结果: {self.recognized_name or '未知'}",
                    fg="green" if self.recognized_name else "red"
                )
            except Exception as e:
                self.status_label.config(text=f"识别错误: {str(e)}", fg="red")
            
            # 转换为RGB格式显示
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像
            self.current_image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            
            # 更新显示
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk
            
            # 自动保存识别结果
            if self.recognized_name:
                self.save_recognized_face(frame)
            
        # 每30ms更新一次
        self.root.after(30, self.update_video)
        
    def record_face(self):
        # 弹出输入框获取姓名
        name = tk.simpledialog.askstring("录入人脸", "请输入姓名:")
        if name:
            try:
                self.is_recording = True  # 设置录入状态
                self.status_label.config(text="正在录入...", fg="blue")
                # 使用独立窗口进行录入
                capture_faces(name)
                self.status_label.config(
                    text=f"{name} 录入成功！",
                    fg="blue"
                )
            except Exception as e:
                self.status_label.config(
                    text=f"录入失败: {str(e)}",
                    fg="red"
                )
            finally:
                self.is_recording = False  # 重置录入状态
                
    def save_recognized_face(self, frame):
        """保存识别到的人脸"""
        if self.recognized_name:
            timestamp = int(time.time())
            filename = f"pic/{self.recognized_name}_{timestamp}.png"
            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
    def __del__(self):
        # 释放摄像头资源
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
