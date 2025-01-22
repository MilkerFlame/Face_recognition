import cv2
import os
import numpy as np
from PIL import Image

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 创建训练数据目录
if not os.path.exists('trainer'):
    os.makedirs('trainer')

def capture_face_samples(user_id, gui_callback=None):
    """采集人脸样本"""
    cam = cv2.VideoCapture(0)
    sample_num = 0
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        if len(faces) > 0:
            for (x,y,w,h) in faces:
                sample_num += 1
                # 保存人脸图像
                cv2.imwrite(f"trainer/User.{user_id}.{sample_num}.jpg", gray[y:y+h,x:x+w])
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.putText(img, f"Captured: {sample_num}/50", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.waitKey(300)  # 增加延迟以确保不同样本
            
        if gui_callback:
            gui_callback(img)
        else:
            cv2.imshow('Capturing Face Samples', img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
        
        if sample_num > 50:  # 增加样本数量到50
            break
            
    cam.release()
    if not gui_callback:
        cv2.destroyAllWindows()

def train_recognizer():
    """训练人脸识别模型"""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    ids = []
    id_map = {}
    current_id = 0
    
    image_paths = [os.path.join('trainer', f) for f in os.listdir('trainer') if f.endswith('.jpg')]
    
    # 创建用户ID映射
    for image_path in image_paths:
        user_id = os.path.split(image_path)[-1].split('.')[1]
        if user_id not in id_map:
            id_map[user_id] = current_id
            current_id += 1
    
    # 保存ID映射
    with open('trainer/id_map.txt', 'w') as f:
        for name, id in id_map.items():
            f.write(f"{id}:{name}\n")
    
    # 训练模型
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')
        user_id = os.path.split(image_path)[-1].split('.')[1]
        id = id_map[user_id]
        faces.append(img_np)
        ids.append(id)
    
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    print(f"训练完成，共训练了{len(id_map)}个人的面部数据")

def recognize_face():
    """实时人脸识别"""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    
    # 加载ID映射
    id_map = {}
    if os.path.exists('trainer/id_map.txt'):
        with open('trainer/id_map.txt', 'r') as f:
            for line in f:
                id, name = line.strip().split(':')
                id_map[int(id)] = name
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            if confidence < 60:  # 降低置信度阈值
                name = id_map.get(id, f"User{id}")
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                name = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        cv2.imshow('Face Recognition', img)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()

def main_menu():
    while True:
        print("\n请选择操作：")
        print("1. 录入人脸")
        print("2. 训练模型")
        print("3. 人脸识别")
        print("4. 退出")
        
        choice = input("请输入选项：")
        
        if choice == '1':
            user_id = input("请输入用户ID：")
            capture_face_samples(user_id)
        elif choice == '2':
            train_recognizer()
        elif choice == '3':
            recognize_face()
        elif choice == '4':
            break
        else:
            print("无效选项，请重新选择")

if __name__ == "__main__":
    main_menu()
