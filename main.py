import cv2
import os
import numpy as np
from PIL import Image

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 创建训练数据目录
if not os.path.exists('trainer'):
    os.makedirs('trainer')

def capture_face_samples(user_id):
    """采集人脸样本"""
    cam = cv2.VideoCapture(0)
    sample_num = 0
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            sample_num += 1
            # 保存人脸图像
            cv2.imwrite(f"trainer/User.{user_id}.{sample_num}.jpg", gray[y:y+h,x:x+w])
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.waitKey(100)
            
        cv2.imshow('Capturing Face Samples', img)
        cv2.waitKey(1)
        
        if sample_num > 20:  # 采集20个样本
            break
            
    cam.release()
    cv2.destroyAllWindows()

def train_recognizer():
    """训练人脸识别模型"""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    ids = []
    
    for f in os.listdir('trainer'):
        if f.endswith('.jpg'):
            img_path = os.path.join('trainer', f)
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, 'uint8')
    id = 1  # 固定ID为1，因为当前只有一个用户
            
    faces.append(img_np)
    ids.append(id)
    
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    print(f"训练完成，共训练了{len(set(ids))}个人的面部数据")

def recognize_face():
    """实时人脸识别"""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            if confidence < 60:  # 降低置信度阈值
                name = "User" + str(id)
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

if __name__ == "__main__":
    print("请选择操作：")
    print("1. 录入人脸")
    print("2. 训练模型") 
    print("3. 人脸识别")
    
    choice = input("请输入选项：")
    
    if choice == '1':
        user_id = input("请输入用户ID：")
        capture_face_samples(user_id)
    elif choice == '2':
        train_recognizer()
    elif choice == '3':
        recognize_face()
