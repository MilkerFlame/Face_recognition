import sys
import PIL.Image
import cv2
import os
import numpy
import PIL


def detect_face(img):
    # 转化灰度
    face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载分类器
    faceDetect = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = faceDetect.detectMultiScale(face_gray)  # 使用分类器识别人脸位置 返回人脸位置的坐标数组
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 255),
                     thickness=2)  # 在原图上画一个矩形
    # cv2.imshow('face', img)
    # cv2.waitKey(1)
    return face


# 人脸录入
def capture_faces(name, num_samples=20):
    # 创建以人名命名的子目录
    save_dir = os.path.join('face_recognition', name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print(f"开始录入 {name} 的人脸，请面对摄像头...")
    
    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # 检测人脸
        faces = detect_face(frame)
        if len(faces) == 1:  # 只保存有且只有一张人脸的照片
            cv2.imwrite(os.path.join(save_dir, f'{name}_{count+1}.png'), frame)
            count += 1
            print(f"已保存 {count}/{num_samples} 张照片到目录 {save_dir}")
            
        # 显示进度
        cv2.putText(frame, f"Capturing {count}/{num_samples}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Capturing Faces', frame)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):  # 按q键提前退出
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"{name} 的人脸录入完成")


# 获取训练图像
def getImageAndLabel(path):
    # 存储人脸数据
    facesSamples = []
    # 存储名字
    names = []
    # 存储图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    
    for imagePath in imagePaths:
        # 使用灰度格式打开照片
        PIL_img = PIL.Image.open(imagePath).convert('L')
        # 数组化
        np_img = numpy.array(PIL_img, 'uint8')
        # 获取人脸
        # 将灰度图转换回BGR格式用于检测
        color_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
        faces = detect_face(color_img)
        # 获取名字
        name = os.path.split(imagePath)[1].split('_')[0]
        for x, y, w, h in faces:
            names.append(name)
            facesSamples.append(np_img[y:y+h, x:x+w])
    return facesSamples, names


# 人脸识别函数
# 全局变量存储名字到ID的映射
name_to_id = {}
id_to_name = {}
names = []
recognizer = cv2.face.LBPHFaceRecognizer_create()

def recognize_face():
    recognizer.read('trainer/trainer.yml')
    
    # 读取名字映射
    global id_to_name, names
    if not id_to_name:
        try:
            with open('trainer/names.txt', 'r') as f:
                for line in f:
                    id, name = line.strip().split(':')
                    id_to_name[int(id)] = name
                    names.append(name)
        except FileNotFoundError:
            print("警告：未找到names.txt文件，使用默认名称")
            id_to_name = {0: "Unknown"}
            names = ["Unknown"]
    
    # 创建命名窗口
    try:
        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognition', 800, 600)
        print("窗口创建成功")
    except Exception as e:
        print(f"窗口创建失败: {str(e)}")
        return
        
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
        
    print("摄像头已成功打开")
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头画面")
            break
            
        # 确保窗口在最前面
        cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_TOPMOST, 1)
            
        # 直接使用彩色图像进行人脸检测
        faces = detect_face(frame)
        
        # 转换为灰度图像用于识别
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for (x, y, w, h) in faces:
            # 使用灰度图像进行预测
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            if confidence < 50 and id in id_to_name:  # 可信度阈值且ID存在
                name = id_to_name[id]
                cv2.putText(frame, f'You are {name}', (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Unknown', (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/record', methods=['POST'])
def record():
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'error': '姓名不能为空'}), 400
    
    try:
        capture_faces(name)
        return jsonify({'message': '录入成功'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recognize', methods=['GET'])
def recognize():
    try:
        recognize_face()
        return jsonify({'message': '开始识别'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import argparse

# 主程序
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸识别系统')
    parser.add_argument('--record', action='store_true', help='进入人脸录入模式')
    parser.add_argument('--name', type=str, help='要录入的人名')
    parser.add_argument('--recognize', action='store_true', help='进入人脸识别模式')
    args = parser.parse_args()

    if args.record:
        if not args.name:
            print("错误：请使用--name参数指定要录入的人名")
            sys.exit(1)
        capture_faces(args.name)
    elif args.recognize:
        recognize_face()
    else:
        # 创建trainer目录
        trainer_path = 'trainer'
        if not os.path.exists(trainer_path):
            try:
                os.makedirs(trainer_path)
                print(f"成功创建目录: {trainer_path}")
            except Exception as e:
                print(f"创建目录失败: {e}")
                sys.exit(1)
            
        # 训练模型
        path = 'face_recognition\\'
        try:
            faces, names = getImageAndLabel(path)
            if len(faces) == 0:
                print("错误：没有找到任何人脸图片")
                sys.exit(1)
                
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            name_to_id = {name: idx for idx, name in enumerate(set(names))}
            recognizer.train(faces, numpy.array([name_to_id[n] for n in names], dtype=numpy.int32))
            
            # 保存训练文件和名字映射
            try:
                model_path = os.path.join(trainer_path, 'trainer.yml')
                recognizer.write(model_path)
                print(f"模型已成功保存到: {model_path}")
                
                # 保存名字映射
                with open(os.path.join(trainer_path, 'names.txt'), 'w') as f:
                    for name, id in name_to_id.items():
                        f.write(f"{id}:{name}\n")
                print("名字映射已保存")
            except Exception as e:
                print(f"保存失败: {e}")
                sys.exit(1)
                
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            sys.exit(1)
        
        # 开始识别
        try:
            recognize_face()
        except Exception as e:
            print(f"人脸识别过程中发生错误: {e}")
            sys.exit(1)
