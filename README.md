# 人脸识别系统

## 项目概述
这是一个基于OpenCV的人脸识别系统，使用Haar级联分类器进行人脸检测，并使用LBPH算法进行人脸识别训练和识别。

## 技术栈
- Python 3
- OpenCV
- NumPy
- Pillow

### 使用的分类器
- 类型：Haar级联分类器
- 模型文件：haarcascade_frontalface_default.xml
- 描述：OpenCV提供的预训练人脸检测模型，用于检测图像中的人脸区域

### 使用的识别模型
- 类型：LBPH（Local Binary Patterns Histograms）
- 模型文件：trainer/trainer.yml
- 描述：基于局部二值模式直方图的人脸识别算法，适合处理小规模数据集

## 安装和运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行程序：
```bash
python gui.py
```

## 使用方法

### 1. 录入人脸
- 选择菜单选项1
- 输入用户ID（如：user1）
- 程序将打开摄像头，采集50张人脸样本
- 样本将保存在`face_recognition/user_<用户ID>`目录下

### 2. 训练模型
- 选择菜单选项2
- 程序将读取所有用户的人脸样本
- 训练LBPH模型并保存到`trainer/trainer.yml`
- 用户ID映射将保存到`trainer/id_map.txt`

### 3. 人脸识别
- 选择菜单选项3
- 程序将打开摄像头进行实时人脸识别
- 识别结果将显示在视频窗口中
- 按'q'键退出识别模式

## 文件结构说明

```
.
├── face_recognition/        # 存储用户人脸样本
│   └── user_<用户ID>/       # 每个用户的样本目录
├── trainer/                 # 训练相关文件
│   ├── trainer.yml          # 训练好的LBPH模型
│   └── id_map.txt           # 用户ID映射文件
├── main.py                  # 主程序
├── gui.py                   # GUI界面（可选）
└── requirements.txt         # 依赖文件
```

## 注意事项
1. 确保环境光线充足，人脸清晰可见
2. 采集样本时保持不同角度和表情
3. 每个用户建议采集50张以上样本
4. 训练模型前确保有足够样本
5. 识别时保持适当距离（0.5-1米）
