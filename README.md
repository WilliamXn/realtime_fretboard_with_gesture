项目概述
这是一个基于计算机视觉的吉他弹奏识别系统，能够实时检测吉他指板位置、追踪手指位置，并识别当前按下的和弦。系统包含三个主要功能模块：

​​数据收集模块​​ - 采集吉他演奏时的手部关键点数据
​​模型训练模块​​ - 使用收集的数据训练和弦分类模型
​​实时识别模块​​ - 实时识别当前演奏的和弦
系统要求
Python 3.8+
支持OpenCV的摄像头设备
推荐使用NVIDIA GPU以获得更好的YOLO模型性能
安装指南
克隆项目仓库：
git clone https://github.com/your-repo/guitar-chord-recognition.git
cd guitar-chord-recognition
创建并激活虚拟环境：
python -m venv venv
source venv/bin/activate  # Linux/MacOS)
venv\Scripts\activate  # Windows
安装依赖：
pip install -r requirements.txt
使用说明
1. 数据收集
运行主程序：
python main.py
切换到"Data Collection"标签页
设置YOLO模型路径（默认使用fretboard-detection-200epochs.pt）
设置输出目录（默认guitar_data）
从下拉菜单中选择要收集的和弦
点击"Run Collector"初始化采集器
点击"Start Collection"开始采集数据
在摄像头前正确摆放吉他并按选定的和弦
点击"Stop Collection"停止采集并保存数据
​​提示​​：

确保指板清晰可见
保持手部在摄像头范围内
每个和弦建议采集100-200个样本
数据将保存为CSV文件，文件名包含时间戳和和弦名称
2. 模型训练
收集足够数据后，切换到"Model Training"标签页
设置训练数据目录（默认guitar_data）
设置模型保存路径（默认chord_classifier）
点击"Train Model"开始训练
训练完成后会显示准确率和分类报告
​​注意​​：

训练过程可能需要几分钟，取决于数据量
确保至少收集了3种不同和弦的数据
3. 实时识别
训练完成后，切换到"Real-time Classification"标签页
设置模型目录（默认chord_classifier）
点击"Run Classifier"启动实时识别
在摄像头前演奏吉他，系统会实时显示识别结果
​​使用技巧​​：

保持指板清晰可见
按弦时手指应清晰可见
系统会显示识别置信度，高置信度(>0.7)结果更可靠
文件结构
guitar-chord-recognition/
├── README.md               # 本文件
├── requirements.txt        # 依赖列表
├── main.py                 # 主程序入口
├── chords.json             # 和弦列表定义
├── guitar_data_collector.py # 数据采集模块
├── realtime_chord_classifier.py # 实时识别模块
├── train_chord_classifier.py # 模型训练模块
├── guitar_data/            # 默认数据存储目录
└── chord_classifier/       # 默认模型存储目录
常见问题
​​Q: 系统无法检测到指板​​

确保使用正确的YOLO模型
检查摄像头是否正常工作
调整吉他位置，确保指板清晰可见
​​Q: 识别准确率低​​

收集更多训练数据
确保采集数据时姿势与实际演奏一致
尝试调整模型训练参数
​​Q: 手部检测不稳定​​

确保良好的光照条件
避免快速移动手部
可以尝试调整MediaPipe的检测参数
贡献指南
欢迎提交Pull Request或Issue报告问题。主要开发方向包括：

改进检测算法
增加更多和弦支持
优化用户界面
添加新功能
