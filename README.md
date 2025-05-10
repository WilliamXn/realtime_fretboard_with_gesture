# 吉他和弦识别系统使用指南

## 项目概述

吉他和弦识别系统是一个基于计算机视觉和机器学习的应用程序，能够实时识别吉他演奏中的和弦。该系统通过摄像头捕获吉他手的演奏画面，使用YOLO模型检测吉他指板关键点，MediaPipe检测手部关键点，并通过随机森林分类器预测和弦。系统支持数据收集、模型训练和实时和弦分类功能，适用于吉他教学、练习和研究。

本指南将帮助您安装、配置和使用该系统。

---

## 系统要求

### 硬件

- 带摄像头的计算机（推荐分辨率1280x720或更高）
- 推荐配置：4核CPU，8GB RAM，GPU（如NVIDIA）以加速YOLO模型推理
- 吉他（用于数据采集和实时分类）

### 软件

- **操作系统**：Windows、Linux 或 macOS
- **Python**：版本 3.8 或更高
- **依赖库**：见 `requirements.txt↳`
- **外部模型**：
  - YOLO模型文件：`fretboard-detection-200epochs.pt`（需自行训练或获取）
  - 可选：预训练和弦分类模型（位于 `chord_classifier` 目录）

---

## 安装步骤

### 1. 克隆或下载项目

```bash
git clone <repository_url>
cd guitar-chord-recognition
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. 安装依赖

确保已下载 `requirements.txt`，然后运行：

```bash
pip install -r requirements.txt
```

**注意**：

- 如果 `lightglue` 安装失败，请根据 `requirements.txt` 中的路径手动安装或移除此依赖（本项目未直接使用）。
- 某些依赖（如 `torch`、`opencv-python`）可能需要根据您的系统和CUDA支持手动选择版本。

### 4. 获取 YOLO 模型

将预训练的 YOLO 模型 `fretboard-detection-200epochs.pt` 放入项目根目录，或更新 `main.py` 中的默认路径：

```python
self.model_path_entry = QLineEdit("path/to/your/fretboard-detection-200epochs.pt")
```

---

## 项目结构

- `main.py`：主应用程序，包含 GUI 和核心逻辑
- `guitar_data_collector.py`：数据采集模块，检测指板和手部关键点并保存数据
- `train_chord_classifier.py`：训练随机森林分类器
- `realtime_chord_classifier.py`：实时和弦分类↳
- `requirements.txt`：项目依赖↳
- `main.spec`：PyInstaller 打包配置文件
- `chords.json`：和弦列表（可选，默认包含 C、D、E、F、G、A、B）
- `guitar_data/`：数据存储目录
- `chord_classifier/`：训练好的模型存储目录

---

## 使用说明

### 1. 启动应用程序

运行以下命令启动 GUI：

```bash
python main.py
```

界面包含三个选项卡：

- **数据采集**：收集吉他和弦数据
- **模型训练**：训练和弦分类模型
- **实时分类**：实时识别和弦

### 2. 数据采集

1. 在“数据采集”选项卡中：
   - **YOLO 模型路径**：确保指向正确的 `fretboard-detection-200epochs.pt`。
   - **输出目录**：指定数据保存路径（默认 `guitar_data`）。
   - **和弦选择**：从下拉菜单选择要采集的和弦（基于 `chords.json` 或默认列表）。
2. 点击“运行收集器”初始化采集器。
3. 点击“开始采集”录制指定和弦的数据，演奏对应和弦。
4. 点击“停止采集”保存数据为 CSV 文件（格式：`<session_id>_<chord>.csv`）。

**注意**：

- 确保摄像头能清晰捕捉吉他指板和左手。
- 数据保存在 `guitar_data` 目录，包含归一化的手部关键点坐标。

### 3. 模型训练

1. 在“模型训练”选项卡中：
   - **数据目录**：指向包含 CSV 文件的目录（默认 `guitar_data`）。
   - **模型保存路径**：指定训练后模型的保存路径（默认 `chord_classifier`）。
2. 点击“训练模型”开始训练。
3. 训练完成后，模型（`random_forest.joblib`、`scaler.joblib`、`label_encoder.joblib`）保存至指定目录。

**输出**：

- 训练集和测试集的准确率
- 分类报告（每个和弦的精确度、召回率等）
- 特征重要性分析（关键点的 x/y 坐标重要性）

### 4. 实时分类

1. 在“实时分类”选项卡中：
   - **模型目录**：指向包含训练好的模型文件的目录（默认 `chord_classifier`）。
2. 点击“运行分类器”启动实时和弦识别。
3. 系统将显示：
   - 检测到的指板和手部关键点
   - 预测的和弦及其置信度（置信度 &gt; 0.7 绿色，0.5-0.7 橙色，&lt; 0.5 红色）
   - 帧率（FPS）

**退出**：按 `q` 键退出分类模式。

### 5. 键盘控制（实时分类和数据采集）

- `d` **键**：打印当前帧的归一化坐标（用于调试）。
- 其他键（如 `1`-`9`）在 `guitar_data_collector.py` 中定义了和弦录制（未在 GUI 中使用）。

---


## 故障排查

### 1. 摄像头无法打开

- 检查摄像头是否被其他程序占用。
- 在 `main.py` 中调整 `CAMERA_INDEX`（默认 `0`）：

  ```python
  CAMERA_INDEX = 1  # 尝试其他索引
  ```

### 2. YOLO 模型加载失败

- 确保 `fretboard-detection-200epochs.pt` 存在且路径正确。
- 检查 `ultralytics` 版本兼容性。

### 3. 数据采集无数据保存

- 确保指板和手部关键点被正确检测（屏幕上应显示绿色/红色圆点）。
- 检查 `guitar_data` 目录是否有写权限。

### 4. 分类器准确率低

- 增加训练数据量（每个和弦至少 1000 帧）。
- 检查数据质量（确保和弦姿势正确，摄像头角度一致）。
- 调整 `train_chord_classifier.py` 中的随机森林参数：

  ```python
  self.model = RandomForestClassifier(
      n_estimators=300,  # 增加树数量
      max_depth=20,     # 调整深度
      min_samples_split=3
  )
  ```

---

## 扩展功能

1. **添加新和弦**：

   - 编辑 `chords.json` 添加和弦名称（确保为字符串列表）。
   - 采集新和弦数据并重新训练模型。

2. **优化模型**：

   - 尝试其他分类器（如 XGBoost 或神经网络）。
   - 增加关键点特征（如手指角度、弦位置）。

3. **实时反馈**：

   - 集成音频分析以验证和弦正确性。
   - 添加和弦练习模式，提示用户切换和弦。

---

## 常见问题

**Q: 为什么实时分类总是预测“Unknown”**？A: 可能是模型未训练或训练数据不足。请确保 `chord_classifier` 目录包含有效的模型文件，并检查训练数据是否涵盖所有目标和弦。

**Q: 如何提高模型准确率**？A: 增加训练样本（每个和弦 1000+ 帧），确保数据采集时姿势标准，优化随机森林参数，或尝试其他分类算法。

**Q: 为什么 GUI 界面卡顿**？A: 可能是因为 YOLO 模型推理耗时长。尝试降低摄像头分辨率（修改 `main.py` 中的 `CAP_PROP_FRAME_WIDTH/HEIGHT`），或使用 GPU 加速。

---

