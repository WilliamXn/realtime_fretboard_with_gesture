import os
import sys
import cv2
import numpy as np
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget, 
                            QMessageBox, QFileDialog, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from guitar_data_collector import GuitarPerformanceCollector
from realtime_chord_classifier import RealTimeChordClassifier
from train_chord_classifier import ChordClassifierTrainer

CAMERA_INDEX = 0 # Default camera index

class GuitarChordApp(QMainWindow):
    update_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guitar Chord Recognition System")
        self.setGeometry(100, 100, 1000, 800)  # Increased window size

        # Initialize variables
        self.collector = None
        self.classifier = None
        self.trainer = None
        self.is_collecting = False
        self.current_chord = None
        self.capture = None
        self.timer = QTimer()
        self.current_frame = None

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.main_layout.setSpacing(0)  # Remove spacing

        # Create GUI elements
        self.create_widgets()

        # Connect signals
        self.update_frame_signal.connect(self.update_frame)
        self.timer.timeout.connect(self.update_video)

    def create_widgets(self):
        # Create video display area
        self.create_video_display()

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget, stretch=1)

        # Data Collection Tab
        self.collect_tab = QWidget()
        self.tab_widget.addTab(self.collect_tab, "Data Collection")

        # Model Path
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("YOLO Model Path:")
        self.model_path_entry = QLineEdit("fretboard-detection-200epochs.pt")
        browse_model_btn = QPushButton("Browse")
        browse_model_btn.clicked.connect(self.browse_model)
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_entry)
        model_path_layout.addWidget(browse_model_btn)

        # Output Directory
        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel("Output Directory:")
        self.output_dir_entry = QLineEdit("guitar_data")
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(output_dir_label)
        output_dir_layout.addWidget(self.output_dir_entry)
        output_dir_layout.addWidget(browse_output_btn)

        # Chord Selection
        chord_layout = QHBoxLayout()
        chord_label = QLabel("Chord to Collect:")
        self.chord_combo = QComboBox()
        # Load chords from JSON file
        try:
            with open("chords.json", "r") as f:
                chord_list = json.load(f)
                if not isinstance(chord_list, list) or not all(isinstance(chord, str) for chord in chord_list):
                    raise ValueError("Invalid chord list format in chords.json")
                self.chord_combo.addItems(chord_list)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            # Fallback to default chord list if JSON loading fails
            default_chords = ["C", "D", "E", "F", "G", "A", "B"]
            self.chord_combo.addItems(default_chords)
            QMessageBox.warning(self, "Warning", f"Failed to load chords from chords.json: {str(e)}. Using default chords: {default_chords}")
        chord_layout.addWidget(chord_label)
        chord_layout.addWidget(self.chord_combo)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_collect_btn = QPushButton("Start Collection")
        self.start_collect_btn.clicked.connect(self.start_collection)
        self.stop_collect_btn = QPushButton("Stop Collection")
        self.stop_collect_btn.clicked.connect(self.stop_collection)
        self.stop_collect_btn.setEnabled(False)
        run_collector_btn = QPushButton("Run Collector")
        run_collector_btn.clicked.connect(self.reload_model)
        button_layout.addWidget(self.start_collect_btn)
        button_layout.addWidget(self.stop_collect_btn)
        button_layout.addWidget(run_collector_btn)

        # Add all to collect tab
        collect_tab_layout = QVBoxLayout(self.collect_tab)
        collect_tab_layout.addLayout(model_path_layout)
        collect_tab_layout.addLayout(output_dir_layout)
        collect_tab_layout.addLayout(chord_layout)
        collect_tab_layout.addLayout(button_layout)
        collect_tab_layout.addStretch()

        # Training Tab
        self.train_tab = QWidget()
        self.tab_widget.addTab(self.train_tab, "Model Training")

        # Data Directory
        train_data_layout = QHBoxLayout()
        train_data_label = QLabel("Data Directory:")
        self.train_data_dir_entry = QLineEdit("guitar_data")
        browse_train_data_btn = QPushButton("Browse")
        browse_train_data_btn.clicked.connect(self.browse_train_data_dir)
        train_data_layout.addWidget(train_data_label)
        train_data_layout.addWidget(self.train_data_dir_entry)
        train_data_layout.addWidget(browse_train_data_btn)

        # Model Save Path
        model_save_layout = QHBoxLayout()
        model_save_label = QLabel("Model Save Path:")
        self.model_save_entry = QLineEdit("chord_classifier")
        browse_model_save_btn = QPushButton("Browse")
        browse_model_save_btn.clicked.connect(self.browse_model_save_dir)
        model_save_layout.addWidget(model_save_label)
        model_save_layout.addWidget(self.model_save_entry)
        model_save_layout.addWidget(browse_model_save_btn)

        # Train Button
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.train_model)

        # Add all to train tab
        train_tab_layout = QVBoxLayout(self.train_tab)
        train_tab_layout.addLayout(train_data_layout)
        train_tab_layout.addLayout(model_save_layout)
        train_tab_layout.addWidget(train_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        train_tab_layout.addStretch()

        # Classification Tab
        self.classify_tab = QWidget()
        self.tab_widget.addTab(self.classify_tab, "Real-time Classification")

        # Model Load Path
        classify_model_layout = QHBoxLayout()
        classify_model_label = QLabel("Model Directory:")
        self.classify_model_entry = QLineEdit("chord_classifier")
        browse_classify_model_btn = QPushButton("Browse")
        browse_classify_model_btn.clicked.connect(self.browse_classify_model_dir)
        classify_model_layout.addWidget(classify_model_label)
        classify_model_layout.addWidget(self.classify_model_entry)
        classify_model_layout.addWidget(browse_classify_model_btn)

        # Run Classifier Button
        run_classifier_btn = QPushButton("Run Classifier")
        run_classifier_btn.clicked.connect(self.run_classifier)

        # Add all to classify tab
        classify_tab_layout = QVBoxLayout(self.classify_tab)
        classify_tab_layout.addLayout(classify_model_layout)
        classify_tab_layout.addWidget(run_classifier_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        classify_tab_layout.addStretch()

    def create_video_display(self):
        """Create video display area"""
        # Video display frame
        self.video_frame = QFrame()
        self.video_frame.setFrameShape(QFrame.Shape.NoFrame)
        self.video_frame.setStyleSheet("background-color: black;")
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Layout
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.addWidget(self.video_label)
        video_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        # Add to main layout with increased stretch
        self.main_layout.addWidget(self.video_frame, stretch=3)  # Increased stretch factor
        
        # Initialize video capture
        self.capture = cv2.VideoCapture(CAMERA_INDEX)
        self._init_video_capture()
        
    def _init_video_capture(self):
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.capture.isOpened():
            raise RuntimeError("无法打开摄像头")
        else:
            # Start timer for video updates
            self.timer.start(30)  # ~30 FPS

    def update_video(self):
        """
        更新视频帧
        检测指板和手指关键点
        归一化坐标
        """
        if not (self.capture and self.capture.isOpened()):  # Check if capture is valid
            QMessageBox.critical(self, "Error", "Camera not initialized or opened")
            self.timer.stop()
            return
        
        ret, frame = self.capture.read()
        if not ret:
            QMessageBox.critical(self, "Error", "Failed to read frame from camera")
            self.timer.stop()
            return
        
        # 获取当前按键
        key = cv2.waitKey(1) & 0xFF
        
        # 初始化处理后的帧为原始帧
        processed_frame = frame
        
        # 只有collector存在时才检测关键点
        if self.collector is not None:
            fretboard_kps, hand_kps = self.collector.detect_landmarks(frame)
            normalized_kps = self.collector.normalize_coordinates(fretboard_kps, hand_kps)
            processed_frame = self.collector.handle_landmarks(
                frame, key, fretboard_kps, hand_kps, normalized_kps
            )
        
        # 只有classifier存在时才处理预测
        if self.classifier is not None:
            processed_frame = self.classifier.handle_prediction(
                processed_frame, key, 
                fretboard_kps if 'fretboard_kps' in locals() else None, 
                hand_kps if 'hand_kps' in locals() else None, 
                normalized_kps if 'normalized_kps' in locals() else None
            )
        
        # 显示帧
        self.current_frame = processed_frame
        self.update_frame_signal.emit(processed_frame)
        
        # 处理按键事件
        self.handle_key_event(key)

    def handle_key_event(self, key):
        pass

    def update_frame(self, frame):
        """Convert OpenCV frame to QPixmap and display"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def browse_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "Model files (*.pt)")
        if filename:
            self.model_path_entry.setText(filename)

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_entry.setText(directory)

    def browse_train_data_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Training Data Directory")
        if directory:
            self.train_data_dir_entry.setText(directory)

    def browse_model_save_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Model Save Directory")
        if directory:
            self.model_save_entry.setText(directory)

    def browse_classify_model_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Classifier Model Directory")
        if directory:
            self.classify_model_entry.setText(directory)

    def start_collection(self):
        if not self.collector:
            QMessageBox.critical(self, "Error", "Please initialize collector by running it first!")
            return

        self.current_chord = self.chord_combo.currentText()
        self.collector.start_recording(self.current_chord)
        self.is_collecting = True
        self.start_collect_btn.setEnabled(False)
        self.stop_collect_btn.setEnabled(True)
        QMessageBox.information(self, "Info", f"Started collecting data for chord {self.current_chord}")

    def stop_collection(self):
        if self.collector and self.is_collecting:
            self.collector.stop_recording()
            self.is_collecting = False
            self.start_collect_btn.setEnabled(True)
            self.stop_collect_btn.setEnabled(False)
            QMessageBox.information(self, "Info", "Data collection stopped and saved")
        else:
            QMessageBox.warning(self, "Warning", "No active collection to stop")

    def reload_model(self):
        model_path = self.model_path_entry.text()
        output_dir = self.output_dir_entry.text()

        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", "Invalid YOLO model path!")
            return

        try:
            # Use existing video capture if available
            if not self.capture or not self.capture.isOpened():
                self.capture = cv2.VideoCapture(CAMERA_INDEX)
                if not self.capture.isOpened():
                    raise Exception("Could not open video device")
                
            self.collector = GuitarPerformanceCollector(model_path, output_dir, self.capture)
            
            QMessageBox.information(self, "Info", "Collector initialized. Use GUI to start/stop collection.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run collector: {str(e)}")
            if self.capture:
                self.capture.release()
                self.capture = None

    def train_model(self):
        data_dir = self.train_data_dir_entry.text()
        model_save_path = self.model_save_entry.text()

        if not os.path.exists(data_dir):
            QMessageBox.critical(self, "Error", "Invalid data directory!")
            return

        try:
            self.trainer = ChordClassifierTrainer(data_dir)
            accuracy = self.trainer.train()
            self.trainer.save_model(model_save_path)
            QMessageBox.information(self, "Success", f"Model trained with accuracy: {accuracy:.2f}\nSaved to {model_save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed: {str(e)}")

    def run_classifier(self):
        model_dir = self.classify_model_entry.text()

        if not os.path.exists(os.path.join(model_dir, "random_forest.joblib")):
            QMessageBox.critical(self, "Error", "Trained model not found in specified directory!")
            return

        try:
            # Use existing video capture if available
            if not self.capture or not self.capture.isOpened():
                self.capture = cv2.VideoCapture(0)
                if not self.capture.isOpened():
                    raise Exception("Could not open video device")
            
            self.classifier = RealTimeChordClassifier(model_dir, self.capture)
            QMessageBox.information(self, "Info", "Classifier started. Press 'q' in video window to exit.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run classifier: {str(e)}")
            if self.capture:
                self.capture.release()
                self.capture = None

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        if self.capture and self.capture.isOpened():
            self.capture.release()
        if self.timer.isActive():
            self.timer.stop()
        if self.collector:
            self.collector.destroy()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GuitarChordApp()
    window.show()
    sys.exit(app.exec())