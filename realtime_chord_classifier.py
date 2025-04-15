import cv2
import numpy as np
import time
import os
import joblib
from ultralytics import YOLO
import mediapipe as mp
from collections import deque

class RealTimeChordClassifier:
    def __init__(self, model_path, video_capture):
        if model_path is None:
            model_path = "chord_classifier"

        # 加载模型
        self.model = joblib.load(os.path.join(model_path, "random_forest.joblib"))
        self.scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
        self.label_encoder = joblib.load(os.path.join(model_path, "label_encoder.joblib"))
        self.reverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
        
        # 初始化计算机视觉模型
        self.yolo_model = YOLO("fretboard-detection-200epochs.pt")
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 参考坐标系设置
        self.origin_idx = 0  # 点0
        self.vec1_idx = 1    # 点1
        self.vec2_idx = 2    # 点2
        
        # 预测平滑
        self.prediction_history = deque(maxlen=7)  # 使用7帧历史记录
        self.last_prediction = None
        self.last_confidence = 0
        
        # 摄像头
        self.cap = video_capture
    
    def start_capture(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def detect_landmarks(self, frame):
        # YOLO指板检测
        yolo_results = self.yolo_model.predict(source=frame, show=False, save=False, verbose=False)
        fretboard_kps = yolo_results[0].keypoints.to('cpu').data.numpy()[0] if yolo_results[0].keypoints else None
        
        # MediaPipe手部检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.mp_hands.process(rgb_frame)
        hand_kps = None
        
        if hand_results.multi_hand_landmarks:
            hand_kps = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] 
                               for lm in hand_results.multi_hand_landmarks[0].landmark])
        
        return fretboard_kps, hand_kps
    
    def normalize_coordinates(self, fretboard_kps, hand_kps):
        if fretboard_kps is None or hand_kps is None or len(fretboard_kps) < 3:
            return None
        
        try:
            origin = fretboard_kps[self.origin_idx][:2]
            vec1 = fretboard_kps[self.vec1_idx][:2] - origin
            vec2 = fretboard_kps[self.vec2_idx][:2] - origin
            
            len_vec1 = np.linalg.norm(vec1)
            len_vec2 = np.linalg.norm(vec2)
            
            if len_vec1 < 1e-6 or len_vec2 < 1e-6:
                return None
                
            vec1 = vec1 / len_vec1
            vec2 = vec2 / len_vec2
            
            transform_mat = np.vstack([vec1, vec2]).T
            
            normalized_kps = []
            for kp in hand_kps:
                relative_vec = kp - origin
                proj_coords = np.linalg.solve(transform_mat, relative_vec)
                normalized_kps.append(proj_coords)
            
            return np.array(normalized_kps)
            
        except Exception as e:
            print(f"坐标归一化错误: {e}")
            return None
    
    def predict_chord(self, normalized_kps):
        if normalized_kps is None:
            return None, 0
        
        features = []
        for kp in normalized_kps:
            features.extend([kp[0], kp[1]])
        
        features = self.scaler.transform([features])
        
        # 获取预测概率
        probas = self.model.predict_proba(features)[0]
        pred_idx = np.argmax(probas)
        confidence = probas[pred_idx]
        
        return self.reverse_label_encoder.get(pred_idx, "Unknown"), confidence
    
    def smooth_prediction(self, prediction, confidence):
        """使用加权投票和历史记录平滑预测结果"""
        if prediction is None:
            return self.last_prediction, self.last_confidence
            
        # 只保留高置信度预测
        if confidence < 0.5:  # 置信度阈值
            return self.last_prediction, self.last_confidence
            
        self.prediction_history.append((prediction, confidence))
        
        # 计算加权投票
        vote_counts = {}
        for pred, conf in self.prediction_history:
            if pred not in vote_counts:
                vote_counts[pred] = 0
            vote_counts[pred] += conf
        
        if not vote_counts:
            return prediction, confidence
            
        # 选择得分最高的预测
        best_pred = max(vote_counts.items(), key=lambda x: x[1])
        avg_confidence = sum(conf for pred, conf in self.prediction_history if pred == best_pred[0]) / \
                        len([p for p in self.prediction_history if p[0] == best_pred[0]])
        
        return best_pred[0], avg_confidence
    
    def visualize(self, frame, fretboard_kps, hand_kps, prediction, confidence):
        # 绘制指板关键点和参考向量
        if fretboard_kps is not None:
            for i, kp in enumerate(fretboard_kps):
                x, y = int(kp[0]), int(kp[1])
                color = (0, 255, 0)  # 绿色
                if i == self.origin_idx:  # 点0用红色标记
                    color = (0, 0, 255)
                elif i == self.vec1_idx:  # 点1用蓝色标记
                    color = (255, 0, 0)
                elif i == self.vec2_idx:  # 点2用黄色标记
                    color = (0, 255, 255)
                cv2.circle(frame, (x, y), 8, color, -1)
                cv2.putText(frame, str(i), (x+10, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # 绘制参考向量
                if i == self.origin_idx:
                    if len(fretboard_kps) > self.vec1_idx:
                        x1, y1 = int(fretboard_kps[self.vec1_idx][0]), int(fretboard_kps[self.vec1_idx][1])
                        cv2.arrowedLine(frame, (x, y), (x1, y1), (255, 0, 0), 2)
                    if len(fretboard_kps) > self.vec2_idx:
                        x2, y2 = int(fretboard_kps[self.vec2_idx][0]), int(fretboard_kps[self.vec2_idx][1])
                        cv2.arrowedLine(frame, (x, y), (x2, y2), (0, 255, 255), 2)
        
        # 绘制手部关键点
        if hand_kps is not None:
            for kp in hand_kps:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        
        # 显示预测结果和置信度
        if prediction:
            text = f"Chord: {prediction} ({confidence:.2f})"
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)
            cv2.putText(frame, text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame
    
    def handle_prediction(self, frame, key, fretboard_kps, hand_kps, normalized_kps):
        # 预测和弦
        prediction, confidence = self.predict_chord(normalized_kps)
        smoothed_pred, smoothed_conf = self.smooth_prediction(prediction, confidence)
        
        # 更新最后预测结果
        if smoothed_pred:
            self.last_prediction = smoothed_pred
            self.last_confidence = smoothed_conf
        
        # 可视化
        frame = self.visualize(frame, fretboard_kps, hand_kps, self.last_prediction, self.last_confidence)
        return frame

    
    def run(self):
        try:
            self.start_capture()
            prev_time = time.time()
            fps = 0
            frame_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 检测关键点
                fretboard_kps, hand_kps = self.detect_landmarks(frame)
                normalized_kps = self.normalize_coordinates(fretboard_kps, hand_kps)
                
                # 预测和弦
                prediction, confidence = self.predict_chord(normalized_kps)
                smoothed_pred, smoothed_conf = self.smooth_prediction(prediction, confidence)
                
                # 更新最后预测结果
                if smoothed_pred:
                    self.last_prediction = smoothed_pred
                    self.last_confidence = smoothed_conf
                
                # 可视化
                frame = self.visualize(frame, fretboard_kps, hand_kps, self.last_prediction, self.last_confidence)
                
                # 计算FPS
                if frame_count % 10 == 0:
                    curr_time = time.time()
                    fps = 10 / (curr_time - prev_time)
                    prev_time = curr_time
                
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Real-time Chord Classifier", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists("chord_classifier/random_forest.joblib"):
        print("未找到训练好的模型，请先运行 train_chord_classifier.py")
    else:
        classifier = RealTimeChordClassifier(model_path="chord_classifier", video_capture=None)
        classifier.run()