import cv2
import numpy as np
import time
import os
import csv
from ultralytics import YOLO
import mediapipe as mp
from collections import defaultdict
from datetime import datetime

class GuitarPerformanceCollector:
    def __init__(self, yolo_model_path, output_dir, video_capture):
        if output_dir is None:
            output_dir="guitar_data"
        # 初始化模型
        self.yolo_model = YOLO(yolo_model_path)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.cap = video_capture
        
        # 数据存储
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 状态变量
        self.is_recording = False
        self.current_chord = None
        self.collected_chord_data : list = []

        # 指板参考点索引 (使用点0作为原点，点0到点1和点0到点2作为基向量)
        self.origin_idx = 0  # 点0
        self.vec1_idx = 1    # 点1
        self.vec2_idx = 2    # 点2
        
        # 手部跟踪状态
        self.last_hand_kps = None  # 存储最后有效的手部关键点
        self.missing_hand_frames = 0  # 统计连续未检测到手的帧数
        self.max_missing_frames = 10  # 最多允许10帧（约0.33秒@30FPS）未检测到手
    
    def detect_landmarks(self, frame):
        # YOLO指板检测
        yolo_results = self.yolo_model.predict(source=frame, show=False, save=False, verbose=False)
        fretboard_kps = yolo_results[0].keypoints.to('cpu').data.numpy()[0] if yolo_results[0].keypoints else None
        
        # MediaPipe手部检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.mp_hands.process(rgb_frame)
        hand_kps = None
        
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for idx, handedness in enumerate(hand_results.multi_handedness):
                # 检查手是否标记为“Right”（镜像后的左手）
                if handedness.classification[0].label == "Right":
                    hand_kps = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] 
                                        for lm in hand_results.multi_hand_landmarks[idx].landmark])
                    # 更新最后有效关键点并重置未检测帧计数
                    self.last_hand_kps = hand_kps
                    self.missing_hand_frames = 0
                    break  # 优先处理第一个“Right”手
        
        # 如果未检测到“Right”手，使用最后有效关键点（如果可用）
        if hand_kps is None:
            self.missing_hand_frames += 1
            if self.last_hand_kps is not None and self.missing_hand_frames <= self.max_missing_frames:
                hand_kps = self.last_hand_kps  # 重用最后有效关键点
            else:
                self.last_hand_kps = None  # 长时间未检测到后清除
        
        return fretboard_kps, hand_kps
    
    def normalize_coordinates(self, fretboard_kps, hand_kps):
        """
        使用点0作为原点，点0到点1和点0到点2作为基向量
        构建参考坐标系并归一化手部关键点
        """
        if fretboard_kps is None or hand_kps is None or len(fretboard_kps) < 3:
            return None
        
        try:
            # 获取参考点和向量
            origin = fretboard_kps[self.origin_idx][:2]
            vec1 = fretboard_kps[self.vec1_idx][:2] - origin
            vec2 = fretboard_kps[self.vec2_idx][:2] - origin
            
            # 计算基向量的长度作为归一化因子
            len_vec1 = np.linalg.norm(vec1)
            len_vec2 = np.linalg.norm(vec2)
            
            if len_vec1 < 1e-6 or len_vec2 < 1e-6:
                return None
                
            # 归一化基向量
            vec1 = vec1 / len_vec1
            vec2 = vec2 / len_vec2
            
            # 构建变换矩阵
            transform_mat = np.vstack([vec1, vec2]).T
            
            # 计算手部关键点在新坐标系中的坐标
            normalized_kps = []
            for kp in hand_kps:
                # 计算相对于原点的向量
                relative_vec = kp - origin
                
                # 投影到新的坐标系
                proj_coords = np.linalg.solve(transform_mat, relative_vec)
                normalized_kps.append(proj_coords)
            
            return np.array(normalized_kps)
            
        except Exception as e:
            print(f"坐标归一化错误: {e}")
            return None
    
    def start_recording(self, chord_name):
        self.is_recording = True
        self.current_chord = chord_name
        print(f"开始记录和弦: {chord_name}")
    
    def stop_recording(self):
        if not self.current_chord or not self.collected_chord_data:
            print("没有数据需要保存")
            return
            
        filename = os.path.join(self.output_dir, f"{self.session_id}_{self.current_chord}.csv")
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                print(self.collected_chord_data)
                # 写入表头
                headers = []
                for i in range(21):  # 21个手部关键点
                    headers.extend([f'kp{i}_x', f'kp{i}_y'])
                writer.writerow(headers)
                
                # 写入数据
                for sample in self.collected_chord_data:
                    row = []
                    for i in range(21):  # 21个手部关键点
                        kp = sample[i] if i < len(sample) else [0, 0]
                        row.extend(kp)
                    writer.writerow(row)
                
            print(f"已保存 {len(self.collected_chord_data)} 个样本到 {filename}")
            
        except Exception as e:
            print(f"保存CSV文件时出错: {e}")
        
        finally:
            self.is_recording = False
            self.current_chord = None
            self.collected_chord_data = []  # 清空当前和弦数据
    
    def visualize(self, frame, fretboard_kps, hand_kps, normalized_kps=None):
        # 绘制指板关键点
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
        
        # 显示状态信息
        status = f"Recording: {self.is_recording}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if self.current_chord:
            cv2.putText(frame, f"Chord: {self.current_chord}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 显示归一化信息
        if normalized_kps is not None and len(normalized_kps) > 0:
            sample_kp = normalized_kps[0]
            coord_text = f"Norm coords: ({sample_kp[0]:.2f}, {sample_kp[1]:.2f})"
            cv2.putText(frame, coord_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        pass
        # try:
        #     prev_time = time.time()
        #     frame_count = 0
        #     fps = 0
        #     
        #     while True:
        #         ret, frame = self.cap.read()
        #         key = cv2.waitKey(1) & 0xFF
        #         if not ret:
        #             print("无法获取视频帧")
        #             break
        #         
        #         frame_count += 1
        #         
        #         frame = self.handle_landmarks(frame, key)
        #         time.sleep(0.1)  # 控制采样频率
        #         
        #         # 计算FPS
        #         if frame_count % 10 == 0:
        #             curr_time = time.time()
        #             fps = 10 / (curr_time - prev_time)
        #             prev_time = curr_time
        #         
        #         cv2.putText(frame, f"FPS: {int(fps)}", (10, 150), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #         
        #         
        #         # 键盘控制
        #         if key == ord('q'):
        #             break
        #         elif ord('1') <= key <= ord('9'):
        #             chord_name = f"C{chr(key)}"
        #             self.start_recording(chord_name)
        #         elif key == ord('s'):
        #             self.stop_recording()
        # 
        # finally:
        #     self.destroy()

    def handle_landmarks(self, frame, key, fretboard_kps=None, hand_kps=None, normalized_kps=None):
        # 如果检测到两者，则记录数据
        if self.is_recording and fretboard_kps is not None and hand_kps is not None:
            if normalized_kps is not None:
                self.collected_chord_data.append(normalized_kps.tolist())
        # 可视化
        frame = self.visualize(frame, fretboard_kps, hand_kps, normalized_kps)
        # 处理键盘事件
        if key == ord('d'):
            if normalized_kps is not None:
                print(f"当前归一化坐标 (第一个关键点): {normalized_kps[0]}")
        return frame
  
    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()
        if self.is_recording:
            self.stop_recording()
        self.last_hand_kps = None
        self.missing_hand_frames = 0

if __name__ == "__main__":
    collector = GuitarPerformanceCollector(
        yolo_model_path="fretboard-detection-200epochs.pt",
        output_dir="guitar_data"
    )
    
    collector.start_capture()
    collector.run()