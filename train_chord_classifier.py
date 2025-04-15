import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import joblib
from tqdm import tqdm

class ChordClassifierTrainer:
    def __init__(self, data_dir="guitar_data"):
        self.data_dir = data_dir
        self.model = RandomForestClassifier(
            n_estimators=200,  # 增加树的数量
            max_depth=15,      # 限制最大深度防止过拟合
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'  # 处理类别不平衡
        )
        self.scaler = StandardScaler()
        self.label_encoder = {}
        
    def load_data_(self):
        X, y = [], []
        
        # 遍历数据目录中的所有CSV文件
        for filename in tqdm(os.listdir(self.data_dir)):
            if not filename.endswith('.csv'):
                continue
                
            chord = filename.split('_')[-1].replace('.csv', '')
            if chord not in self.label_encoder:
                self.label_encoder[chord] = len(self.label_encoder)
                
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            
            for _, row in df.iterrows():
                features = []
                for i in range(21):  # 21个手部关键点
                    features.extend([row[f'kp{i}_x'], row[f'kp{i}_y']])
                
                X.append(features)
                y.append(self.label_encoder[chord])
        
        return np.array(X), np.array(y)
    

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        
        # 遍历数据目录中的所有CSV文件
        for filename in tqdm(os.listdir(self.data_dir)):
            if not filename.endswith('.csv'):
                continue
                
            chord = filename.split('_')[-1].replace('.csv', '')
            if chord not in self.label_encoder:
                self.label_encoder[chord] = len(self.label_encoder)
                
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            
            for _, row in df.iterrows():
                features = []
                for i in range(21):  # 21个手部关键点
                    # 直接从CSV读取kp{i}_x和kp{i}_y
                    features.extend([row[f'kp{i}_x'], row[f'kp{i}_y']])
                
                X.append(features)
                y.append(self.label_encoder[chord])
        
        return np.array(X), np.array(y)
    
    def train(self):
        print("正在加载数据...")
        X, y = self.load_data()
        
        if len(X) == 0:
            raise ValueError("没有找到训练数据！请先收集数据。")
        
        print(f"加载完成，共 {len(X)} 个样本，{len(self.label_encoder)} 个类别")
        
        # 数据标准化
        print("标准化数据...")
        X = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("开始训练模型...")
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n模型准确率: {accuracy:.2f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.keys()))
        
        # 特征重要性分析
        if hasattr(self.model, 'feature_importances_'):
            print("\n特征重要性分析:")
            important_features = np.argsort(self.model.feature_importances_)[-10:]  # 取最重要的10个特征
            for idx in important_features:
                kp_idx = idx // 2
                coord_type = "x" if idx % 2 == 0 else "y"
                print(f"关键点 {kp_idx} 的 {coord_type} 坐标: {self.model.feature_importances_[idx]:.4f}")
        
        return accuracy
    
    def save_model(self, model_path="chord_classifier"):
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_path, "random_forest.joblib"))
        joblib.dump(self.scaler, os.path.join(model_path, "scaler.joblib"))
        joblib.dump(self.label_encoder, os.path.join(model_path, "label_encoder.joblib"))
        print(f"模型已保存到 {model_path} 目录")
    
    @classmethod
    def load_model(cls, model_path="chord_classifier"):
        model = joblib.load(os.path.join(model_path, "random_forest.joblib"))
        scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
        label_encoder = joblib.load(os.path.join(model_path, "label_encoder.joblib"))
        
        trainer = cls()
        trainer.model = model
        trainer.scaler = scaler
        trainer.label_encoder = label_encoder
        return trainer

if __name__ == "__main__":
    trainer = ChordClassifierTrainer()
    trainer.train()
    trainer.save_model()