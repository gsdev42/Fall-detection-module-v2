import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import joblib
import warnings
warnings.filterwarnings('ignore')

class FallDetectionCNN:
    def __init__(self, data_file):
        self.data_file = data_file
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.history = None
        self.results = {}
        
    def load_data(self):
        data = np.load(self.data_file)
        self.X = data['X']
        self.y = data['y']
        print(f"Loaded data shape: {self.X.shape}")
        print(f"Labels shape: {self.y.shape}")
        print(f"Class distribution: {np.bincount(self.y)}")
        return True
    
    def prepare_data(self, test_size=0.2, val_size=0.2, random_state=42):
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=random_state
        )
        val_relative = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=random_state
        )
        print(f"Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        return True
    
    def build_cnn(self, input_shape, num_classes=2):
        model = Sequential([
            Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        print(model.summary())
        return model
    
    def train_model(self, epochs=50, batch_size=32):
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        return self.history
    
    def evaluate(self):
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        self.results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'auc': roc_auc_score(self.y_test, y_pred_proba[:, 1]),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba[:, 1]
        }
        print(f"Test Accuracy: {self.results['accuracy']:.4f}")
        print(f"F1 Score: {self.results['f1']:.4f}")
        print(f"AUC: {self.results['auc']:.4f}")
        return self.results
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history.history['accuracy'], label='Train Acc')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Acc')
        axes[0].set_title('Accuracy')
        axes[0].legend()
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Loss')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig('cnn_training_history.png')
        plt.show()
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.results['y_pred'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
        plt.title('Confusion Matrix - CNN')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig('cnn_confusion_matrix.png')
        plt.show()
    
    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.results['y_pred_proba'])
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"CNN (AUC = {self.results['auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('cnn_roc_curve.png')
        plt.show()
    
    def save_model(self, filename='best_cnn_fall_detection.h5'):
        self.model.save(filename)
        joblib.dump(self.results, 'cnn_results.pkl')
        print(f"Model saved as {filename}")
    
    def run_pipeline(self, test_size=0.2, val_size=0.2, epochs=50, batch_size=32):
        self.load_data()
        self.prepare_data(test_size, val_size)
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        self.build_cnn(input_shape)
        self.train_model(epochs, batch_size)
        self.evaluate()
        self.plot_training_history()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.save_model()

if __name__ == "__main__":
    cnn_trainer = FallDetectionCNN('mobiact_cnn_data.npz')
    cnn_trainer.run_pipeline(test_size=0.2, val_size=0.2, epochs=50, batch_size=32)
