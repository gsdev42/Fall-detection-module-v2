import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

class CompleteMobiActProcessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = []
        self.processed_data = None
        
        self.adl_activities = {
            'CSI': 'Car Step-in', 'CSO': 'Car Step-out', 'JOG': 'Jogging',
            'JUM': 'Jumping', 'SCH': 'Sitting on Chair', 'STD': 'Standing',
            'STN': 'Stand to Sit', 'STU': 'Sit to Stand', 'WAL': 'Walking'
        }
        
        self.fall_activities = {
            'FOL': 'Forward Fall', 'FKL': 'Fall on Knees',
            'BSC': 'Backward Sitting Chair', 'SDL': 'Sideways Fall'
        }
        
        self.sensors = ['acc', 'gyro', 'ori']
    
    def process_single_file(self, file_path, activity_name, sensor_type):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            data_start = None
            for i, line in enumerate(lines):
                if '@DATA' in line:
                    data_start = i + 1
                    break
            
            if data_start is None:
                return None
            
            data_rows = []
            for line in lines[data_start:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            x_val = float(parts[1].strip())
                            y_val = float(parts[2].strip())
                            z_val = float(parts[3].strip())
                            data_rows.append([x_val, y_val, z_val])
                        except ValueError:
                            continue
            
            if data_rows:
                df = pd.DataFrame(data_rows, columns=[f'{sensor_type}_x', f'{sensor_type}_y', f'{sensor_type}_z'])
                df['activity'] = activity_name
                df['sensor_type'] = sensor_type
                df['file_name'] = os.path.basename(file_path)
                return df
            return None
        except Exception:
            return None
    
    def count_total_files(self):
        total_files = 0
        for subject_num in range(1, 10):
            subject_folder = f"sub{subject_num}"
            subject_path = os.path.join(self.dataset_path, subject_folder)
            if os.path.exists(subject_path):
                for activity_type in ['ADL', 'FALLS']:
                    activity_path = os.path.join(subject_path, activity_type)
                    if os.path.exists(activity_path):
                        for folder_name in os.listdir(activity_path):
                            folder_path = os.path.join(activity_path, folder_name)
                            if os.path.isdir(folder_path):
                                txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
                                total_files += len(txt_files)
        return total_files
    
    def load_complete_data(self):
        print("Loading Complete MobiAct Dataset...")
        total_files = self.count_total_files()
        processed_count = 0
        pbar = tqdm(total=total_files, desc="Processing files", unit="files")
        
        for subject_num in range(1, 10):
            subject_folder = f"sub{subject_num}"
            subject_path = os.path.join(self.dataset_path, subject_folder)
            if not os.path.exists(subject_path):
                continue
            adl_count = self._process_activity_type_complete(subject_path, subject_num, 'ADL', 0, pbar)
            falls_count = self._process_activity_type_complete(subject_path, subject_num, 'FALLS', 1, pbar)
            processed_count += adl_count + falls_count
            tqdm.write(f"Finished {subject_folder}: {adl_count + falls_count} files")
        pbar.close()
        print(f"Total files processed: {processed_count}")
    
    def _process_activity_type_complete(self, subject_path, subject_num, activity_type, label, pbar):
        activity_path = os.path.join(subject_path, activity_type)
        if not os.path.exists(activity_path):
            return 0
        processed_count = 0
        activity_folders = [d for d in os.listdir(activity_path) if os.path.isdir(os.path.join(activity_path, d))]
        for folder_name in activity_folders:
            folder_path = os.path.join(activity_path, folder_name)
            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            for file_name in txt_files:
                file_path = os.path.join(folder_path, file_name)
                sensor = None
                for s in self.sensors:
                    if f'_{s}_' in file_name:
                        sensor = s
                        break
                if sensor:
                    df = self.process_single_file(file_path, folder_name, sensor)
                    if df is not None:
                        df['subject'] = subject_num
                        df['activity_type'] = activity_type
                        df['label'] = label
                        self.data.append(df)
                        processed_count += 1
                pbar.update(1)
        return processed_count
    
    def combine_sensor_data_complete(self):
        print("Combining sensor data...")
        if not self.data:
            return
        file_info = {}
        for df in tqdm(self.data, desc="Analyzing files"):
            subject = df['subject'].iloc[0]
            activity = df['activity'].iloc[0]
            file_name = df['file_name'].iloc[0]
            sensor = df['sensor_type'].iloc[0]
            base_name = '_'.join(file_name.split('_')[:-2])
            key = (subject, activity, base_name)
            if key not in file_info:
                file_info[key] = {}
            file_info[key][sensor] = df
        
        combined_data = []
        for key, sensors in tqdm(file_info.items(), desc="Combining sensors"):
            subject, activity, base_name = key
            base_sensor = None
            max_len = 0
            for sensor, df in sensors.items():
                if len(df) > max_len:
                    max_len = len(df)
                    base_sensor = sensor
            if base_sensor:
                combined_df = sensors[base_sensor].copy()
                for sensor, df in sensors.items():
                    if sensor != base_sensor:
                        min_len = min(len(combined_df), len(df))
                        sensor_cols = [col for col in df.columns if col.startswith(sensor)]
                        for col in sensor_cols:
                            combined_df[col] = df[col].iloc[:min_len].values[:min_len]
                        combined_df = combined_df.iloc[:min_len]
                combined_data.append(combined_df)
        
        if combined_data:
            self.processed_data = pd.concat(combined_data, ignore_index=True)
            print(f"Combined dataset shape: {self.processed_data.shape}")
    
    def extract_raw_windows_for_cnn(self, window_size=50, overlap=0.5):
        print(f"Extracting raw windows (window_size={window_size}, overlap={overlap}) for CNN...")
        if self.processed_data is None:
            return
        
        step_size = int(window_size * (1 - overlap))
        grouped = self.processed_data.groupby(['subject', 'activity', 'activity_type', 'label'])
        
        X_windows = []
        y_labels = []
        subject_ids = []
        activity_names = []
        
        for name, group in tqdm(grouped, desc="Extracting windows"):
            subject, activity, activity_type, label = name
            sensor_cols = [col for col in group.columns if any(col.startswith(s) for s in self.sensors)]
            if len(sensor_cols) == 0:
                continue
            sensor_data = group[sensor_cols].values
            
            for i in range(0, len(sensor_data) - window_size + 1, step_size):
                window = sensor_data[i:i + window_size]
                X_windows.append(window)
                y_labels.append(label)
                subject_ids.append(subject)
                activity_names.append(activity)
        
        self.X_raw = np.array(X_windows)
        self.y_raw = np.array(y_labels)
        self.subject_raw = np.array(subject_ids)
        self.activity_raw = np.array(activity_names)
        
        print(f"Raw windows shape: {self.X_raw.shape} (samples, timesteps, channels)")
        print(f"Number of channels: {self.X_raw.shape[2]}")
    
    def scale_raw_windows(self):
        print("Scaling raw windows per channel...")
        if not hasattr(self, 'X_raw') or self.X_raw is None:
            return
        
        n_samples, timesteps, n_channels = self.X_raw.shape
        X_reshaped = self.X_raw.reshape(-1, n_channels)
        self.scaler = StandardScaler()
        X_scaled_reshaped = self.scaler.fit_transform(X_reshaped)
        self.X_scaled = X_scaled_reshaped.reshape(n_samples, timesteps, n_channels)
        print(f"Scaled windows shape: {self.X_scaled.shape}")
    
    def prepare_cnn_dataset(self, window_size=50, overlap=0.5, scale=True):
        self.extract_raw_windows_for_cnn(window_size, overlap)
        if scale:
            self.scale_raw_windows()
        else:
            self.X_scaled = self.X_raw
        return self.X_scaled, self.y_raw, self.subject_raw, self.activity_raw
    
    def save_cnn_dataset(self, output_prefix='mobiact_cnn'):
        if not hasattr(self, 'X_scaled') or self.X_scaled is None:
            print("No data to save")
            return False
        np.savez_compressed(f'{output_prefix}_data.npz',
                           X=self.X_scaled,
                           y=self.y_raw,
                           subject=self.subject_raw,
                           activity=self.activity_raw)
        print(f"Saved {output_prefix}_data.npz")
        return True

def process_for_cnn(dataset_path='mobiAct_dataset', window_size=50, overlap=0.5, scale=True, output_prefix='mobiact_cnn'):
    print("Processing MobiAct for CNN input")
    start_time = time.time()
    processor = CompleteMobiActProcessor(dataset_path)
    processor.load_complete_data()
    if not processor.data:
        print("No data loaded")
        return None
    processor.combine_sensor_data_complete()
    if processor.processed_data is None:
        print("Failed to combine sensors")
        return None
    X, y, subjects, activities = processor.prepare_cnn_dataset(window_size, overlap, scale)
    processor.save_cnn_dataset(output_prefix)
    elapsed = time.time() - start_time
    print(f"Processing completed in {elapsed/60:.1f} minutes")
    print(f"CNN input shape: {X.shape}")
    return processor

if __name__ == "__main__":
    processor = process_for_cnn(
        dataset_path='mobiAct_dataset',
        window_size=50,
        overlap=0.5,
        scale=True,
        output_prefix='mobiact_cnn'
    )
