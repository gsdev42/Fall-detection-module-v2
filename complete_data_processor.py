import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

class CompleteMobiActProcessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = []
        self.processed_data = None
        
        # Activity mappings
        self.adl_activities = {
            'CSI': 'Car Step-in',
            'CSO': 'Car Step-out', 
            'JOG': 'Jogging',
            'JUM': 'Jumping',
            'SCH': 'Sitting on Chair',
            'STD': 'Standing',
            'STN': 'Stand to Sit',
            'STU': 'Sit to Stand',
            'WAL': 'Walking'
        }
        
        self.fall_activities = {
            'FOL': 'Forward Fall',
            'FKL': 'Fall on Knees',
            'BSC': 'Backward Sitting Chair',
            'SDL': 'Sideways Fall'
        }
        
        self.sensors = ['acc', 'gyro', 'ori']
        
    def process_single_file(self, file_path, activity_name, sensor_type):
        """Process a single file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find @DATA marker
            data_start = None
            for i, line in enumerate(lines):
                if '@DATA' in line:
                    data_start = i + 1
                    break
            
            if data_start is None:
                return None
            
            # Parse CSV data after @DATA
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
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def count_total_files(self):
        """Count total files to process for progress bar"""
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
        """Load data from all subjects with progress bar"""
        print("🚀 Loading Complete MobiAct Dataset...")
        print("=" * 60)
        
        total_files = self.count_total_files()
        print(f"Total files to process: {total_files}")
        
        processed_count = 0
        failed_count = 0
        
        # Progress bar setup
        pbar = tqdm(total=total_files, desc="Processing files", unit="files")
        
        for subject_num in range(1, 10):
            subject_folder = f"sub{subject_num}"
            subject_path = os.path.join(self.dataset_path, subject_folder)
            
            if not os.path.exists(subject_path):
                continue
                
            subject_files = 0
            
            # Process ADL activities (label = 0)
            adl_count = self._process_activity_type_complete(
                subject_path, subject_num, 'ADL', 0, pbar
            )
            
            # Process FALLS activities (label = 1)  
            falls_count = self._process_activity_type_complete(
                subject_path, subject_num, 'FALLS', 1, pbar
            )
            
            subject_files = adl_count + falls_count
            processed_count += subject_files
            
            tqdm.write(f"✅ {subject_folder}: {subject_files} files processed")
        
        pbar.close()
        
        print(f"\n📊 Processing Summary:")
        print(f"  Total files processed: {processed_count}")
        print(f"  Total datasets created: {len(self.data)}")
        print(f"  Success rate: {len(self.data)/total_files*100:.1f}%")
        
    def _process_activity_type_complete(self, subject_path, subject_num, activity_type, label, pbar):
        """Process ADL or FALLS activities with progress tracking"""
        activity_path = os.path.join(subject_path, activity_type)
        
        if not os.path.exists(activity_path):
            return 0
        
        processed_count = 0
        
        activity_folders = [d for d in os.listdir(activity_path) 
                          if os.path.isdir(os.path.join(activity_path, d))]
        
        for folder_name in activity_folders:
            folder_path = os.path.join(activity_path, folder_name)
            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            
            for file_name in txt_files:
                file_path = os.path.join(folder_path, file_name)
                
                # Extract sensor type from filename
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
        """Combine sensor data efficiently"""
        print("\n🔗 Combining sensor data...")
        
        if not self.data:
            print("❌ No data to combine!")
            return
        
        # Create combination strategy
        print("Analyzing data structure...")
        
        file_info = {}
        for df in tqdm(self.data, desc="Analyzing files"):
            subject = df['subject'].iloc[0] 
            activity = df['activity'].iloc[0]
            file_name = df['file_name'].iloc[0]
            sensor = df['sensor_type'].iloc[0]
            
            # Create base key (remove sensor info)
            base_name = '_'.join(file_name.split('_')[:-2])  # Remove sensor_number.txt
            key = (subject, activity, base_name)
            
            if key not in file_info:
                file_info[key] = {}
            
            file_info[key][sensor] = df
        
        print(f"Found {len(file_info)} unique recording instances")
        
        # Combine sensors
        combined_data = []
        
        for key, sensors in tqdm(file_info.items(), desc="Combining sensors"):
            subject, activity, base_name = key
            
            # Find the sensor with most data to use as base
            base_sensor = None
            max_len = 0
            
            for sensor, df in sensors.items():
                if len(df) > max_len:
                    max_len = len(df)
                    base_sensor = sensor
            
            if base_sensor:
                combined_df = sensors[base_sensor].copy()
                
                # Add other sensors
                for sensor, df in sensors.items():
                    if sensor != base_sensor:
                        # Match lengths
                        min_len = min(len(combined_df), len(df))
                        
                        sensor_cols = [col for col in df.columns if col.startswith(sensor)]
                        for col in sensor_cols:
                            combined_df[col] = df[col].iloc[:min_len].values[:min_len]
                        
                        # Trim combined_df to match
                        combined_df = combined_df.iloc[:min_len]
                
                combined_data.append(combined_df)
        
        if combined_data:
            print("Concatenating final dataset...")
            self.processed_data = pd.concat(combined_data, ignore_index=True)
            print(f"✅ Combined dataset shape: {self.processed_data.shape}")
            
            # Show available sensor columns
            sensor_cols = [col for col in self.processed_data.columns 
                          if any(col.startswith(s) for s in self.sensors)]
            print(f"Available sensor features: {len(sensor_cols)}")
            
            # Show sensor coverage
            for sensor in self.sensors:
                sensor_count = len([col for col in sensor_cols if col.startswith(sensor)])
                print(f"  {sensor}: {sensor_count} features")
        
    def create_features_complete(self, window_size=50, overlap=0.5):
        """Create comprehensive features"""
        print(f"\n🔧 Creating features (window_size={window_size}, overlap={overlap})...")
        
        if self.processed_data is None:
            print("❌ No processed data available!")
            return
        
        features_list = []
        step_size = int(window_size * (1 - overlap))
        
        # Group data
        grouped = self.processed_data.groupby(['subject', 'activity', 'activity_type', 'label'])
        
        print(f"Processing {len(grouped)} activity groups...")
        
        for name, group in tqdm(grouped, desc="Creating features"):
            subject, activity, activity_type, label = name
            
            # Get sensor columns
            sensor_cols = [col for col in group.columns 
                          if any(col.startswith(sensor) for sensor in self.sensors)]
            
            if len(sensor_cols) == 0:
                continue
                
            sensor_data = group[sensor_cols].values
            
            # Create sliding windows
            num_windows = (len(sensor_data) - window_size) // step_size + 1
            
            for i in range(0, len(sensor_data) - window_size + 1, step_size):
                window_data = sensor_data[i:i + window_size]
                
                features = {}
                
                # Statistical features for each sensor axis
                for j, col in enumerate(sensor_cols):
                    col_data = window_data[:, j]
                    
                    # Basic statistics
                    features[f'{col}_mean'] = np.mean(col_data)
                    features[f'{col}_std'] = np.std(col_data)
                    features[f'{col}_min'] = np.min(col_data)
                    features[f'{col}_max'] = np.max(col_data)
                    features[f'{col}_median'] = np.median(col_data)
                    features[f'{col}_range'] = np.ptp(col_data)  # peak-to-peak
                    features[f'{col}_var'] = np.var(col_data)
                    features[f'{col}_rms'] = np.sqrt(np.mean(col_data**2))
                    
                    # Percentiles
                    features[f'{col}_q25'] = np.percentile(col_data, 25)
                    features[f'{col}_q75'] = np.percentile(col_data, 75)
                    features[f'{col}_iqr'] = np.percentile(col_data, 75) - np.percentile(col_data, 25)
                
                # Cross-sensor features (if multiple sensors available)
                acc_cols = [col for col in sensor_cols if col.startswith('acc')]
                if len(acc_cols) == 3:  # X, Y, Z axes
                    acc_data = window_data[:, [sensor_cols.index(col) for col in acc_cols]]
                    # Magnitude
                    magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
                    features['acc_magnitude_mean'] = np.mean(magnitude)
                    features['acc_magnitude_std'] = np.std(magnitude)
                    features['acc_magnitude_max'] = np.max(magnitude)
                
                gyro_cols = [col for col in sensor_cols if col.startswith('gyro')]
                if len(gyro_cols) == 3:
                    gyro_data = window_data[:, [sensor_cols.index(col) for col in gyro_cols]]
                    magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
                    features['gyro_magnitude_mean'] = np.mean(magnitude)
                    features['gyro_magnitude_std'] = np.std(magnitude)
                    features['gyro_magnitude_max'] = np.max(magnitude)
                
                # Metadata
                features['subject'] = subject
                features['activity'] = activity
                features['activity_type'] = activity_type
                features['label'] = label
                
                features_list.append(features)
        
        self.feature_data = pd.DataFrame(features_list)
        print(f"✅ Feature dataset created: {self.feature_data.shape}")
        
        # Show feature distribution
        feature_cols = [col for col in self.feature_data.columns 
                       if col not in ['subject', 'activity', 'activity_type', 'label']]
        print(f"Total features: {len(feature_cols)}")
        
    def prepare_final_dataset_complete(self):
        """Prepare and validate final dataset"""
        print("\n📊 Preparing final dataset...")
        
        if not hasattr(self, 'feature_data') or self.feature_data is None:
            print("❌ No feature data available!")
            return
        
        print("Dataset validation...")
        
        # Check for missing values
        missing_counts = self.feature_data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"⚠️  Found missing values: {missing_counts.sum()}")
            cols_with_missing = missing_counts[missing_counts > 0]
            for col, count in cols_with_missing.items():
                print(f"  {col}: {count} missing")
        
        # Separate features and metadata
        metadata_cols = ['subject', 'activity', 'activity_type', 'label']
        feature_cols = [col for col in self.feature_data.columns if col not in metadata_cols]
        
        X = self.feature_data[feature_cols].copy()
        y = self.feature_data['label'].copy()
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("Filling missing values with column means...")
            X = X.fillna(X.mean())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # Add back metadata
        for col in metadata_cols:
            X_scaled[col] = self.feature_data[col].values
        
        self.final_data = X_scaled
        self.scaler = scaler
        self.feature_columns = feature_cols
        
        # Final statistics
        print("\n✅ Final Dataset Ready!")
        print("=" * 40)
        print(f"Dataset shape: {self.final_data.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(self.final_data)}")
        
        print(f"\nClass Distribution:")
        class_counts = y.value_counts().sort_index()
        for label, count in class_counts.items():
            class_name = "Fall" if label == 1 else "Non-Fall (ADL)"
            percentage = (count / len(y)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\nSubjects: {sorted(self.final_data['subject'].unique())}")
        print(f"Activities: {sorted(self.final_data['activity'].unique())}")
        
        return self.final_data
    
    def save_complete_dataset(self, output_file='complete_mobiact_dataset.csv', 
                             save_metadata=True):
        """Save the complete processed dataset"""
        print(f"\n💾 Saving dataset to {output_file}...")
        
        if not hasattr(self, 'final_data') or self.final_data is None:
            print("❌ No final data to save!")
            return False
        
        try:
            # Save main dataset
            self.final_data.to_csv(output_file, index=False)
            print(f"✅ Main dataset saved: {output_file}")
            
            if save_metadata:
                # Save metadata and info
                metadata_file = output_file.replace('.csv', '_metadata.txt')
                with open(metadata_file, 'w') as f:
                    f.write("MobiAct Dataset Processing Metadata\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Dataset shape: {self.final_data.shape}\n")
                    f.write(f"Features: {len(self.feature_columns)}\n")
                    f.write(f"Samples: {len(self.final_data)}\n\n")
                    
                    f.write("Class Distribution:\n")
                    class_counts = self.final_data['label'].value_counts().sort_index()
                    for label, count in class_counts.items():
                        class_name = "Fall" if label == 1 else "Non-Fall (ADL)"
                        percentage = (count / len(self.final_data)) * 100
                        f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
                    
                    f.write(f"\nSubjects: {sorted(self.final_data['subject'].unique())}\n")
                    f.write(f"Activities: {sorted(self.final_data['activity'].unique())}\n")
                    
                    f.write(f"\nFeature Columns ({len(self.feature_columns)}):\n")
                    for i, col in enumerate(self.feature_columns, 1):
                        f.write(f"  {i:3d}. {col}\n")
                
                print(f"✅ Metadata saved: {metadata_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            return False

def process_complete_mobiact_dataset(dataset_path='mobiAct_dataset',
                                   window_size=50,
                                   overlap=0.5,
                                   output_file='complete_mobiact_dataset.csv'):
    """
    Process the complete MobiAct dataset
    """
    
    print("🚀 COMPLETE MOBIACT DATASET PROCESSING")
    print("=" * 60)
    print(f"📁 Dataset path: {dataset_path}")
    print(f"🔧 Window size: {window_size}")
    print(f"🔄 Overlap: {overlap}")
    print(f"💾 Output file: {output_file}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialize processor
        processor = CompleteMobiActProcessor(dataset_path)
        
        # Step 1: Load all data
        processor.load_complete_data()
        
        if not processor.data:
            print("❌ No data loaded!")
            return None
        
        # Step 2: Combine sensor data
        processor.combine_sensor_data_complete()
        
        if processor.processed_data is None:
            print("❌ Failed to combine sensor data!")
            return None
        
        # Step 3: Create features
        processor.create_features_complete(window_size=window_size, overlap=overlap)
        
        if not hasattr(processor, 'feature_data'):
            print("❌ Failed to create features!")
            return None
        
        # Step 4: Prepare final dataset
        final_data = processor.prepare_final_dataset_complete()
        
        if final_data is None:
            print("❌ Failed to prepare final dataset!")
            return None
        
        # Step 5: Save dataset
        success = processor.save_complete_dataset(output_file)
        
        if not success:
            print("❌ Failed to save dataset!")
            return None
        
        # Processing complete
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n🎉 PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"⏱️  Total processing time: {processing_time/60:.1f} minutes")
        print(f"📄 Output file: {output_file}")
        print(f"📊 Dataset ready for machine learning!")
        
        return processor, final_data
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Process complete dataset
    result = process_complete_mobiact_dataset(
        dataset_path='mobiAct_dataset',
        window_size=50,
        overlap=0.5,
        output_file='complete_mobiact_dataset.csv'
    )
    
    if result:
        processor, data = result
        print("\n✅ Dataset is ready for machine learning!")
        print("Next: Run the model training script")
    else:
        print("❌ Processing failed!")