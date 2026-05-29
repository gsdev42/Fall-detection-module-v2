import os
import time
from datetime import datetime
import numpy as np

def main():
    print("COMPLETE FALL DETECTION PIPELINE (CNN)")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    DATASET_PATH = 'mobiAct_dataset'
    PROCESSED_DATA_FILE = 'mobiact_cnn_data.npz'
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path '{DATASET_PATH}' not found!")
        return
    
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output file: {PROCESSED_DATA_FILE}")
    
    print("\nSelect an option:")
    print("1. Run complete pipeline (data processing + CNN training)")
    print("2. Only data processing (raw windows for CNN)")
    print("3. Only CNN training (requires existing .npz file)")
    print("4. Quick test (2 subjects + CNN)")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\nPipeline cancelled by user")
        return
    
    start_time = time.time()
    
    if choice == '1':
        run_complete_pipeline(DATASET_PATH, PROCESSED_DATA_FILE)
    elif choice == '2':
        run_data_processing(DATASET_PATH, PROCESSED_DATA_FILE)
    elif choice == '3':
        if os.path.exists(PROCESSED_DATA_FILE):
            run_cnn_training(PROCESSED_DATA_FILE)
        else:
            print(f"Processed data file '{PROCESSED_DATA_FILE}' not found!")
            print("Please run data processing first (option 1 or 2)")
    elif choice == '4':
        run_quick_test(DATASET_PATH)
    else:
        print("Invalid choice. Please select 1-4.")
        return
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nPIPELINE COMPLETED!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_complete_pipeline(dataset_path, output_file):
    print("\n" + "="*50)
    print("RUNNING COMPLETE PIPELINE (DATA + CNN)")
    print("="*50)
    
    print("\nSTEP 1: DATA PROCESSING (RAW WINDOWS)")
    print("-" * 30)
    success = run_data_processing(dataset_path, output_file)
    if not success:
        print("Data processing failed! Stopping pipeline.")
        return False
    
    print("\nSTEP 2: CNN TRAINING")
    print("-" * 30)
    success = run_cnn_training(output_file)
    if success:
        print("\nCOMPLETE PIPELINE SUCCESSFUL!")
        print("Generated files:")
        print("   - mobiact_cnn_data.npz (raw windows)")
        print("   - cnn_training_history.png")
        print("   - cnn_confusion_matrix.png")
        print("   - cnn_roc_curve.png")
        print("   - best_cnn_fall_detection.h5")
        return True
    else:
        print("CNN training failed!")
        return False

def run_data_processing(dataset_path, output_file):
    print("Processing raw windows for CNN...")
    try:
        from complete_data_processor import process_for_cnn
        processor = process_for_cnn(
            dataset_path=dataset_path,
            window_size=50,
            overlap=0.5,
            scale=True,
            output_prefix=output_file.replace('.npz', '')
        )
        if processor:
            print("Data processing completed successfully!")
            return True
        else:
            print("Data processing failed!")
            return False
    except ImportError:
        print("Data processor module not found!")
        print("Make sure 'complete_data_processor.py' is in the same directory.")
        return False
    except Exception as e:
        print(f"Data processing error: {e}")
        return False

def run_cnn_training(data_file):
    print(f"Starting CNN training with data file: {data_file}")
    try:
        from ml_model_trainer_cnn import FallDetectionCNN
        trainer = FallDetectionCNN(data_file)
        trainer.run_pipeline(test_size=0.2, val_size=0.2, epochs=50, batch_size=32)
        print("CNN training completed successfully!")
        return True
    except ImportError:
        print("CNN trainer module not found!")
        print("Make sure 'ml_model_trainer_cnn.py' is in the same directory.")
        return False
    except Exception as e:
        print(f"CNN training error: {e}")
        return False

def run_quick_test(dataset_path):
    print("Running quick test with 2 subjects...")
    try:
        from complete_data_processor import process_for_cnn
        from ml_model_trainer_cnn import FallDetectionCNN
        
        print("\nProcessing test data...")
        processor = process_for_cnn(
            dataset_path=dataset_path,
            window_size=30,
            overlap=0.3,
            scale=True,
            output_prefix='quick_test_cnn'
        )
        if not processor:
            print("Quick test data processing failed!")
            return False
        
        print("Quick test data processed!")
        
        print("\nTraining CNN on test data...")
        trainer = FallDetectionCNN('quick_test_cnn_data.npz')
        trainer.run_pipeline(test_size=0.3, val_size=0.2, epochs=20, batch_size=16)
        
        print("\nQuick test completed successfully!")
        print("Generated test files:")
        print("   - quick_test_cnn_data.npz")
        print("   - cnn_training_history.png")
        print("   - cnn_confusion_matrix.png")
        print("   - cnn_roc_curve.png")
        return True
    except ImportError as e:
        print(f"Module import error: {e}")
        return False
    except Exception as e:
        print(f"Quick test error: {e}")
        return False

def check_requirements():
    required_modules = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'tqdm', 'joblib', 'tensorflow']
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    if missing_modules:
        print("Missing required modules:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\nInstall them using:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    return True

def print_system_info():
    import platform
    import sys
    print("\nSYSTEM INFORMATION")
    print("-" * 30)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Current directory: {os.getcwd()}")

if __name__ == "__main__":
    if not check_requirements():
        print("\nPlease install missing modules before running the pipeline.")
        exit(1)
    print_system_info()
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
