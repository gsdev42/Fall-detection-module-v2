"""
Complete Fall Detection Pipeline
================================

This script combines data processing and model training into a single pipeline.
It processes the raw MobiAct dataset and trains multiple machine learning models
for fall detection.

Usage:
    python complete_pipeline.py

Author: Assistant
Date: 2024
"""

import os
import time
from datetime import datetime

def main():
    print("🚀 COMPLETE FALL DETECTION PIPELINE")
    print("=" * 70)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Configuration
    DATASET_PATH = 'mobiAct_dataset'
    PROCESSED_DATA_FILE = 'complete_mobiact_dataset.csv'
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path '{DATASET_PATH}' not found!")
        print("Please make sure your MobiAct dataset folder is in the current directory.")
        return
    
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"💾 Output file: {PROCESSED_DATA_FILE}")
    
    # Ask user what to do
    print("\nSelect an option:")
    print("1. Run complete pipeline (data processing + model training)")
    print("2. Only data processing")
    print("3. Only model training (requires existing processed data)")
    print("4. Quick test (process 2 subjects + train models)")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 Pipeline cancelled by user")
        return
    
    start_time = time.time()
    
    if choice == '1':
        # Complete pipeline
        run_complete_pipeline(DATASET_PATH, PROCESSED_DATA_FILE)
        
    elif choice == '2':
        # Only data processing
        run_data_processing(DATASET_PATH, PROCESSED_DATA_FILE)
        
    elif choice == '3':
        # Only model training
        if os.path.exists(PROCESSED_DATA_FILE):
            run_model_training(PROCESSED_DATA_FILE)
        else:
            print(f"❌ Processed data file '{PROCESSED_DATA_FILE}' not found!")
            print("Please run data processing first (option 1 or 2)")
            
    elif choice == '4':
        # Quick test
        run_quick_test(DATASET_PATH)
        
    else:
        print("❌ Invalid choice. Please select 1-4.")
        return
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n⏱️  PIPELINE COMPLETED!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_complete_pipeline(dataset_path, output_file):
    """Run complete data processing + model training pipeline"""
    print("\n" + "="*50)
    print("RUNNING COMPLETE PIPELINE")
    print("="*50)
    
    # Step 1: Data Processing
    print("\n🔄 STEP 1: DATA PROCESSING")
    print("-" * 30)
    
    success = run_data_processing(dataset_path, output_file)
    
    if not success:
        print("❌ Data processing failed! Stopping pipeline.")
        return False
    
    # Step 2: Model Training
    print("\n🔄 STEP 2: MODEL TRAINING")
    print("-" * 30)
    
    success = run_model_training(output_file)
    
    if success:
        print("\n🎉 COMPLETE PIPELINE SUCCESSFUL!")
        print("📁 Generated files:")
        print("   - complete_mobiact_dataset.csv (processed data)")
        print("   - complete_mobiact_dataset_metadata.txt (data info)")
        print("   - fall_detection_results.png (model comparison)")
        print("   - feature_importance_*.png (feature analysis)")
        print("   - best_fall_detection_model_*.pkl (saved model)")
        return True
    else:
        print("❌ Model training failed!")
        return False

def run_data_processing(dataset_path, output_file):
    """Run only data processing"""
    print("Starting data processing...")
    
    try:
        # Import data processor
        from complete_data_processor import process_complete_mobiact_dataset
        
        # Process complete dataset
        result = process_complete_mobiact_dataset(
            dataset_path=dataset_path,
            window_size=50,
            overlap=0.5,
            output_file=output_file
        )
        
        if result:
            print("✅ Data processing completed successfully!")
            return True
        else:
            print("❌ Data processing failed!")
            return False
            
    except ImportError:
        print("❌ Data processor module not found!")
        print("Make sure 'complete_data_processor.py' is in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

def run_model_training(data_file):
    """Run only model training"""
    print(f"Starting model training with data file: {data_file}")
    
    try:
        # Import model trainer
        from ml_model_trainer import train_fall_detection_models
        
        # Train models
        trainer = train_fall_detection_models(
            data_file=data_file,
            test_size=0.2,
            feature_selection='k_best',  # Use feature selection
            n_features=100,              # Select top 100 features
            save_model=True
        )
        
        if trainer:
            print("✅ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed!")
            return False
            
    except ImportError:
        print("❌ Model trainer module not found!")
        print("Make sure 'ml_model_trainer.py' is in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def run_quick_test(dataset_path):
    """Run quick test with 2 subjects"""
    print("Running quick test with 2 subjects...")
    
    try:
        # Import modules
        from final_working_preprocessor import process_mobiact_dataset
        from ml_model_trainer import train_fall_detection_models
        
        # Process test data
        print("\n📁 Processing test data...")
        result = process_mobiact_dataset(
            dataset_path=dataset_path,
            subjects=[1, 2],  # Only 2 subjects for quick test
            window_size=30,   # Smaller window
            overlap=0.3,      # Less overlap
            output_file='quick_test_data.csv'
        )
        
        if not result:
            print("❌ Quick test data processing failed!")
            return False
        
        print("✅ Quick test data processed!")
        
        # Train models on test data
        print("\n🤖 Training models on test data...")
        trainer = train_fall_detection_models(
            data_file='quick_test_data.csv',
            test_size=0.3,    # Larger test set for small data
            feature_selection='k_best',
            n_features=50,    # Fewer features for small dataset
            save_model=True
        )
        
        if trainer:
            print("\n✅ Quick test completed successfully!")
            print("📁 Generated test files:")
            print("   - quick_test_data.csv")
            print("   - fall_detection_results.png")
            print("   - best_fall_detection_model_*.pkl")
            return True
        else:
            print("❌ Quick test model training failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Quick test error: {e}")
        return False

def check_requirements():
    """Check if all required modules are available"""
    required_modules = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'seaborn', 'tqdm', 'joblib'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required modules:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\nInstall them using:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    return True

def print_system_info():
    """Print system information"""
    import platform
    import sys
    
    print("\n📋 SYSTEM INFORMATION")
    print("-" * 30)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Current directory: {os.getcwd()}")

if __name__ == "__main__":
    # Check requirements
    if not check_requirements():
        print("\n⚠️  Please install missing modules before running the pipeline.")
        exit(1)
    
    # Print system info
    print_system_info()
    
    # Run main pipeline
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()