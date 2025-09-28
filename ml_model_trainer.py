import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
import warnings
warnings.filterwarnings('ignore')

class FallDetectionModelTrainer:
    def __init__(self, data_file):
        """
        Initialize the model trainer
        
        Parameters:
        data_file: Path to processed MobiAct dataset CSV file
        """
        self.data_file = data_file
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the processed dataset"""
        print("📁 Loading processed dataset...")
        
        try:
            self.data = pd.read_csv(self.data_file)
            print(f"✅ Dataset loaded: {self.data.shape}")
            
            # Show basic info
            print(f"\nDataset Info:")
            print(f"  Total samples: {len(self.data)}")
            print(f"  Features: {len([col for col in self.data.columns if col not in ['subject', 'activity', 'activity_type', 'label']])}")
            
            # Class distribution
            print(f"\nClass Distribution:")
            class_counts = self.data['label'].value_counts().sort_index()
            for label, count in class_counts.items():
                class_name = "Fall" if label == 1 else "Non-Fall (ADL)"
                percentage = (count / len(self.data)) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def prepare_data(self, test_size=0.2, random_state=42, 
                    feature_selection=None, n_features=None):
        """
        Prepare data for training
        
        Parameters:
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        feature_selection: Feature selection method ('k_best', 'rfe', or None)
        n_features: Number of features to select (if feature_selection is not None)
        """
        print("\n🔧 Preparing data for training...")
        
        if self.data is None:
            print("❌ Data not loaded! Run load_data() first.")
            return False
        
        # Separate features and labels
        metadata_cols = ['subject', 'activity', 'activity_type', 'label']
        self.feature_columns = [col for col in self.data.columns if col not in metadata_cols]
        
        X = self.data[self.feature_columns].copy()
        y = self.data['label'].copy()
        
        print(f"Original features: {len(self.feature_columns)}")
        
        # Feature selection if requested
        if feature_selection and n_features:
            print(f"Applying {feature_selection} feature selection...")
            
            if feature_selection == 'k_best':
                selector = SelectKBest(score_func=f_classif, k=n_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = [self.feature_columns[i] for i in selector.get_support(indices=True)]
                
            elif feature_selection == 'rfe':
                # Use RandomForest for RFE
                estimator = RandomForestClassifier(n_estimators=50, random_state=random_state)
                selector = RFE(estimator, n_features_to_select=n_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = [self.feature_columns[i] for i in selector.get_support(indices=True)]
            
            X = pd.DataFrame(X_selected, columns=selected_features)
            self.feature_columns = selected_features
            print(f"Selected features: {len(self.feature_columns)}")
        
        # Split data - stratified to maintain class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Data split completed:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        
        # Show class distribution in splits
        print(f"\nTraining set distribution:")
        train_counts = self.y_train.value_counts().sort_index()
        for label, count in train_counts.items():
            class_name = "Fall" if label == 1 else "Non-Fall"
            percentage = (count / len(self.y_train)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\nTest set distribution:")
        test_counts = self.y_test.value_counts().sort_index()
        for label, count in test_counts.items():
            class_name = "Fall" if label == 1 else "Non-Fall"
            percentage = (count / len(self.y_test)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return True
    
    def initialize_models(self):
        """Initialize different ML models"""
        print("\n🤖 Initializing machine learning models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,  # For ROC curve
                random_state=42
            ),
            
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=1000,
                random_state=42
            ),
            
            'Logistic Regression': LogisticRegression(
                C=1.0,
                solver='liblinear',
                random_state=42
            ),
            
            'Naive Bayes': GaussianNB()
        }
        
        print(f"✅ Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_models(self, cv_folds=5):
        """Train all models with cross-validation"""
        print(f"\n🎯 Training models with {cv_folds}-fold cross-validation...")
        
        if self.X_train is None:
            print("❌ Data not prepared! Run prepare_data() first.")
            return False
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n📚 Training {name}...")
            
            try:
                # Cross-validation
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=cv, scoring='accuracy', n_jobs=-1)
                
                # Train on full training set
                model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                y_test_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                results = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_accuracy': accuracy_score(self.y_train, y_train_pred),
                    'test_accuracy': accuracy_score(self.y_test, y_test_pred),
                    'precision': precision_score(self.y_test, y_test_pred),
                    'recall': recall_score(self.y_test, y_test_pred),
                    'f1_score': f1_score(self.y_test, y_test_pred),
                    'y_test_pred': y_test_pred,
                    'y_test_proba': y_test_proba
                }
                
                if y_test_proba is not None:
                    results['auc_score'] = roc_auc_score(self.y_test, y_test_proba)
                
                self.results[name] = results
                
                print(f"✅ {name} completed:")
                print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
                print(f"   Test Accuracy: {results['test_accuracy']:.3f}")
                print(f"   F1-Score: {results['f1_score']:.3f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        print(f"\n✅ Model training completed! {len(self.results)} models trained successfully.")
        return True
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        if not self.results:
            print("❌ No trained models! Run train_models() first.")
            return
        
        # Create results summary
        summary_data = []
        
        for name, results in self.results.items():
            row = {
                'Model': name,
                'CV_Accuracy': f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}",
                'Test_Accuracy': f"{results['test_accuracy']:.3f}",
                'Precision': f"{results['precision']:.3f}",
                'Recall': f"{results['recall']:.3f}",
                'F1_Score': f"{results['f1_score']:.3f}",
            }
            
            if 'auc_score' in results:
                row['AUC'] = f"{results['auc_score']:.3f}"
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print("\nModel Performance Summary:")
        print(summary_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['f1_score'])
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   F1-Score: {self.results[best_model_name]['f1_score']:.3f}")
        print(f"   Test Accuracy: {self.results[best_model_name]['test_accuracy']:.3f}")
        
        return best_model_name
    
    def plot_results(self, figsize=(15, 12)):
        """Create comprehensive visualization of results"""
        print("\n📈 Creating visualizations...")
        
        if not self.results:
            print("❌ No results to plot!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Fall Detection Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Comparison
        models = list(self.results.keys())
        accuracies = [self.results[model]['test_accuracy'] for model in models]
        
        axes[0, 0].bar(range(len(models)), accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. F1-Score Comparison
        f1_scores = [self.results[model]['f1_score'] for model in models]
        
        axes[0, 1].bar(range(len(models)), f1_scores, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Precision vs Recall
        precisions = [self.results[model]['precision'] for model in models]
        recalls = [self.results[model]['recall'] for model in models]
        
        scatter = axes[0, 2].scatter(precisions, recalls, c=f1_scores, 
                                   s=100, cmap='viridis', alpha=0.7)
        axes[0, 2].set_title('Precision vs Recall')
        axes[0, 2].set_xlabel('Precision')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add model names as labels
        for i, model in enumerate(models):
            axes[0, 2].annotate(model, (precisions[i], recalls[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=axes[0, 2], label='F1-Score')
        
        # 4. ROC Curves (if available)
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (name, results) in enumerate(self.results.items()):
            if 'auc_score' in results and results['y_test_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results['y_test_proba'])
                axes[1, 0].plot(fpr, tpr, color=colors[i % len(colors)], 
                              label=f'{name} (AUC={results["auc_score"]:.3f})')
        
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Best Model Confusion Matrix
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['f1_score'])
        
        best_pred = self.results[best_model_name]['y_test_pred']
        cm = confusion_matrix(self.y_test, best_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_xticklabels(['Non-Fall', 'Fall'])
        axes[1, 1].set_yticklabels(['Non-Fall', 'Fall'])
        
        # 6. Cross-validation scores
        cv_means = [self.results[model]['cv_mean'] for model in models]
        cv_stds = [self.results[model]['cv_std'] for model in models]
        
        axes[1, 2].bar(range(len(models)), cv_means, yerr=cv_stds, 
                      capsize=5, color='orange', alpha=0.7)
        axes[1, 2].set_title('Cross-Validation Accuracy')
        axes[1, 2].set_ylabel('CV Accuracy')
        axes[1, 2].set_xticks(range(len(models)))
        axes[1, 2].set_xticklabels(models, rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        plt.savefig('fall_detection_results.png', dpi=300, bbox_inches='tight')
        print("✅ Results visualization saved as 'fall_detection_results.png'")
    
    def detailed_classification_report(self, model_name=None):
        """Show detailed classification report"""
        if not self.results:
            print("❌ No results available!")
            return
        
        if model_name is None:
            # Use best model
            model_name = max(self.results.keys(), 
                           key=lambda x: self.results[x]['f1_score'])
        
        if model_name not in self.results:
            print(f"❌ Model {model_name} not found!")
            return
        
        print(f"\n📋 DETAILED CLASSIFICATION REPORT - {model_name}")
        print("=" * 60)
        
        y_pred = self.results[model_name]['y_test_pred']
        
        # Classification report
        report = classification_report(self.y_test, y_pred, 
                                     target_names=['Non-Fall (ADL)', 'Fall'],
                                     output_dict=True)
        
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Non-Fall (ADL)', 'Fall']))
        
        # Confusion matrix details
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nConfusion Matrix Details:")
        print(f"  True Negatives (Correct Non-Fall):  {tn}")
        print(f"  False Positives (False Fall Alert): {fp}")
        print(f"  False Negatives (Missed Fall):      {fn}")
        print(f"  True Positives (Correct Fall):      {tp}")
        
        # Additional metrics
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)  # Same as recall
        
        print(f"\nAdditional Metrics:")
        print(f"  Sensitivity (Fall Detection Rate): {sensitivity:.3f}")
        print(f"  Specificity (Non-Fall Accuracy):   {specificity:.3f}")
        print(f"  False Positive Rate:                {fp/(fp+tn):.3f}")
        print(f"  False Negative Rate:                {fn/(fn+tp):.3f}")
    
    def save_best_model(self, filename=None):
        """Save the best performing model"""
        if not self.results:
            print("❌ No trained models to save!")
            return False
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['f1_score'])
        
        best_model = self.results[best_model_name]['model']
        
        if filename is None:
            filename = f'best_fall_detection_model_{best_model_name.replace(" ", "_").lower()}.pkl'
        
        try:
            # Save model and metadata
            model_data = {
                'model': best_model,
                'model_name': best_model_name,
                'feature_columns': self.feature_columns,
                'performance': {
                    'accuracy': self.results[best_model_name]['test_accuracy'],
                    'precision': self.results[best_model_name]['precision'],
                    'recall': self.results[best_model_name]['recall'],
                    'f1_score': self.results[best_model_name]['f1_score']
                }
            }
            
            joblib.dump(model_data, filename)
            print(f"✅ Best model saved: {filename}")
            print(f"   Model: {best_model_name}")
            print(f"   F1-Score: {self.results[best_model_name]['f1_score']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def feature_importance_analysis(self, top_n=20):
        """Analyze feature importance for tree-based models"""
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS (Top {top_n})")
        print("=" * 60)
        
        tree_models = ['Random Forest', 'Gradient Boosting']
        
        for model_name in tree_models:
            if model_name in self.results:
                model = self.results[model_name]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\n{model_name} - Top {top_n} Important Features:")
                    print("-" * 50)
                    
                    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
                        print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
                    
                    # Plot feature importance
                    plt.figure(figsize=(10, 8))
                    top_features = importance_df.head(top_n)
                    
                    plt.barh(range(len(top_features)), top_features['importance'])
                    plt.yticks(range(len(top_features)), top_features['feature'])
                    plt.xlabel('Importance')
                    plt.title(f'Top {top_n} Feature Importance - {model_name}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    
                    # Save plot
                    plt.savefig(f'feature_importance_{model_name.replace(" ", "_").lower()}.png', 
                              dpi=300, bbox_inches='tight')
                    plt.show()

def train_fall_detection_models(data_file='complete_mobiact_dataset.csv',
                               test_size=0.2,
                               feature_selection=None,
                               n_features=None,
                               save_model=True):
    """
    Complete pipeline for training fall detection models
    
    Parameters:
    data_file: Path to processed dataset
    test_size: Test set proportion  
    feature_selection: 'k_best', 'rfe', or None
    n_features: Number of features to select
    save_model: Whether to save the best model
    """
    
    print("🚀 FALL DETECTION MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize trainer
    trainer = FallDetectionModelTrainer(data_file)
    
    # Load data
    if not trainer.load_data():
        return None
    
    # Prepare data
    if not trainer.prepare_data(test_size=test_size, 
                               feature_selection=feature_selection,
                               n_features=n_features):
        return None
    
    # Initialize models
    trainer.initialize_models()
    
    # Train models
    if not trainer.train_models():
        return None
    
    # Evaluate models
    best_model_name = trainer.evaluate_models()
    
    # Create visualizations
    trainer.plot_results()
    
    # Detailed report for best model
    trainer.detailed_classification_report(best_model_name)
    
    # Feature importance analysis
    trainer.feature_importance_analysis()
    
    # Save best model
    if save_model:
        trainer.save_best_model()
    
    print(f"\n🎉 MODEL TRAINING COMPLETED!")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📊 Results visualized and saved")
    
    return trainer

if __name__ == "__main__":
    # Train models on complete dataset
    trainer = train_fall_detection_models(
        data_file='complete_mobiact_dataset.csv',
        test_size=0.2,
        feature_selection='k_best',  # Try feature selection
        n_features=100,              # Select top 100 features
        save_model=True
    )
    
    if trainer:
        print("\n✅ Training completed successfully!")
        print("📁 Check the generated files:")
        print("   - fall_detection_results.png")
        print("   - feature_importance_*.png") 
        print("   - best_fall_detection_model_*.pkl")
    else:
        print("❌ Training failed!")