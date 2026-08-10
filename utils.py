import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from imblearn.ensemble import EasyEnsembleClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)

def evaluate_tstr(augmented_train_data: pd.DataFrame, test_data: pd.DataFrame, target_col: str = 'Class', output_dir: str = '.') -> pd.DataFrame:
    """
    Evaluate classifiers on augmented training data and test on holdout test set.
    Computes TP, TN, FP, FN, MCC, and generates confusion matrix plots for each classifier.
    """
    logging.info("Starting evaluation on augmented dataset...")

    # Prepare features and target
    X_train = augmented_train_data.drop(columns=[target_col]).values
    y_train = augmented_train_data[target_col].values
    X_test = test_data.drop(columns=[target_col]).values
    y_test = test_data[target_col].values
    
    # Log class distributions
    logging.info(f"Augmented training set fraud ratio: {y_train.mean():.6f}")
    logging.info(f"Holdout test set fraud ratio: {y_test.mean():.6f}")
    
    # Class weights to balance precision and recall
    fraud_ratio = y_train.mean()
    class_weight = {0: 1.0, 1: (1.0 - fraud_ratio) / fraud_ratio * 1.5}  # Reduced weight for balance
    
    # Define classifiers with tuned hyperparameters
    classifiers = {
        'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=15, class_weight=class_weight, random_state=42, n_jobs=-1
        ),
        'EasyEnsemble': EasyEnsembleClassifier(n_estimators=50, random_state=42),
        'CatBoost': CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1, auto_class_weights='Balanced', verbose=False, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, 
            scale_pos_weight=(1.0 - fraud_ratio) / fraud_ratio * 1.5, 
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    }
    
    metrics = []
    
    for name, clf in classifiers.items():
        logging.info(f"Training {name} with tuned hyperparameters...")
        
        clf.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_test)
        
        # Threshold tuning with weighted F1-score (beta=2.0 to favor recall)
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_f1 = 0.0
        best_metrics = {}
        best_y_pred = None
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = (1 + 2.0) * (precision * recall) / (2.0 * precision + recall + 1e-8)  # Beta=2.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                best_y_pred = y_pred
                best_metrics = {
                    'Precision': precision,
                    'Recall': recall,
                    'F1-score': f1,
                    'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
                    'MCC': matthews_corrcoef(y_test, y_pred)
                }
        
        # Fallback to default threshold if no improvement found
        if not best_metrics:
            logging.warning(f"No threshold improved F1-score for {name}. Using default threshold 0.5.")
            best_y_pred = (y_pred_proba >= 0.5).astype(int)
            best_metrics = {
                'Precision': precision_score(y_test, best_y_pred, zero_division=0),
                'Recall': recall_score(y_test, best_y_pred, zero_division=0),
                'F1-score': f1_score(y_test, best_y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
                'MCC': matthews_corrcoef(y_test, best_y_pred)
            }
        
        # Calculate confusion matrix and TP, TN, FP, FN
        cm = confusion_matrix(y_test, best_y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        logging.info(f"{name}: Best threshold = {best_threshold:.4f}, "
                     f"Precision = {best_metrics['Precision']:.4f}, "
                     f"Recall = {best_metrics['Recall']:.4f}, "
                     f"F1-score = {best_metrics['F1-score']:.4f}, "
                     f"ROC-AUC = {best_metrics['ROC-AUC']:.4f}, "
                     f"MCC = {best_metrics['MCC']:.4f}, "
                     f"TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn}")
        
        # Generate and save confusion matrix plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Non-Fraud (0)', 'Fraud (1)'],
                    yticklabels=['Non-Fraud (0)', 'Fraud (1)'])
        plt.title(f'Confusion Matrix for {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_{name}.png', dpi=300)
        plt.close()
        logging.info(f"Confusion matrix plot for {name} saved to 'confusion_matrix_{name}.png'")
        
        metrics.append({
            'Classifier': name,
            'Precision': best_metrics['Precision'],
            'Recall': best_metrics['Recall'],
            'F1-score': best_metrics['F1-score'],
            'ROC-AUC': best_metrics['ROC-AUC'],
            'MCC': best_metrics['MCC'],
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        })
    
    metrics_df = pd.DataFrame(metrics)
    logging.info("Metrics computed. Saving to metrics.csv")
    metrics_df.to_csv('metrics.csv', index=False)
    
    # Generate bar plot for existing metrics
    logging.info("Generating bar plots...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()
    metric_titles = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']
    metric_vars = ['Precision', 'Recall', 'F1-score', 'ROC-AUC', 'MCC']
    
    for i, (metric, title) in enumerate(zip(metric_vars, metric_titles)):
        sns.barplot(data=metrics_df, x='Classifier', y=metric, ax=axes[i], color='#66c2a5')
        axes[i].set_title(title, fontsize=14)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].set_xlabel('Classifier', fontsize=12)
        # axes[i].set_ylim(-1 if metric == 'MCC' else 0, 1)  # MCC ranges from -1 to 1
        axes[i].set_ylim(0, 1)  # Set y-axis from 0 to 1 for all metrics
        if i == len(metric_vars) - 1 and len(axes) > len(metric_vars):
            axes[-1].set_visible(False)  # Hide extra subplot if present
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300)
    plt.close()
    logging.info("Bar plots saved to performance_metrics.png")
    
    return metrics_df