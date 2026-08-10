# import pandas as pd
# import logging
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.model_selection import train_test_split
# from optimized_tabddpm import OptimizedTabDDPM  # Updated import to match your Quantile TabDDPM file
# from utils import evaluate_tstr
# import argparse
# import random
# import torch

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
# )

# def load_and_split_data():
#     logging.info("Loading dataset...")
#     df = pd.read_csv('creditcard.csv')
#     logging.info(f"Total dataset size: {len(df)} samples")
#     logging.info(f"Original dataset fraud ratio: {df['Class'].mean():.6f}")

#     df = df.drop_duplicates()
#     logging.info(f"Removed {284807 - len(df)} duplicated rows")
#     logging.info(f"Dataset size after removing duplicates: {len(df)} samples")

#     train_val, test_data = train_test_split(
#         df, test_size=0.10, stratify=df['Class'], random_state=42
#     )
#     train_data, val_data = train_test_split(
#         train_val, test_size=0.10/0.90, stratify=train_val['Class'], random_state=42
#     )

#     test_data.to_csv('test_data.csv', index=False)
#     logging.info(f"Test data saved to 'test_data.csv' with {len(test_data)} samples")

#     logging.info(f"Training set size: {len(train_data)} samples")
#     logging.info(f"Training set fraud ratio: {train_data['Class'].mean():.6f}")
#     logging.info(f"Validation set size: {len(val_data)} samples")
#     logging.info(f"Validation set fraud ratio: {val_data['Class'].mean():.6f}")
#     logging.info(f"Test set size: {len(test_data)} samples")
#     logging.info(f"Test set fraud ratio: {test_data['Class'].mean():.6f}")

#     return train_data, val_data, test_data

# def plot_distribution_comparison(original_data, augmented_data, features, model_name):
#     logging.info(f"Generating distribution comparison plots for Quantile {model_name}...")
#     fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 16))
#     axes = axes.flatten()

#     for idx, feature in enumerate(features):
#         if feature not in original_data.columns:
#             logging.warning(f"Feature {feature} not found in the dataset. Skipping...")
#             continue
#         sns.kdeplot(data=original_data, x=feature, color='blue', ax=axes[idx])
#         sns.kdeplot(data=augmented_data, x=feature, color='orange', ax=axes[idx])
#         axes[idx].set_title(f'{feature}')
#         # axes[idx].legend().set_visible(False)
#         legend = axes[idx].get_legend()
#         if legend is not None:
#             legend.set_visible(False)

#     for idx in range(len(features), len(axes)):
#         fig.delaxes(axes[idx])

#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], color='blue', lw=2, label='Original'),
#         Line2D([0], [0], color='orange', lw=2, label='Augmented')
#     ]
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=12)

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.95)
#     plot_filename = f'distribution_comparison_Quantile_{model_name}.png'
#     plt.savefig(plot_filename)
#     plt.close()
#     logging.info(f"Distribution comparison plot saved to '{plot_filename}'")

# def plot_performance_metrics(metrics_df, model_name):
#     logging.info(f"Generating performance metrics plot for Quantile {model_name}...")
#     metrics = ['Precision', 'Recall', 'F1-score', 'ROC-AUC', 'MCC']
#     metrics_df_melted = metrics_df.melt(id_vars=['Classifier'], 
#                                         value_vars=metrics, 
#                                         var_name='Metric', 
#                                         value_name='Score')
#     classifier_colors = {
#         'AdaBoost': '#1f77b4',
#         'RandomForest': '#ff7f0e',
#         'CatBoost': '#2ca02c',
#         'XGBoost': '#d62728',
#         'EasyEnsemble': '#9467bd'
#     }
#     plt.figure(figsize=(12, 5))
#     sns.barplot(data=metrics_df_melted, x='Metric', y='Score', hue='Classifier', palette=classifier_colors)
#     plt.title(f'Classifier Performance for Quantile {model_name}')
#     plt.xlabel('Metrics')
#     plt.ylabel('Score')
#     plt.ylim(0, 1)
#     plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plot_filename = f'performance_metrics_Quantile_{model_name}.png'
#     plt.savefig(plot_filename, bbox_inches='tight')
#     plt.close()
#     logging.info(f"Performance metrics plot saved to '{plot_filename}'")

# def main():
#     parser = argparse.ArgumentParser(description='Run fraud detection with Quantile TabDDPM.')
#     parser.add_argument('--model', type=str, default='TabDDPM', choices=['TabDDPM'],
#                         help='Generative model to use: TabDDPM only')
#     args = parser.parse_args()
#     model_name = args.model
#     logging.info(f"Selected generative model: {model_name}")

#     set_seed(42)
#     train_data, val_data, test_data = load_and_split_data()

#     if model_name != 'TabDDPM':
#         logging.error("Only TabDDPM is supported. CGAN and VAE require generative_models.py.")
#         raise ValueError("Unsupported model. Use --model TabDDPM.")
    
#     generative_model = OptimizedTabDDPM(input_dim=30, device='cuda', seed=42)
#     logging.info("Optimizing hyperparameters with Optuna...")
#     start_time = time.time()
#     best_params = generative_model.optimize_hyperparameters(train_data, val_data, n_trials=50, epochs=100) #Optuna trials and epochs
#     #best_params = generative_model.optimize_hyperparameters(train_data, val_data, n_trials=3, epochs=5)

#     optuna_time = time.time() - start_time
#     logging.info(f"Optuna optimization took {optuna_time:.2f} seconds")
#     logging.info(f"Best hyperparameters: {best_params}")

#     # Updated model initialization to include quantile parameter from optimization
#     # generative_model = OptimizedTabDDPM(
#     #     input_dim=30,
#     #     hidden_dim=best_params['hidden_dim'],
#     #     num_layers=best_params['num_layers'],
#     #     num_timesteps=best_params['num_timesteps'],
#     #     lr=best_params['lr'],
#     #     batch_size=best_params['batch_size'],
#     #     quantile=best_params['quantile'],  # Added quantile parameter
#     #     device='cuda',
#     #     seed=42
#     # )
#     # Updated model initialization to include ALL parameters from optimization
#     generative_model = OptimizedTabDDPM(
#         input_dim=30,
#         hidden_dim=best_params['hidden_dim'],
#         num_layers=best_params['num_layers'],
#         num_timesteps=best_params['num_timesteps'],
#         lr=best_params['lr'],
#         batch_size=best_params['batch_size'],
#         quantile=best_params['quantile'],  # Added quantile parameter
#         schedule_type=best_params.get('schedule_type', 'cosine'),  # Added schedule_type parameter
#         cosine_s=best_params.get('cosine_s', 0.008),  # Added cosine_s parameter
#         device='cuda',
#         seed=42
#     )
# #     seed = 42  # or any integer
# #     generative_model = OptimizedTabDDPM(
# #     input_dim=30,
# #     hidden_dim=int(best_params['hidden_dim']),
# #     num_layers=int(best_params['num_layers']),
# #     num_timesteps=int(best_params['num_timesteps']),
# #     lr=float(best_params['lr']),
# #     batch_size=int(best_params['batch_size']),
# #     quantile=float(best_params['quantile']),
# #     device='cuda',
# #     seed=seed
# # )


#     logging.info(f"Training {model_name} model for 1000 epochs with best hyperparameters (including quantile={best_params['quantile']})...")
#     start_time = time.time()
#     generative_model.train(train_data, val_data=val_data, epochs=2000, batch_size=best_params['batch_size'])
#     #generative_model.train(train_data, val_data=val_data, epochs=10, batch_size=best_params['batch_size'])
#     training_time = time.time() - start_time
#     logging.info(f"Training Quantile {model_name} model took {training_time:.2f} seconds")

#     non_fraud_ratio = 0.90
#     fraud_ratio = 0.10
#     process_start_time = time.time()

#     non_fraud_count = len(train_data[train_data['Class'] == 0])
#     fraud_count = len(train_data[train_data['Class'] == 1])
#     total_samples = int(non_fraud_count / non_fraud_ratio)
#     total_fraud_needed = int(total_samples * fraud_ratio)
#     synthetic_fraud_needed = total_fraud_needed - fraud_count
#     logging.info(f"Target total samples: {total_samples}")
#     logging.info(f"Total fraud samples needed: {total_fraud_needed}")
#     logging.info(f"Synthetic fraud samples to generate: {synthetic_fraud_needed}")

#     if synthetic_fraud_needed < 0:
#         logging.warning(f"Synthetic fraud needed is negative ({synthetic_fraud_needed}). Adjusting to 0.")
#         synthetic_fraud_needed = 0

#     logging.info(f"Generating {synthetic_fraud_needed} synthetic fraud samples using {model_name} with Quantile Loss (q={best_params['quantile']})...")
#     gen_start_time = time.time()
#     synthetic_fraud_data = generative_model.generate_synthetic_data(
#         num_legitimate=0, num_fraud=synthetic_fraud_needed, batch_size=64
#     )
#     gen_time = time.time() - gen_start_time
#     logging.info(f"Generation took {gen_time:.2f} seconds")

#     synthetic_fraud_data.to_csv(f'synthetic_fraud_data_{model_name}.csv', index=False)
#     logging.info(f"Synthetic fraud data saved with {len(synthetic_fraud_data)} samples to 'synthetic_fraud_data_{model_name}.csv'")
#     logging.info(f"Synthetic fraud data fraud ratio: {synthetic_fraud_data['Class'].mean():.6f}")

#     logging.info(f"Augmenting training set with synthetic fraud samples...")
#     augmented_train_data = pd.concat([train_data, synthetic_fraud_data], ignore_index=True)
#     augmented_train_data.to_csv(f'augmented_train_data_{model_name}.csv', index=False)
#     logging.info(f"Augmented training set saved with {len(augmented_train_data)} samples to 'augmented_train_data_{model_name}.csv'")
#     logging.info(f"Augmented training set fraud ratio: {augmented_train_data['Class'].mean():.6f}")

#     features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
#     plot_distribution_comparison(train_data, augmented_train_data, features, model_name)

#     logging.info(f"Evaluating classifiers on augmented dataset...")
#     eval_start_time = time.time()
#     metrics_df = evaluate_tstr(augmented_train_data, test_data)
#     eval_time = time.time() - eval_start_time
#     logging.info(f"Classifier evaluation took {eval_time:.2f} seconds")

#     metrics_df.to_csv(f'metrics_{model_name}.csv', index=False)
#     logging.info(f"Evaluation completed. Metrics saved to 'metrics_Quantile_{model_name}.csv'")

#     plot_performance_metrics(metrics_df, model_name)

#     process_total_time = time.time() - process_start_time
#     logging.info(f"Total time for process (generation + augmentation + evaluation): {process_total_time:.2f} seconds")

# if __name__ == "__main__":
#     main()

#Updated code with improved time axis formatting in distribution plots on 19 September 2025

import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from optimized_tabddpm import OptimizedTabDDPM  # Updated import to match your Quantile TabDDPM file
from utils import evaluate_tstr
import argparse
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)

def load_and_split_data():
    logging.info("Loading dataset...")
    df = pd.read_csv('creditcard.csv')
    logging.info(f"Total dataset size: {len(df)} samples")
    logging.info(f"Original dataset fraud ratio: {df['Class'].mean():.6f}")

    df = df.drop_duplicates()
    logging.info(f"Removed {284807 - len(df)} duplicated rows")
    logging.info(f"Dataset size after removing duplicates: {len(df)} samples")

    train_val, test_data = train_test_split(
        df, test_size=0.10, stratify=df['Class'], random_state=42
    )
    train_data, val_data = train_test_split(
        train_val, test_size=0.10/0.90, stratify=train_val['Class'], random_state=42
    )

    test_data.to_csv('test_data.csv', index=False)
    logging.info(f"Test data saved to 'test_data.csv' with {len(test_data)} samples")

    logging.info(f"Training set size: {len(train_data)} samples")
    logging.info(f"Training set fraud ratio: {train_data['Class'].mean():.6f}")
    logging.info(f"Validation set size: {len(val_data)} samples")
    logging.info(f"Validation set fraud ratio: {val_data['Class'].mean():.6f}")
    logging.info(f"Test set size: {len(test_data)} samples")
    logging.info(f"Test set fraud ratio: {test_data['Class'].mean():.6f}")

    return train_data, val_data, test_data

def format_time_axis(data, feature_name):
    """
    Convert time values to more readable units for plotting
    """
    if feature_name == 'Time':
        # Convert seconds to units (1 unit = 10,000 seconds for better readability)
        # You can adjust this scaling factor as needed
        time_unit = 100000  # 1 unit = 100,000 seconds
        return data / time_unit, f'Time (×{time_unit:,} sec)'
    else:
        return data, feature_name

# def plot_distribution_comparison(original_data, augmented_data, features, model_name):
#     logging.info(f"Generating distribution comparison plots for Quantile {model_name}...")
#     fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 16))
#     axes = axes.flatten()

#     for idx, feature in enumerate(features):
#         if feature not in original_data.columns:
#             logging.warning(f"Feature {feature} not found in the dataset. Skipping...")
#             continue
        
#         # Format data for better axis readability
#         orig_data_formatted, xlabel = format_time_axis(original_data[feature], feature)
#         aug_data_formatted, _ = format_time_axis(augmented_data[feature], feature)
        
#         # Create temporary series with formatted data for plotting
#         orig_series = pd.Series(orig_data_formatted, name=feature)
#         aug_series = pd.Series(aug_data_formatted, name=feature)
        
#         sns.kdeplot(data=orig_series, color='blue', ax=axes[idx], label='Original')
#         sns.kdeplot(data=aug_series, color='orange', ax=axes[idx], label='Augmented')
        
#         # Set the title and xlabel
#         axes[idx].set_title(f'{feature}')
#         axes[idx].set_xlabel(xlabel)
        
#         # Remove individual legends
#         legend = axes[idx].get_legend()
#         if legend is not None:
#             legend.set_visible(False)

#     # Remove unused subplots
#     for idx in range(len(features), len(axes)):
#         fig.delaxes(axes[idx])

#     # Add global legend
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], color='blue', lw=2, label='Original'),
#         Line2D([0], [0], color='orange', lw=2, label='Augmented')
#     ]
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=12)

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.95)
#     plot_filename = f'distribution_comparison_Quantile_{model_name}.png'
#     plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
#     plt.close()
#     logging.info(f"Distribution comparison plot saved to '{plot_filename}'")



def plot_distribution_comparison(original_data, augmented_data, features, model_name):
    logging.info(f"Generating distribution comparison plots for Quantile {model_name}...")
    
    # Calculate the number of features and appropriate grid size
    num_features = len(features)
    logging.info(f"Total features to plot: {num_features}")
    
    # Calculate grid dimensions - aim for roughly square grid
    ncols = 6  # Keep 6 columns for better readability
    nrows = (num_features + ncols - 1) // ncols  # Ceiling division
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows*3.2))
    
    # Handle case where we have only one row
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        if feature not in original_data.columns:
            logging.warning(f"Feature {feature} not found in the dataset. Skipping...")
            continue
        
        # Format data for better axis readability
        orig_data_formatted, xlabel = format_time_axis(original_data[feature], feature)
        aug_data_formatted, _ = format_time_axis(augmented_data[feature], feature)
        
        # Create temporary series with formatted data for plotting
        orig_series = pd.Series(orig_data_formatted, name=feature)
        aug_series = pd.Series(aug_data_formatted, name=feature)
        
        sns.kdeplot(data=orig_series, color='blue', ax=axes[idx], label='Original')
        sns.kdeplot(data=aug_series, color='orange', ax=axes[idx], label='Augmented')
        
        # Set the title and xlabel
        axes[idx].set_title(f'{feature}')
        axes[idx].set_xlabel(xlabel)
        
        # Remove individual legends
        legend = axes[idx].get_legend()
        if legend is not None:
            legend.set_visible(False)

    # Remove unused subplots
    total_subplots = nrows * ncols
    for idx in range(len(features), total_subplots):
        fig.delaxes(axes[idx])

    # Add global legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Original'),
        Line2D([0], [0], color='orange', lw=2, label='Augmented')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plot_filename = f'distribution_comparison_Quantile_{model_name}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Distribution comparison plot saved to '{plot_filename}'")
    logging.info(f"Successfully plotted all {len(features)} features")


def plot_individual_feature_distributions(original_data, augmented_data, model_name):
    """
    Create individual distribution comparison plots for selected features.
    Each feature gets its own plot file saved as PDF.
    """
    logging.info(f"Generating individual distribution plots for selected features...")
    
    # Selected features to plot individually
    selected_features = ['V2', 'V3', 'V5', 'V12', 'V14', 'V16', 'V17', 'V18']
    
    for feature in selected_features:
        if feature not in original_data.columns:
            logging.warning(f"Feature {feature} not found in the dataset. Skipping...")
            continue
        
        # Format data for better axis readability
        orig_data_formatted, xlabel = format_time_axis(original_data[feature], feature)
        aug_data_formatted, _ = format_time_axis(augmented_data[feature], feature)
        
        # Create temporary series with formatted data for plotting
        orig_series = pd.Series(orig_data_formatted, name=feature)
        aug_series = pd.Series(aug_data_formatted, name=feature)
        
        # Create individual plot
        plt.figure(figsize=(8, 6))
        sns.kdeplot(data=orig_series, color='blue', label='Original', linewidth=2)
        sns.kdeplot(data=aug_series, color='orange', label='Augmented', linewidth=2)
        
        plt.title(f'Distribution Comparison: {feature}', fontsize=20, fontweight='bold')
        plt.xlabel(xlabel, fontsize=18, fontweight='bold')
        plt.ylabel('Density', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save as PDF
        plot_filename = f'distribution_{feature}_Quantile_{model_name}.pdf'
        plt.savefig(plot_filename, dpi=600, bbox_inches='tight', format='pdf')
        plt.close()
        
        logging.info(f"Individual plot for {feature} saved to '{plot_filename}'")
    
    logging.info(f"Successfully created individual plots for all selected features")

def plot_performance_metrics(metrics_df, model_name):
    logging.info(f"Generating performance metrics plot for Quantile {model_name}...")
    metrics = ['Precision', 'Recall', 'F1-score', 'ROC-AUC', 'MCC']
    metrics_df_melted = metrics_df.melt(id_vars=['Classifier'], 
                                        value_vars=metrics, 
                                        var_name='Metric', 
                                        value_name='Score')
    classifier_colors = {
        'AdaBoost': '#1f77b4',
        'RandomForest': '#ff7f0e',
        'CatBoost': '#2ca02c',
        'XGBoost': '#d62728',
        'EasyEnsemble': '#9467bd'
    }
    plt.figure(figsize=(12, 5))
    sns.barplot(data=metrics_df_melted, x='Metric', y='Score', hue='Classifier', palette=classifier_colors)
    plt.title(f'Classifier Performance for Quantile {model_name}')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_filename = f'performance_metrics_Quantile_{model_name}.png'
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Performance metrics plot saved to '{plot_filename}'")

def main():
    parser = argparse.ArgumentParser(description='Run fraud detection with Quantile TabDDPM.')
    parser.add_argument('--model', type=str, default='TabDDPM', choices=['TabDDPM'],
                        help='Generative model to use: TabDDPM only')
    args = parser.parse_args()
    model_name = args.model
    logging.info(f"Selected generative model: {model_name}")

    set_seed(42)
    train_data, val_data, test_data = load_and_split_data()

    if model_name != 'TabDDPM':
        logging.error("Only TabDDPM is supported. CGAN and VAE require generative_models.py.")
        raise ValueError("Unsupported model. Use --model TabDDPM.")
    
    generative_model = OptimizedTabDDPM(input_dim=30, device='cuda', seed=42)
    logging.info("Optimizing hyperparameters with Optuna...")
    start_time = time.time()
    best_params = generative_model.optimize_hyperparameters(train_data, val_data, n_trials=50, epochs=100) #Optuna trials and epochs
    #best_params = generative_model.optimize_hyperparameters(train_data, val_data, n_trials=3, epochs=5)

    optuna_time = time.time() - start_time
    logging.info(f"Optuna optimization took {optuna_time:.2f} seconds")
    logging.info(f"Best hyperparameters: {best_params}")

    # Updated model initialization to include ALL parameters from optimization
    generative_model = OptimizedTabDDPM(
        input_dim=30,
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        num_timesteps=best_params['num_timesteps'],
        lr=best_params['lr'],
        batch_size=best_params['batch_size'],
        quantile=best_params['quantile'],  # Added quantile parameter
        schedule_type=best_params.get('schedule_type', 'cosine'),  # Added schedule_type parameter
        cosine_s=best_params.get('cosine_s', 0.008),  # Added cosine_s parameter
        device='cuda',
        seed=42
    )

    logging.info(f"Training {model_name} model for 1000 epochs with best hyperparameters (including quantile={best_params['quantile']})...")
    start_time = time.time()
    generative_model.train(train_data, val_data=val_data, epochs=1500, batch_size=best_params['batch_size'])
    #generative_model.train(train_data, val_data=val_data, epochs=10, batch_size=best_params['batch_size'])
    training_time = time.time() - start_time
    logging.info(f"Training Quantile {model_name} model took {training_time:.2f} seconds")

    non_fraud_ratio = 0.90 #0.90
    fraud_ratio = 0.10 #0.10
    process_start_time = time.time()

    non_fraud_count = len(train_data[train_data['Class'] == 0])
    fraud_count = len(train_data[train_data['Class'] == 1])
    total_samples = int(non_fraud_count / non_fraud_ratio)
    total_fraud_needed = int(total_samples * fraud_ratio)
    synthetic_fraud_needed = total_fraud_needed - fraud_count
    logging.info(f"Target total samples: {total_samples}")
    logging.info(f"Total fraud samples needed: {total_fraud_needed}")
    logging.info(f"Synthetic fraud samples to generate: {synthetic_fraud_needed}")

    if synthetic_fraud_needed < 0:
        logging.warning(f"Synthetic fraud needed is negative ({synthetic_fraud_needed}). Adjusting to 0.")
        synthetic_fraud_needed = 0

    logging.info(f"Generating {synthetic_fraud_needed} synthetic fraud samples using {model_name} with Quantile Loss (q={best_params['quantile']})...")
    gen_start_time = time.time()
    synthetic_fraud_data = generative_model.generate_synthetic_data(
        num_legitimate=0, num_fraud=synthetic_fraud_needed, batch_size=64
    )
    gen_time = time.time() - gen_start_time
    logging.info(f"Generation took {gen_time:.2f} seconds")

    synthetic_fraud_data.to_csv(f'synthetic_fraud_data_{model_name}.csv', index=False)
    logging.info(f"Synthetic fraud data saved with {len(synthetic_fraud_data)} samples to 'synthetic_fraud_data_{model_name}.csv'")
    logging.info(f"Synthetic fraud data fraud ratio: {synthetic_fraud_data['Class'].mean():.6f}")

    logging.info(f"Augmenting training set with synthetic fraud samples...")
    augmented_train_data = pd.concat([train_data, synthetic_fraud_data], ignore_index=True)
    augmented_train_data.to_csv(f'augmented_train_data_{model_name}.csv', index=False)
    logging.info(f"Augmented training set saved with {len(augmented_train_data)} samples to 'augmented_train_data_{model_name}.csv'")
    logging.info(f"Augmented training set fraud ratio: {augmented_train_data['Class'].mean():.6f}")

    features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    plot_distribution_comparison(train_data, augmented_train_data, features, model_name)

    plot_individual_feature_distributions(train_data, augmented_train_data, model_name)  # Add this line



    logging.info(f"Evaluating classifiers on augmented dataset...")
    eval_start_time = time.time()
    metrics_df = evaluate_tstr(augmented_train_data, test_data)
    eval_time = time.time() - eval_start_time
    logging.info(f"Classifier evaluation took {eval_time:.2f} seconds")

    metrics_df.to_csv(f'metrics_{model_name}.csv', index=False)
    logging.info(f"Evaluation completed. Metrics saved to 'metrics_Quantile_{model_name}.csv'")

    plot_performance_metrics(metrics_df, model_name)

    process_total_time = time.time() - process_start_time
    logging.info(f"Total time for process (generation + augmentation + evaluation): {process_total_time:.2f} seconds")

if __name__ == "__main__":
    main()