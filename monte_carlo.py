# import pandas as pd
# import logging
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import argparse
# import random
# import torch
# from sklearn.model_selection import train_test_split
# from optimized_tabddpm import OptimizedTabDDPM  # Updated import to match your Quantile TabDDPM file
# from utils import evaluate_tstr

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# import os

# # Ensure log file is created with proper permissions
# log_file = 'monte_carlo.log'
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_file, mode='w'),  # 'w' mode ensures fresh log each run
#         logging.StreamHandler()
#     ]
# )

# # Log the working directory for debugging
# logging.info(f"Working directory: {os.getcwd()}")
# logging.info(f"Log file path: {os.path.abspath(log_file)}")

# def load_and_split_data(seed):
#     logging.info(f"Loading dataset with seed {seed}...")
#     df = pd.read_csv('creditcard.csv')
#     logging.info(f"Total dataset size: {len(df)} samples")
#     logging.info(f"Original fraud ratio: {df['Class'].mean():.6f}")

#     df = df.drop_duplicates()
#     logging.info(f"Removed {284807 - len(df)} duplicates")
#     logging.info(f"Dataset size after duplicates: {len(df)} samples")

#     train_val, test_data = train_test_split(
#         df, test_size=0.10, stratify=df['Class'], random_state=seed
#     )
#     train_data, val_data = train_test_split(
#         train_val, test_size=0.10/0.90, stratify=train_val['Class'], random_state=seed
#     )

#     test_data.to_csv(f'test_data_sim_{seed}.csv', index=False)
#     logging.info(f"Saved test data ({len(test_data)} samples) to 'test_data_sim_{seed}.csv'")
#     logging.info(f"Train size: {len(train_data)}, fraud ratio: {train_data['Class'].mean():.6f}")
#     logging.info(f"Validation size: {len(val_data)}, fraud ratio: {val_data['Class'].mean():.6f}")
#     logging.info(f"Test size: {len(test_data)}, fraud ratio: {test_data['Class'].mean():.6f}")

#     return train_data, val_data, test_data

# def plot_distribution_comparison(original_data, augmented_data, features, model_name, sim_id):
#     logging.info(f"Generating distribution plots for {model_name} (sim {sim_id})...")
#     fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 16))
#     axes = axes.flatten()

#     for idx, feature in enumerate(features):
#         if feature not in original_data.columns:
#             logging.warning(f"Feature {feature} not found. Skipping...")
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
#     plot_filename = f'distribution_comparison_Quantile {model_name}_sim_{sim_id}.png'
#     plt.savefig(plot_filename)
#     plt.close()
#     logging.info(f"Saved distribution plot to '{plot_filename}'")

# def plot_performance_metrics(metrics_df, model_name, sim_id):
#     logging.info(f"Generating performance plot for Quantile {model_name} (sim {sim_id})...")
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
#     plt.title(f'Classifier Performance for Quantile {model_name} (Sim {sim_id})')
#     plt.xlabel('Metrics')
#     plt.ylabel('Score')
#     plt.ylim(0, 1)
#     plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plot_filename = f'performance_metrics_Quantile{model_name}_sim_{sim_id}.png'
#     plt.savefig(plot_filename, bbox_inches='tight')
#     plt.close()
#     logging.info(f"Saved performance plot to '{plot_filename}'")

# def plot_monte_carlo_summary(all_metrics, model_name):
#     logging.info(f"Generating Monte Carlo summary plots for {model_name}...")
#     classifier_colors = {
#         'AdaBoost': '#ff9999',
#         'RandomForest': '#99ccff',
#         'CatBoost': '#ccff99',
#         'XGBoost': '#ffcc99',
#         'EasyEnsemble': '#cc99ff'
#     }
#     plt.figure(figsize=(10, 6))
#     for classifier in all_metrics['Classifier'].unique():
#         classifier_data = all_metrics[all_metrics['Classifier'] == classifier]['F1-score']
#         sns.kdeplot(data=classifier_data, label=classifier, color=classifier_colors[classifier], fill=True, alpha=0.4)
#     plt.title(f'Monte Carlo: F1-score Density for {model_name}')
#     plt.xlabel('F1-score')
#     plt.ylabel('Density')
#     plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(f'monte_carlo_f1_density_Quantile{model_name}.png', bbox_inches='tight')
#     plt.close()
#     logging.info(f"Saved F1-score density plot to 'monte_carlo_f1_density_Quantile{model_name}.png'")

#     plt.figure(figsize=(10, 6))
#     for classifier in all_metrics['Classifier'].unique():
#         classifier_data = all_metrics[all_metrics['Classifier'] == classifier]['MCC']
#         sns.kdeplot(data=classifier_data, label=classifier, color=classifier_colors[classifier], fill=True, alpha=0.4)
#     plt.title(f'Monte Carlo: MCC Density for {model_name}')
#     plt.xlabel('MCC')
#     plt.ylabel('Density')
#     plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(f'monte_carlo_mcc_density_Quantile model_name.png', bbox_inches='tight')
#     plt.close()
#     logging.info(f"Saved MCC density plot to 'monte_carlo_mcc_density_Quantile{model_name}.png'")

# def main():
#     parser = argparse.ArgumentParser(description='Monte Carlo simulations for fraud detection with Quantile TabDDPM.')
#     parser.add_argument('--model', type=str, default='TabDDPM', choices=['TabDDPM'],
#                         help='Generative model: TabDDPM only')
#     parser.add_argument('--n_simulations', type=int, default=20,
#                         help='Number of Monte Carlo simulations')
#     args = parser.parse_args()
#     model_name = args.model
#     n_simulations = args.n_simulations
#     logging.info(f"Model: {model_name}, Simulations: {n_simulations}")

#     if model_name != 'TabDDPM':
#         logging.error("Only TabDDPM is supported.")
#         raise ValueError("Unsupported model. Use --model TabDDPM.")

#     all_metrics = []
#     base_seed = 42

#     for sim_id in range(n_simulations):
#         sim_seed = base_seed + sim_id
#         logging.info(f"Simulation {sim_id + 1}/{n_simulations} with seed {sim_seed}")
#         set_seed(sim_seed)

#         train_data, val_data, test_data = load_and_split_data(sim_seed)

#         generative_model = OptimizedTabDDPM(input_dim=30, device='cuda', seed=sim_seed)
#         logging.info(f"Optimizing hyperparameters (sim {sim_id + 1})...")
#         start_time = time.time()
#         # best_params = generative_model.optimize_hyperparameters(train_data, val_data, n_trials=5, epochs=100)
#         best_params = generative_model.optimize_hyperparameters(train_data, val_data, n_trials=2, epochs=30)

#         optuna_time = time.time() - start_time
#         logging.info(f"Sim {sim_id + 1}: Optuna took {optuna_time:.2f} s, Best params: {best_params}")

#         # Updated model initialization to include quantile parameter from optimization
#         generative_model = OptimizedTabDDPM(
#             input_dim=30,
#             hidden_dim=best_params['hidden_dim'],
#             num_layers=best_params['num_layers'],
#             num_timesteps=best_params['num_timesteps'],
#             lr=best_params['lr'],
#             batch_size=best_params['batch_size'],
#             quantile=best_params['quantile'],  # Added quantile parameter
#             device='cuda',
#             seed=sim_seed
#         )

#         logging.info(f"Training {model_name} for 1000 epochs with quantile={best_params['quantile']} (sim {sim_id + 1})...")
#         start_time = time.time()
#         # generative_model.train(train_data, val_data=val_data, epochs=1000, batch_size=best_params['batch_size'])
#         generative_model.train(train_data, val_data=val_data, epochs=100, batch_size=best_params['batch_size'])

#         training_time = time.time() - start_time
#         logging.info(f"Sim {sim_id + 1}: Training took {training_time:.2f} s")

#         non_fraud_ratio = 0.90
#         fraud_ratio = 0.10
#         non_fraud_count = len(train_data[train_data['Class'] == 0])
#         fraud_count = len(train_data[train_data['Class'] == 1])
#         total_samples = int(non_fraud_count / non_fraud_ratio)
#         total_fraud_needed = int(total_samples * fraud_ratio)
#         synthetic_fraud_needed = total_fraud_needed - fraud_count
#         logging.info(f"Sim {sim_id + 1}: Non-fraud = {non_fraud_count}, Fraud = {fraud_count}, Synthetic fraud = {synthetic_fraud_needed}")

#         if synthetic_fraud_needed < 0:
#             logging.warning(f"Sim {sim_id + 1}: Synthetic fraud negative ({synthetic_fraud_needed}). Set to 0.")
#             synthetic_fraud_needed = 0

#         logging.info(f"Generating {synthetic_fraud_needed} synthetic fraud samples with Quantile Loss (q={best_params['quantile']})...")
#         gen_start_time = time.time()
#         synthetic_fraud_data = generative_model.generate_synthetic_data(
#             num_legitimate=0, num_fraud=synthetic_fraud_needed, batch_size=64
#         )
#         gen_time = time.time() - gen_start_time
#         logging.info(f"Sim {sim_id + 1}: Generation took {gen_time:.2f} s")

#         synthetic_fraud_data.to_csv(f'synthetic_fraud_data_{model_name}_sim_{sim_id + 1}.csv', index=False)
#         logging.info(f"Sim {sim_id + 1}: Saved synthetic data ({len(synthetic_fraud_data)} samples)")

#         augmented_train_data = pd.concat([train_data, synthetic_fraud_data], ignore_index=True)
#         augmented_train_data.to_csv(f'augmented_train_data_{model_name}_sim_{sim_id + 1}.csv', index=False)
#         logging.info(f"Sim {sim_id + 1}: Saved augmented data ({len(augmented_train_data)} samples, fraud ratio: {augmented_train_data['Class'].mean():.6f})")

#         features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
#         plot_distribution_comparison(train_data, augmented_train_data, features, model_name, sim_id + 1)

#         logging.info(f"Evaluating classifiers (sim {sim_id + 1})...")
#         eval_start_time = time.time()
#         metrics_df = evaluate_tstr(augmented_train_data, test_data)
#         eval_time = time.time() - eval_start_time
#         metrics_df['Simulation'] = sim_id + 1
#         metrics_df.to_csv(f'metrics_{model_name}_sim_{sim_id + 1}.csv', index=False)
#         all_metrics.append(metrics_df)
#         logging.info(f"Sim {sim_id + 1}: Evaluation took {eval_time:.2f} s")

#         plot_performance_metrics(metrics_df, model_name, sim_id + 1)

#     all_metrics_df = pd.concat(all_metrics, ignore_index=True)
#     summary_metrics = all_metrics_df.groupby('Classifier')[['Precision', 'Recall', 'F1-score', 'ROC-AUC', 'MCC']].agg(['mean', 'std']).reset_index()
#     summary_metrics.to_csv(f'monte_carlo_summary_Quantile{model_name}.csv', index=False)
#     logging.info(f"Saved Monte Carlo summary to 'monte_carlo_summary_Quantile{model_name}.csv'")
#     logging.info(f"Summary metrics:\n{summary_metrics.to_string()}")

#     plot_monte_carlo_summary(all_metrics_df, model_name)

# if __name__ == "__main__":
#     main()


# commenting below on 28 August 2025 

# import pandas as pd
# import logging
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import argparse
# import random
# import torch
# import json
# import os
# from sklearn.model_selection import train_test_split
# from optimized_tabddpm import OptimizedTabDDPM
# from utils import evaluate_tstr

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # Ensure log file is created with proper permissions
# log_file = 'monte_carlo.log'
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_file, mode='w'),  # 'w' mode ensures fresh log each run
#         logging.StreamHandler()
#     ]
# )

# # Log the working directory for debugging
# logging.info(f"Working directory: {os.getcwd()}")
# logging.info(f"Log file path: {os.path.abspath(log_file)}")

# # def load_best_hyperparameters():
# #     """Load best hyperparameters from main.py run"""
# #     try:
# #         # Try to load from CSV first (from main.py Optuna output)
# #         if os.path.exists('best_optuna_params.csv'):
# #             params_df = pd.read_csv('best_optuna_params.csv')
# #             params_dict = dict(zip(params_df['Parameter'], params_df['Value']))
# #             logging.info(f"Loaded hyperparameters from best_optuna_params.csv: {params_dict}")
# #             return params_dict
# #         else:
# #             # If CSV doesn't exist, use default optimized parameters
# #             logging.warning("best_optuna_params.csv not found. Using default optimized parameters.")
# #             default_params = {
# #                 'hidden_dim': 512,
# #                 'num_layers': 4,
# #                 'num_timesteps': 1000,
# #                 'lr': 0.001,
# #                 'batch_size': 512,
# #                 'quantile': 0.5,
# #                 'proportion_samples': 1.0
# #             }
# #             logging.info(f"Using default parameters: {default_params}")
# #             return default_params
# #     except Exception as e:
# #         logging.error(f"Error loading hyperparameters: {e}. Using defaults.")
# #         default_params = {
# #             'hidden_dim': 512,
# #             'num_layers': 4,
# #             'num_timesteps': 1000,
# #             'lr': 0.001,
# #             'batch_size': 512,
# #             'quantile': 0.5,
# #             'proportion_samples': 1.0
# #         }
# #         return default_params
    

# # def load_best_hyperparameters():
# #     """Load best hyperparameters from main.py run"""
# #     try:
# #         # Try to load from CSV first (from main.py Optuna output)
# #         if os.path.exists('best_optuna_params.csv'):
# #             params_df = pd.read_csv('best_optuna_params.csv')
# #             params_dict = dict(zip(params_df['Parameter'], params_df['Value']))
# #             # Convert to correct types
# #             params_dict = {
# #                 'hidden_dim': int(float(params_dict['hidden_dim'])),
# #                 'num_layers': int(float(params_dict['num_layers'])),
# #                 'num_timesteps': int(float(params_dict['num_timesteps'])),
# #                 'lr': float(params_dict['lr']),
# #                 'batch_size': int(float(params_dict['batch_size'])),
# #                 'quantile': float(params_dict['quantile']),
# #                 'proportion_samples': float(params_dict.get('proportion_samples', 1.0))
# #             }
# #             logging.info(f"Loaded hyperparameters from best_optuna_params.csv: {params_dict}")
# #             return params_dict
# #         else:
# #             # If CSV doesn't exist, use default optimized parameters
# #             logging.warning("best_optuna_params.csv not found. Using default optimized parameters.")
# #             default_params = {
# #                 'hidden_dim': 512,
# #                 'num_layers': 4,
# #                 'num_timesteps': 1000,
# #                 'lr': 0.001,
# #                 'batch_size': 512,
# #                 'quantile': 0.5,
# #                 'proportion_samples': 1.0
# #             }
# #             logging.info(f"Using default parameters: {default_params}")
# #             return default_params
# #     except Exception as e:
# #         logging.error(f"Error loading hyperparameters: {e}. Using defaults.")
# #         default_params = {
# #             'hidden_dim': 512,
# #             'num_layers': 4,
# #             'num_timesteps': 1000,
# #             'lr': 0.001,
# #             'batch_size': 512,
# #             'quantile': 0.5,
# #             'proportion_samples': 1.0
# #         }
# #         return default_params

# def load_best_hyperparameters():
#     """Load best hyperparameters from main.py run"""
#     try:
#         # Try to load from CSV first (from main.py Optuna output)
#         if os.path.exists('best_optuna_params.csv'):
#             params_df = pd.read_csv('best_optuna_params.csv')
#             params_dict = dict(zip(params_df['Parameter'], params_df['Value']))
#             # Convert to correct types
#             params_dict = {
#                 'hidden_dim': int(float(params_dict['hidden_dim'])),
#                 'num_layers': int(float(params_dict['num_layers'])),
#                 'num_timesteps': int(float(params_dict['num_timesteps'])),
#                 'lr': float(params_dict['lr']),
#                 'batch_size': int(float(params_dict['batch_size'])),
#                 'quantile': float(params_dict['quantile']),
#                 'proportion_samples': float(params_dict.get('proportion_samples', 1.0)),
#                 'schedule_type': params_dict.get('schedule_type', 'cosine'),  # Added schedule_type
#                 'cosine_s': float(params_dict.get('cosine_s', 0.008))  # Added cosine_s
#             }
#             logging.info(f"Loaded hyperparameters from best_optuna_params.csv: {params_dict}")
#             return params_dict
#         else:
#             # If CSV doesn't exist, use default optimized parameters
#             logging.warning("best_optuna_params.csv not found. Using default optimized parameters.")
#             default_params = {
#                 'hidden_dim': 512,
#                 'num_layers': 4,
#                 'num_timesteps': 1000,
#                 'lr': 0.001,
#                 'batch_size': 512,
#                 'quantile': 0.5,
#                 'proportion_samples': 1.0,
#                 'schedule_type': 'cosine',  # Added schedule_type
#                 'cosine_s': 0.008  # Added cosine_s
#             }
#             logging.info(f"Using default parameters: {default_params}")
#             return default_params
#     except Exception as e:
#         logging.error(f"Error loading hyperparameters: {e}. Using defaults.")
#         default_params = {
#             'hidden_dim': 512,
#             'num_layers': 4,
#             'num_timesteps': 1000,
#             'lr': 0.001,
#             'batch_size': 512,
#             'quantile': 0.5,
#             'proportion_samples': 1.0,
#             'schedule_type': 'cosine',  # Added schedule_type
#             'cosine_s': 0.008  # Added cosine_s
#         }
#         return default_params
    

# def load_and_split_data(seed):
#     logging.info(f"Loading dataset with seed {seed}...")
#     df = pd.read_csv('creditcard.csv')
#     logging.info(f"Total dataset size: {len(df)} samples")
#     logging.info(f"Original fraud ratio: {df['Class'].mean():.6f}")

#     df = df.drop_duplicates()
#     logging.info(f"Removed {284807 - len(df)} duplicates")
#     logging.info(f"Dataset size after duplicates: {len(df)} samples")

#     train_val, test_data = train_test_split(
#         df, test_size=0.10, stratify=df['Class'], random_state=seed
#     )
#     train_data, val_data = train_test_split(
#         train_val, test_size=0.10/0.90, stratify=train_val['Class'], random_state=seed
#     )

#     test_data.to_csv(f'test_data_sim_{seed}.csv', index=False)
#     logging.info(f"Saved test data ({len(test_data)} samples) to 'test_data_sim_{seed}.csv'")
#     logging.info(f"Train size: {len(train_data)}, fraud ratio: {train_data['Class'].mean():.6f}")
#     logging.info(f"Validation size: {len(val_data)}, fraud ratio: {val_data['Class'].mean():.6f}")
#     logging.info(f"Test size: {len(test_data)}, fraud ratio: {test_data['Class'].mean():.6f}")

#     return train_data, val_data, test_data

# def plot_distribution_comparison(original_data, augmented_data, features, model_name, sim_id):
#     logging.info(f"Generating distribution plots for {model_name} (sim {sim_id})...")
#     fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 16))
#     axes = axes.flatten()

#     for idx, feature in enumerate(features):
#         if feature not in original_data.columns:
#             logging.warning(f"Feature {feature} not found. Skipping...")
#             continue
#         sns.kdeplot(data=original_data, x=feature, color='blue', ax=axes[idx])
#         sns.kdeplot(data=augmented_data, x=feature, color='orange', ax=axes[idx])
#         axes[idx].set_title(f'{feature}')
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
#     plot_filename = f'distribution_comparison_Quantile{model_name}_sim_{sim_id}.png'
#     plt.savefig(plot_filename)
#     plt.close()
#     logging.info(f"Saved distribution plot to '{plot_filename}'")

# def plot_performance_metrics(metrics_df, model_name, sim_id):
#     logging.info(f"Generating performance plot for Quantile {model_name} (sim {sim_id})...")
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
#     plt.title(f'Classifier Performance for Quantile {model_name} (Sim {sim_id})')
#     plt.xlabel('Metrics')
#     plt.ylabel('Score')
#     plt.ylim(0, 1)
#     plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plot_filename = f'performance_metrics_Quantile_{model_name}_sim_{sim_id}.png'
#     plt.savefig(plot_filename, bbox_inches='tight')
#     plt.close()
#     logging.info(f"Saved performance plot to '{plot_filename}'")

# def plot_monte_carlo_summary(all_metrics, model_name):
#     logging.info(f"Generating Monte Carlo summary plots for {model_name}...")
#     classifier_colors = {
#         'AdaBoost': '#ff9999',
#         'RandomForest': '#99ccff',
#         'CatBoost': '#ccff99',
#         'XGBoost': '#ffcc99',
#         'EasyEnsemble': '#cc99ff'
#     }
#     plt.figure(figsize=(10, 6))
#     for classifier in all_metrics['Classifier'].unique():
#         classifier_data = all_metrics[all_metrics['Classifier'] == classifier]['F1-score']
#         sns.kdeplot(data=classifier_data, label=classifier, color=classifier_colors[classifier], fill=True, alpha=0.4)
#     plt.title(f'Monte Carlo: F1-score Density for Quantile {model_name}')
#     plt.xlabel('F1-score')
#     plt.ylabel('Density')
#     plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(f'monte_carlo_f1_density_Quantile{model_name}.png', bbox_inches='tight')
#     plt.close()
#     logging.info(f"Saved F1-score density plot to 'monte_carlo_f1_density_Quantile_{model_name}.png'")

#     plt.figure(figsize=(10, 6))
#     for classifier in all_metrics['Classifier'].unique():
#         classifier_data = all_metrics[all_metrics['Classifier'] == classifier]['MCC']
#         sns.kdeplot(data=classifier_data, label=classifier, color=classifier_colors[classifier], fill=True, alpha=0.4)
#     plt.title(f'Monte Carlo: MCC Density for Quantile {model_name}')
#     plt.xlabel('MCC')
#     plt.ylabel('Density')
#     plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(f'monte_carlo_mcc_density_Quantile{model_name}.png', bbox_inches='tight')
#     plt.close()
#     logging.info(f"Saved MCC density plot to 'monte_carlo_mcc_density_Quantile_{model_name}.png'")

# def main():
#     parser = argparse.ArgumentParser(description='Monte Carlo simulations for fraud detection with Quantile TabDDPM using fixed hyperparameters.')
#     parser.add_argument('--model', type=str, default='TabDDPM', choices=['TabDDPM'],
#                         help='Generative model: TabDDPM only')
#     parser.add_argument('--n_simulations', type=int, default=100,   # default=50
#                         help='Number of Monte Carlo simulations')
#     args = parser.parse_args()
#     model_name = args.model
#     n_simulations = args.n_simulations
#     logging.info(f"Model: {model_name}, Simulations: {n_simulations}")

#     if model_name != 'TabDDPM':
#         logging.error("Only TabDDPM is supported.")
#         raise ValueError("Unsupported model. Use --model TabDDPM.")

#     # Load fixed hyperparameters from main.py run
#     best_params = load_best_hyperparameters()
#     logging.info(f"Using fixed hyperparameters for all simulations: {best_params}")

#     all_metrics = []
#     base_seed = 42

#     for sim_id in range(n_simulations):
#         sim_seed = base_seed + sim_id
#         logging.info(f"Simulation {sim_id + 1}/{n_simulations} with seed {sim_seed}")
#         set_seed(sim_seed)

#         train_data, val_data, test_data = load_and_split_data(sim_seed)

#         # Create model with fixed hyperparameters (no Optuna optimization)
#         logging.info(f"Creating model with fixed hyperparameters (sim {sim_id + 1})...")
#         # generative_model = OptimizedTabDDPM(
#         #     input_dim=30,
#         #     hidden_dim=best_params['hidden_dim'],
#         #     num_layers=best_params['num_layers'],
#         #     num_timesteps=best_params['num_timesteps'],
#         #     lr=best_params['lr'],
#         #     batch_size=best_params['batch_size'],
#         #     quantile=best_params['quantile'],
#         #     device='cuda',
#         #     seed=sim_seed
#         # )

#         # Create model with fixed hyperparameters (no Optuna optimization)
#         # logging.info(f"Creating model with fixed hyperparameters (sim {sim_id + 1})...")
#         generative_model = OptimizedTabDDPM(
#             input_dim=30,
#             hidden_dim=best_params['hidden_dim'],
#             num_layers=best_params['num_layers'],
#             num_timesteps=best_params['num_timesteps'],
#             lr=best_params['lr'],
#             batch_size=best_params['batch_size'],
#             quantile=best_params['quantile'],
#             schedule_type=best_params.get('schedule_type', 'cosine'),  # Added schedule_type parameter
#             cosine_s=best_params.get('cosine_s', 0.008),  # Added cosine_s parameter
#             device='cuda',
#             seed=sim_seed
#         )

#         logging.info(f"Training {model_name} with fixed params quantile={best_params['quantile']} (sim {sim_id + 1})...")
#         start_time = time.time()
#         generative_model.train(train_data, val_data=val_data, epochs=000, batch_size=best_params['batch_size'])  #epochs=100

#         training_time = time.time() - start_time
#         logging.info(f"Sim {sim_id + 1}: Training took {training_time:.2f} s")

#         non_fraud_ratio = 0.90
#         fraud_ratio = 0.10
#         non_fraud_count = len(train_data[train_data['Class'] == 0])
#         fraud_count = len(train_data[train_data['Class'] == 1])
#         total_samples = int(non_fraud_count / non_fraud_ratio)
#         total_fraud_needed = int(total_samples * fraud_ratio)
#         synthetic_fraud_needed = total_fraud_needed - fraud_count
#         logging.info(f"Sim {sim_id + 1}: Non-fraud = {non_fraud_count}, Fraud = {fraud_count}, Synthetic fraud = {synthetic_fraud_needed}")

#         if synthetic_fraud_needed < 0:
#             logging.warning(f"Sim {sim_id + 1}: Synthetic fraud negative ({synthetic_fraud_needed}). Set to 0.")
#             synthetic_fraud_needed = 0

#         logging.info(f"Generating {synthetic_fraud_needed} synthetic fraud samples with Quantile Loss (q={best_params['quantile']})...")
#         gen_start_time = time.time()
#         synthetic_fraud_data = generative_model.generate_synthetic_data(
#             num_legitimate=0, num_fraud=synthetic_fraud_needed, batch_size=64
#         )
#         gen_time = time.time() - gen_start_time
#         logging.info(f"Sim {sim_id + 1}: Generation took {gen_time:.2f} s")

#         synthetic_fraud_data.to_csv(f'synthetic_fraud_data_{model_name}_sim_{sim_id + 1}.csv', index=False)
#         logging.info(f"Sim {sim_id + 1}: Saved synthetic data ({len(synthetic_fraud_data)} samples)")

#         augmented_train_data = pd.concat([train_data, synthetic_fraud_data], ignore_index=True)
#         augmented_train_data.to_csv(f'augmented_train_data_{model_name}_sim_{sim_id + 1}.csv', index=False)
#         logging.info(f"Sim {sim_id + 1}: Saved augmented data ({len(augmented_train_data)} samples, fraud ratio: {augmented_train_data['Class'].mean():.6f})")

#         features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
#         plot_distribution_comparison(train_data, augmented_train_data, features, model_name, sim_id + 1)

#         logging.info(f"Evaluating classifiers (sim {sim_id + 1})...")
#         eval_start_time = time.time()
#         metrics_df = evaluate_tstr(augmented_train_data, test_data)
#         eval_time = time.time() - eval_start_time
#         metrics_df['Simulation'] = sim_id + 1
#         metrics_df.to_csv(f'metrics_{model_name}_sim_{sim_id + 1}.csv', index=False)
#         all_metrics.append(metrics_df)
#         logging.info(f"Sim {sim_id + 1}: Evaluation took {eval_time:.2f} s")

#         plot_performance_metrics(metrics_df, model_name, sim_id + 1)

#     all_metrics_df = pd.concat(all_metrics, ignore_index=True)
#     summary_metrics = all_metrics_df.groupby('Classifier')[['Precision', 'Recall', 'F1-score', 'ROC-AUC', 'MCC']].agg(['mean', 'std']).reset_index()
#     summary_metrics.to_csv(f'monte_carlo_summary_Quantile{model_name}.csv', index=False)
#     logging.info(f"Saved Monte Carlo summary to 'monte_carlo_summary_Quantile_{model_name}.csv'")
#     logging.info(f"Summary metrics:\n{summary_metrics.to_string()}")

#     plot_monte_carlo_summary(all_metrics_df, model_name)

#     logging.info("Monte Carlo simulation completed successfully!")
#     logging.info(f"Used fixed hyperparameters: {best_params}")

# if __name__ == "__main__":
#     main()

# commented above after removing the validation dataset

import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import random
import torch
import json
import os
from sklearn.model_selection import train_test_split
from optimized_tabddpm import OptimizedTabDDPM
from utils import evaluate_tstr

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Ensure log file is created with proper permissions
log_file = 'monte_carlo.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # 'w' mode ensures fresh log each run
        logging.StreamHandler()
    ]
)

# Log the working directory for debugging
logging.info(f"Working directory: {os.getcwd()}")
logging.info(f"Log file path: {os.path.abspath(log_file)}")

def load_best_hyperparameters():
    """Load best hyperparameters from main.py run"""
    try:
        # Try to load from CSV first (from main.py Optuna output)
        if os.path.exists('best_optuna_params.csv'):
            params_df = pd.read_csv('best_optuna_params.csv')
            params_dict = dict(zip(params_df['Parameter'], params_df['Value']))
            # Convert to correct types
            params_dict = {
                'hidden_dim': int(float(params_dict['hidden_dim'])),
                'num_layers': int(float(params_dict['num_layers'])),
                'num_timesteps': int(float(params_dict['num_timesteps'])),
                'lr': float(params_dict['lr']),
                'batch_size': int(float(params_dict['batch_size'])),
                'quantile': float(params_dict['quantile']),
                'quantile_weight': float(params_dict.get('quantile_weight', 1.0)), 
                'proportion_samples': float(params_dict.get('proportion_samples', 1.0)),
                'schedule_type': params_dict.get('schedule_type', 'cosine'),  # Added schedule_type
                'cosine_s': float(params_dict.get('cosine_s', 0.008))  # Added cosine_s
            }
            logging.info(f"Loaded hyperparameters from best_optuna_params.csv: {params_dict}")
            return params_dict
        else:
            # If CSV doesn't exist, use default optimized parameters
            logging.warning("best_optuna_params.csv not found. Using default optimized parameters.")
            default_params = {
                'hidden_dim': 512,
                'num_layers': 4,
                'num_timesteps': 1000,
                'lr': 0.001,
                'batch_size': 512,
                'quantile': 0.5,
                'quantile_weight': 1.0,
                'proportion_samples': 1.0,
                'schedule_type': 'cosine',  # Added schedule_type
                'cosine_s': 0.008  # Added cosine_s
            }
            logging.info(f"Using default parameters: {default_params}")
            return default_params
    except Exception as e:
        logging.error(f"Error loading hyperparameters: {e}. Using defaults.")
        default_params = {
            'hidden_dim': 512,
            'num_layers': 4,
            'num_timesteps': 1000,
            'lr': 0.001,
            'batch_size': 512,
            'quantile': 0.5,
            'quantile_weight': 1.0,
            'proportion_samples': 1.0,
            'schedule_type': 'cosine',  # Added schedule_type
            'cosine_s': 0.008  # Added cosine_s
        }
        return default_params
    

def load_and_split_data(seed):
    logging.info(f"Loading dataset with seed {seed}...")
    df = pd.read_csv('creditcard.csv')
    logging.info(f"Total dataset size: {len(df)} samples")
    logging.info(f"Original fraud ratio: {df['Class'].mean():.6f}")

    df = df.drop_duplicates()
    logging.info(f"Removed {284807 - len(df)} duplicates")
    logging.info(f"Dataset size after duplicates: {len(df)} samples")

    train_data, test_data = train_test_split(
        df, test_size=0.10, stratify=df['Class'], random_state=seed
    )

    test_data.to_csv(f'test_data_sim_{seed}.csv', index=False)
    logging.info(f"Saved test data ({len(test_data)} samples) to 'test_data_sim_{seed}.csv'")
    logging.info(f"Train size: {len(train_data)}, fraud ratio: {train_data['Class'].mean():.6f}")
    logging.info(f"Test size: {len(test_data)}, fraud ratio: {test_data['Class'].mean():.6f}")

    return train_data, test_data

def plot_distribution_comparison(original_data, augmented_data, features, model_name, sim_id):
    logging.info(f"Generating distribution plots for {model_name} (sim {sim_id})...")
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 16))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        if feature not in original_data.columns:
            logging.warning(f"Feature {feature} not found. Skipping...")
            continue
        sns.kdeplot(data=original_data, x=feature, color='blue', ax=axes[idx])
        sns.kdeplot(data=augmented_data, x=feature, color='orange', ax=axes[idx])
        axes[idx].set_title(f'{feature}')
        legend = axes[idx].get_legend()
        if legend is not None:
            legend.set_visible(False)

    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Original'),
        Line2D([0], [0], color='orange', lw=2, label='Augmented')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plot_filename = f'distribution_comparison_Quantile{model_name}_sim_{sim_id}.png'
    plt.savefig(plot_filename)
    plt.close()
    logging.info(f"Saved distribution plot to '{plot_filename}'")

def plot_performance_metrics(metrics_df, model_name, sim_id):
    logging.info(f"Generating performance plot for Quantile {model_name} (sim {sim_id})...")
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
    plt.title(f'Classifier Performance for Quantile {model_name} (Sim {sim_id})')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_filename = f'performance_metrics_Quantile_{model_name}_sim_{sim_id}.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved performance plot to '{plot_filename}'")

def plot_monte_carlo_summary(all_metrics, model_name):
    logging.info(f"Generating Monte Carlo summary plots for {model_name}...")
    # classifier_colors = {
    #     'AdaBoost': '#ff9999',
    #     'RandomForest': '#99ccff',
    #     'CatBoost': '#ccff99',
    #     'XGBoost': '#ffcc99',
    #     'EasyEnsemble': '#cc99ff'
    # }
    classifier_colors = {
    'AdaBoost': '#e74c3c',      # Red
    'RandomForest': '#3498db',   # Blue
    'CatBoost': '#2ecc71',       # Green
    'XGBoost': '#f39c12',        # Orange
    'EasyEnsemble': '#9b59b6'    # Purple
}
    plt.figure(figsize=(10, 6))
    for classifier in all_metrics['Classifier'].unique():
        classifier_data = all_metrics[all_metrics['Classifier'] == classifier]['F1-score']
        sns.kdeplot(data=classifier_data, label=classifier, color=classifier_colors[classifier], fill=True, alpha=0.4)
    plt.title(f'Monte Carlo: F1-score Density for Quantile {model_name}')
    plt.xlabel('F1-score')
    plt.ylabel('Density')
    plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'monte_carlo_f1_density_Quantile{model_name}.png', bbox_inches='tight')
    plt.close()
    logging.info(f"Saved F1-score density plot to 'monte_carlo_f1_density_Quantile_{model_name}.png'")

    plt.figure(figsize=(10, 6))
    for classifier in all_metrics['Classifier'].unique():
        classifier_data = all_metrics[all_metrics['Classifier'] == classifier]['MCC']
        sns.kdeplot(data=classifier_data, label=classifier, color=classifier_colors[classifier], fill=True, alpha=0.4)
    plt.title(f'Monte Carlo: MCC Density for Quantile {model_name}')
    plt.xlabel('MCC')
    plt.ylabel('Density')
    plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'monte_carlo_mcc_density_Quantile{model_name}.png', bbox_inches='tight')
    plt.close()
    logging.info(f"Saved MCC density plot to 'monte_carlo_mcc_density_Quantile_{model_name}.png'")

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo simulations for fraud detection with Quantile TabDDPM using fixed hyperparameters.')
    parser.add_argument('--model', type=str, default='TabDDPM', choices=['TabDDPM'],
                        help='Generative model: TabDDPM only')
    parser.add_argument('--n_simulations', type=int, default=100,   # default=50
                        help='Number of Monte Carlo simulations')
    args = parser.parse_args()
    model_name = args.model
    n_simulations = args.n_simulations
    logging.info(f"Model: {model_name}, Simulations: {n_simulations}")

    if model_name != 'TabDDPM':
        logging.error("Only TabDDPM is supported.")
        raise ValueError("Unsupported model. Use --model TabDDPM.")

    # Load fixed hyperparameters from main.py run
    best_params = load_best_hyperparameters()
    logging.info(f"Using fixed hyperparameters for all simulations: {best_params}")

    all_metrics = []
    base_seed = 42

    for sim_id in range(n_simulations):
        sim_seed = base_seed + sim_id
        logging.info(f"Simulation {sim_id + 1}/{n_simulations} with seed {sim_seed}")
        set_seed(sim_seed)

        train_data, test_data = load_and_split_data(sim_seed)

        # Create model with fixed hyperparameters (no Optuna optimization)
        logging.info(f"Creating model with fixed hyperparameters (sim {sim_id + 1})...")
        generative_model = OptimizedTabDDPM(
            input_dim=30,
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_layers'],
            num_timesteps=best_params['num_timesteps'],
            lr=best_params['lr'],
            batch_size=best_params['batch_size'],
            quantile=best_params['quantile'],
            quantile_weight=best_params.get('quantile_weight', 1.0),
            schedule_type=best_params.get('schedule_type', 'cosine'),  # Added schedule_type parameter
            cosine_s=best_params.get('cosine_s', 0.008),  # Added cosine_s parameter
            device='cuda',
            seed=sim_seed
        )

        logging.info(f"Training Quantile {model_name} with fixed params quantile={best_params['quantile']} (sim {sim_id + 1})...")
        start_time = time.time()
        generative_model.train(train_data, val_data=None, epochs=1500, batch_size=best_params['batch_size'])  #epochs=100

        training_time = time.time() - start_time
        logging.info(f"Sim {sim_id + 1}: Training took {training_time:.2f} s")

        non_fraud_ratio = 0.90
        fraud_ratio = 0.10
        non_fraud_count = len(train_data[train_data['Class'] == 0])
        fraud_count = len(train_data[train_data['Class'] == 1])
        total_samples = int(non_fraud_count / non_fraud_ratio)
        total_fraud_needed = int(total_samples * fraud_ratio)
        synthetic_fraud_needed = total_fraud_needed - fraud_count
        logging.info(f"Sim {sim_id + 1}: Non-fraud = {non_fraud_count}, Fraud = {fraud_count}, Synthetic fraud = {synthetic_fraud_needed}")

        if synthetic_fraud_needed < 0:
            logging.warning(f"Sim {sim_id + 1}: Synthetic fraud negative ({synthetic_fraud_needed}). Set to 0.")
            synthetic_fraud_needed = 0

        logging.info(f"Generating {synthetic_fraud_needed} synthetic fraud samples with Quantile Loss (q={best_params['quantile']})...")
        gen_start_time = time.time()
        synthetic_fraud_data = generative_model.generate_synthetic_data(
            num_legitimate=0, num_fraud=synthetic_fraud_needed, batch_size=64
        )
        gen_time = time.time() - gen_start_time
        logging.info(f"Sim {sim_id + 1}: Generation took {gen_time:.2f} s")

        synthetic_fraud_data.to_csv(f'synthetic_fraud_data_{model_name}_sim_{sim_id + 1}.csv', index=False)
        logging.info(f"Sim {sim_id + 1}: Saved synthetic data ({len(synthetic_fraud_data)} samples)")

        augmented_train_data = pd.concat([train_data, synthetic_fraud_data], ignore_index=True)
        augmented_train_data.to_csv(f'augmented_train_data_{model_name}_sim_{sim_id + 1}.csv', index=False)
        logging.info(f"Sim {sim_id + 1}: Saved augmented data ({len(augmented_train_data)} samples, fraud ratio: {augmented_train_data['Class'].mean():.6f})")

        features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        plot_distribution_comparison(train_data, augmented_train_data, features, model_name, sim_id + 1)

        logging.info(f"Evaluating classifiers (sim {sim_id + 1})...")
        eval_start_time = time.time()
        metrics_df = evaluate_tstr(augmented_train_data, test_data)
        eval_time = time.time() - eval_start_time
        metrics_df['Simulation'] = sim_id + 1
        metrics_df.to_csv(f'metrics_{model_name}_sim_{sim_id + 1}.csv', index=False)
        all_metrics.append(metrics_df)
        logging.info(f"Sim {sim_id + 1}: Evaluation took {eval_time:.2f} s")

        plot_performance_metrics(metrics_df, model_name, sim_id + 1)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_metrics = all_metrics_df.groupby('Classifier')[['Precision', 'Recall', 'F1-score', 'ROC-AUC', 'MCC']].agg(['mean', 'std']).reset_index()
    summary_metrics.to_csv(f'monte_carlo_summary_Quantile{model_name}.csv', index=False)
    logging.info(f"Saved Monte Carlo summary to 'monte_carlo_summary_Quantile_{model_name}.csv'")
    logging.info(f"Summary metrics:\n{summary_metrics.to_string()}")

    plot_monte_carlo_summary(all_metrics_df, model_name)

    logging.info("Monte Carlo simulation completed successfully!")
    logging.info(f"Used fixed hyperparameters: {best_params}")

if __name__ == "__main__":
    main()