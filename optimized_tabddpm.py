# import csv
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, WeightedRandomSampler
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import f1_score
# from xgboost import XGBClassifier
# from tqdm import tqdm
# from typing import Optional, Dict, Tuple, List
# import logging
# import os
# from dataset import TabularDataset
# from models import MLPDiffusion
# import optuna

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('training.log'),
#         logging.StreamHandler()
#     ]
# )

# class OptimizedTabDDPM(nn.Module):
#     """
#     Optimized TabDDPM model for conditional synthetic data generation with Quantile Loss.
#     Used for generating high-quality fraud samples by conditioning on the fraud class.
#     """
#     def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 4,
#                  num_timesteps: int = 1000, device: str = 'auto', 
#                  seed: int = 42, lr: float = 1e-4, batch_size: int = 512, quantile: float = 0.5):
#         super(OptimizedTabDDPM, self).__init__()
#         self.device = self._get_device(device)
#         self.input_dim = input_dim
#         self.num_classes = 2
#         self.num_timesteps = num_timesteps
#         self.seed = seed
#         self.lr = lr
#         self.batch_size = batch_size
#         self.quantile = quantile
#         torch.manual_seed(self.seed)
        
#         logging.info(f"Initializing TabDDPM on {self.device} with input_dim={input_dim}, "
#                      f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
#                      f"lr={lr}, batch_size={batch_size}, seed={seed}, quantile={quantile}")
        
#         d_layers = [hidden_dim] * num_layers
#         self.model = MLPDiffusion(
#             d_in=input_dim + self.num_classes,
#             d_layers=d_layers,
#             d_out=input_dim,
#             dropout=0.3
#         ).to(self.device)
        
#         self._setup_noise_schedule()
        
#         self.optimizer = torch.optim.AdamW(
#             self.model.parameters(), 
#             lr=lr,
#             weight_decay=1e-4,
#             betas=(0.9, 0.999),
#             eps=1e-8
#         )
        
#         self.scheduler = None
#         self.numerical_scalers = {}
#         self.categorical_cols = ['Class']
#         self.numerical_cols = None
#         self.best_val_loss = float('inf')
#         self.patience = 200
#         self.patience_counter = 0
#         self.best_model_state = None
#         self.best_epoch = 0
#         self.current_epoch = 0

#     def _get_device(self, device: str) -> str:
#         if device == 'auto':
#             if torch.cuda.is_available():
#                 return 'cuda'
#             elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#                 return 'mps'
#             return 'cpu'
#         return device

#     # def _setup_noise_schedule(self):
#     #     timesteps = torch.arange(0, self.num_timesteps, dtype=torch.float32, device=self.device)
#     #     sigmoid = 1 / (1 + torch.exp(-10 * (timesteps / self.num_timesteps - 0.5)))
#     #     betas = sigmoid * (0.999 - 1e-4) + 1e-4
#     #     betas = betas.clamp(min=1e-4, max=0.999)
#     #     alphas = 1.0 - betas
#     #     alpha_bar = torch.cumprod(alphas, dim=0)
#     #     alpha_bar = torch.clamp(alpha_bar, 1e-8, 0.9999)
        
#     #     self.register_buffer('betas', betas)
#     #     self.register_buffer('alphas', alphas)
#     #     self.register_buffer('alpha_bar', alpha_bar)
#     #     self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
#     #     self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))
#     #     self.register_buffer('sqrt_alpha', torch.sqrt(alphas))
#     #     self.register_buffer('sqrt_beta', torch.sqrt(betas))

#     # REPLACE THIS METHOD in optimized_tabddpm.py
# # LOCATION: Around line 44-55 in the OptimizedTabDDPM class

#     def _setup_noise_schedule(self):
#         """Setup cosine noise schedule (from Improved DDPM paper)"""
#         def cosine_beta_schedule(timesteps, s=0.008):
#             steps = timesteps + 1
#             x = torch.linspace(0, timesteps, steps, device=self.device)
#             alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#             alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#             betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#             # return torch.clip(betas, 0.0001, 0.9999)
#             return torch.clip(betas, 1e-4, 0.999)  # Slightly tighter bounds
        
#         betas = cosine_beta_schedule(self.num_timesteps, s=0.008)
#         betas = betas.to(self.device)
#         alphas = 1.0 - betas
#         alpha_bar = torch.cumprod(alphas, dim=0)
#         alpha_bar = torch.clamp(alpha_bar, 1e-8, 0.9999)
        
#         self.register_buffer('betas', betas)
#         self.register_buffer('alphas', alphas)
#         self.register_buffer('alpha_bar', alpha_bar)
#         self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
#         self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))
#         self.register_buffer('sqrt_alpha', torch.sqrt(alphas))
#         self.register_buffer('sqrt_beta', torch.sqrt(betas))

#     def _q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         sqrt_alpha_t = self.sqrt_alpha_bar[t].view(-1, 1)
#         sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
#         noise = torch.randn_like(x)
#         x_t = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
#         return x_t, noise

#     def _quantile_loss(self, true: torch.Tensor, pred: torch.Tensor, quantile: float) -> torch.Tensor:
#         errors = true - pred
#         loss = torch.max(quantile * errors, (quantile - 1) * errors)
#         return torch.mean(loss)

#     def _loss(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device)
#         x_t, noise = self._q_sample(x, t)
        
#         if c.dim() > 1:
#             c = c.squeeze(-1)
        
#         c_embed = torch.nn.functional.one_hot(c, num_classes=self.num_classes).float()
        
#         if c_embed.dim() != 2 or x_t.dim() != 2:
#             raise ValueError(f"Shape mismatch: c_embed {c_embed.shape}, x_t {x_t.shape}")
        
#         input_x = torch.cat([x_t, c_embed], dim=-1)
#         predicted_noise = self.model(input_x)
        
#         quantile_loss = self._quantile_loss(noise, predicted_noise, self.quantile)
        
#         return quantile_loss, quantile_loss, torch.tensor(0.0, device=self.device)

#     def _get_numerical_categorical_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
#         numerical_cols = [col for col in data.columns if col != 'Class' and data[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
#         categorical_cols = ['Class']
#         return numerical_cols, categorical_cols

#     def preprocess_data(self, data: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
#         if self.numerical_cols is None:
#             self.numerical_cols, self.categorical_cols = self._get_numerical_categorical_columns(data)
        
#         logging.info(f"Preprocessing data with {len(self.numerical_cols)} numerical columns: {self.numerical_cols}")
        
#         if data[self.numerical_cols].isna().any().any():
#             data[self.numerical_cols] = data[self.numerical_cols].fillna(data[self.numerical_cols].median())
        
#         numerical_data = data[self.numerical_cols].copy()
        
#         if fit_scaler:
#             self.numerical_scalers = {}
#             for col in self.numerical_cols:
#                 scaler = RobustScaler()
#                 self.numerical_scalers[col] = scaler.fit(numerical_data[[col]])
        
#         for col in self.numerical_cols:
#             if col not in self.numerical_scalers:
#                 raise ValueError(f"Scaler for {col} not initialized. Run with fit_scaler=True first.")
#             numerical_data[[col]] = self.numerical_scalers[col].transform(numerical_data[[col]])
        
#         numerical_data = numerical_data.values
#         numerical_data = np.clip(numerical_data, -10, 10)
        
#         categorical_data = data[self.categorical_cols].values.flatten().astype(np.int64)
        
#         return numerical_data, categorical_data

#     def _inverse_transform_data(self, numerical_data: np.ndarray, categorical_data: np.ndarray) -> pd.DataFrame:
#         numerical_data = np.clip(numerical_data, -10, 10)
#         transformed_data = pd.DataFrame(numerical_data, columns=self.numerical_cols)
        
#         for col in self.numerical_cols:
#             if col in self.numerical_scalers:
#                 transformed_data[[col]] = self.numerical_scalers[col].inverse_transform(transformed_data[[col]])
        
#         if 'Amount' in transformed_data.columns:
#             transformed_data['Amount'] = transformed_data['Amount'].clip(lower=0)
        
#         transformed_data[self.categorical_cols[0]] = categorical_data.flatten()
        
#         if transformed_data.isna().any().any():
#             logging.warning("Inverse transformed data contains NaN. Replacing with 0.")
#             transformed_data = transformed_data.fillna(0)
        
#         return transformed_data

#     def _train_with_mixed_precision(self, train_loader: DataLoader, val_loader: Optional[DataLoader], epochs: int) -> float:
#         scaler = torch.amp.GradScaler('cuda') if self.device == 'cuda' else None
#         best_val_loss = float('inf')
        
#         for epoch in range(epochs):
#             self.current_epoch = epoch
#             self.model.train()
#             train_loss = 0
#             num_batches = 0
            
#             for batch_idx, (x, c) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
#                 x, c = x.to(self.device), c.to(self.device)
#                 total_loss, quantile_loss, _ = self._loss(x, c)
                
#                 if not torch.isfinite(total_loss):
#                     logging.warning(f"Loss is not finite at epoch {epoch+1}, batch {batch_idx+1}: {total_loss.item()}")
#                     continue
                
#                 if scaler:
#                     scaler.scale(total_loss).backward()
#                     scaler.unscale_(self.optimizer)
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                     scaler.step(self.optimizer)
#                     scaler.update()
#                 else:
#                     total_loss.backward()
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                     self.optimizer.step()
                
#                 self.scheduler.step()
#                 self.optimizer.zero_grad(set_to_none=True)
                
#                 train_loss += total_loss.item()
#                 num_batches += 1
                
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
            
#             avg_train_loss = train_loss / max(num_batches, 1)
            
#             val_loss = 0.0
#             if val_loader is not None:
#                 self.model.eval()
#                 val_batches = 0
#                 with torch.no_grad():
#                     for x, c in val_loader:
#                         x, c = x.to(self.device), c.to(self.device)
#                         total_loss, _, _ = self._loss(x, c)
#                         val_loss += total_loss.item()
#                         val_batches += 1
#                 val_loss = val_loss / max(val_batches, 1)
                
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     self.best_val_loss = val_loss
#                     self.best_model_state = self.model.state_dict()
#                     self.best_epoch = epoch + 1
#                     self.patience_counter = 0
#                 else:
#                     self.patience_counter += 1
#                     if self.patience_counter >= self.patience:
#                         logging.info(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
#                         break
                
#                 logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
#             else:
#                 logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}")
            
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
        
#         return best_val_loss

#     def train(self, train_data: pd.DataFrame, target_col: str = 'Class', epochs: int = 10,   # epochs: int = 1000
#               batch_size: int = 512, val_data: Optional[pd.DataFrame] = None) -> None:
#         logging.info(f"Starting training with batch_size={batch_size}, epochs={epochs}")
        
#         logging.info(f"Training data class distribution: Non-fraud = {len(train_data[train_data['Class'] == 0])}, Fraud = {len(train_data[train_data['Class'] == 1])}")
#         if val_data is not None:
#             logging.info(f"Validation data class distribution: Non-fraud = {len(val_data[val_data['Class'] == 0])}, Fraud = {len(val_data[val_data['Class'] == 1])}")
        
#         X_train, y_train = self.preprocess_data(train_data, fit_scaler=True)
#         train_dataset = TabularDataset(X_train, y_train)
        
#         fraud_ratio = y_train.mean()
#         class_weights = torch.tensor([1.0 / (1.0 - fraud_ratio), 1.0 / fraud_ratio], dtype=torch.float32)
#         sample_weights = class_weights[y_train]
#         sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
#         train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True
#         )
        
#         val_loader = None
#         if val_data is not None:
#             X_val, y_val = self.preprocess_data(val_data, fit_scaler=False)
#             val_dataset = TabularDataset(X_val, y_val)
#             val_loader = DataLoader(
#                 val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
#             )
        
#         total_steps = len(train_loader) * epochs
#         warmup_steps = int(total_steps * 0.1)
#         annealing_steps = max(total_steps - warmup_steps, 1)
#         logging.info(f"Scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}, annealing_steps={annealing_steps}")
        
#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(
#             self.optimizer,
#             lr_lambda=lambda step: min(step / warmup_steps, 1.0) if step < warmup_steps else 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / annealing_steps))
#         )
        
#         best_val_loss = self._train_with_mixed_precision(train_loader, val_loader, epochs)
        
#         if self.best_model_state is not None:
#             self.model.load_state_dict(self.best_model_state)
#             torch.save({
#                 'model_state': self.best_model_state,
#                 'epoch': self.best_epoch,
#                 'val_loss': self.best_val_loss
#             }, 'final_model.pth')
#             logging.info(f"Training completed. Best val loss: {self.best_val_loss:.6f}")
        
#         return best_val_loss

#     def optimize_hyperparameters(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
#                                 n_trials: int = 3, epochs: int = 10) -> Dict:               # n_trials: int = 30, epochs: int = 150)
#         def build_model(params: Dict, seed: int) -> "OptimizedTabDDPM":
#             return OptimizedTabDDPM(
#                 input_dim=self.input_dim,
#                 hidden_dim=params['hidden_dim'],
#                 num_layers=params['num_layers'],
#                 num_timesteps=params['num_timesteps'],
#                 device=self.device,
#                 seed=seed,
#                 lr=params['lr'],
#                 batch_size=params['batch_size'],
#                 quantile=params['quantile']
#             )

#         def objective(trial):
#             trial_seed = self.seed + trial.number
#             torch.manual_seed(trial_seed)
#             np.random.seed(trial_seed)

#             params = {
#                 'hidden_dim': trial.suggest_int('hidden_dim', 128, 1024, step=128),
#                 'num_layers': trial.suggest_int('num_layers', 2, 8, step=2),
#                 'num_timesteps': trial.suggest_categorical('num_timesteps', [100, 1000]),
#                 'lr': trial.suggest_float('lr', 1e-5, 3e-3, log=True),
#                 'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 4096]),
#                 'proportion_samples': trial.suggest_categorical('proportion_samples', [0.25, 0.5, 1.0, 2.0]),
#                 'quantile': trial.suggest_float('quantile', 0.1, 0.9, step=0.05)
#             }

#             logging.info(f"Trial {trial.number} parameters: {params}")

#             model = build_model(params, seed=trial_seed)
#             model.train(train_data, val_data=val_data, epochs=epochs, batch_size=params['batch_size'])

#             non_fraud_count = len(train_data[train_data['Class'] == 0])
#             fraud_count = len(train_data[train_data['Class'] == 1])
#             total_samples = int(non_fraud_count * params['proportion_samples'])
#             total_fraud_needed = int(total_samples * 0.1)
#             synthetic_fraud_needed = max(total_fraud_needed - fraud_count, 0)

#             synthetic_data = model.generate_synthetic_data(
#                 num_legitimate=0, num_fraud=synthetic_fraud_needed, batch_size=64
#             )
#             augmented_data = pd.concat([train_data, synthetic_data], ignore_index=True)

#             X_train = augmented_data.drop(columns=['Class']).values
#             y_train = augmented_data['Class'].values
#             X_val = val_data.drop(columns=['Class']).values
#             y_val = val_data['Class'].values

#             clf = XGBClassifier(
#                 n_estimators=300, max_depth=5, learning_rate=0.1,
#                 scale_pos_weight=(1.0 - y_train.mean()) / y_train.mean() * 1.5,
#                 eval_metric='logloss', random_state=trial_seed
#             )
#             clf.fit(X_train, y_train)
#             y_pred_proba = clf.predict_proba(X_val)[:, 1]

#             thresholds = np.linspace(0.1, 0.9, 20)
#             best_f1 = 0.0
#             for thresh in thresholds:
#                 y_pred = (y_pred_proba >= thresh).astype(int)
#                 f1 = f1_score(y_val, y_pred, zero_division=0)
#                 if f1 > best_f1:
#                     best_f1 = f1

#             logging.info(f"Trial {trial.number}: F1-score = {best_f1:.4f}")
#             return best_f1

#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=n_trials)

#         best_params = study.best_params
#         best_value = study.best_value

#         logging.info(f"Best hyperparameters: {best_params}")
#         logging.info(f"Best F1-score: {best_value:.6f}")

#         csv_file = 'best_optuna_params.csv'
#         with open(csv_file, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['Parameter', 'Value'])
#             for key, value in best_params.items():
#                 writer.writerow([key, value])
#             writer.writerow(['best_f1_score', best_value])
#         logging.info(f"Saved best hyperparameters to {csv_file}")

#         best_model = build_model(best_params, seed=self.seed)
#         self.__dict__.update(best_model.__dict__)

#         return best_params

#     def generate_synthetic_data(self, num_legitimate: int = 0, num_fraud: int = 1000, 
#                               batch_size: int = 64, fraud_median_amt: float = None) -> pd.DataFrame:
#         self.model.eval()
#         logging.info(f"Generating {num_legitimate} legitimate samples and {num_fraud} fraud samples")
        
#         torch.manual_seed(self.seed)

#         legitimate_samples = []
#         fraud_samples = []
        
#         for c_val, num_samples in [(0, num_legitimate), (1, num_fraud)]:
#             if num_samples == 0:
#                 continue
#             num_generated = 0
#             while num_generated < num_samples:
#                 current_batch_size = min(batch_size, num_samples - num_generated)
#                 x_t = torch.randn((current_batch_size, self.input_dim), device=self.device)
#                 c_tensor = torch.full((current_batch_size,), c_val, dtype=torch.long, device=self.device)
                
#                 for t in reversed(range(self.num_timesteps)):
#                     t_tensor = torch.full((current_batch_size,), t, device=self.device)
#                     with torch.no_grad():
#                         c_embed = torch.nn.functional.one_hot(c_tensor, num_classes=self.num_classes).float()
#                         input_x = torch.cat([x_t, c_embed], dim=-1)
#                         predicted_noise = self.model(input_x)
#                         alpha_t = self.alphas[t]
#                         beta_t = self.betas[t]
#                         sqrt_alpha_t = self.sqrt_alpha[t]
#                         sigma_t = torch.sqrt(self.betas[t]) if t > 0 else 0
#                         noise = torch.randn_like(x_t) if t > 0 else 0
#                         x_t = (1 / sqrt_alpha_t) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - self.alpha_bar[t])) * predicted_noise) + sigma_t * noise
#                         x_t = torch.clamp(x_t, -3, 3)
                
#                 samples = x_t.cpu().numpy()
#                 categorical = np.full((current_batch_size, 1), c_val)
                
#                 if c_val == 1 and fraud_median_amt is not None:
#                     temp_df = self._inverse_transform_data(samples, categorical)
#                     mask = temp_df['Amount'] >= fraud_median_amt * 0.7
#                     samples = samples[mask]
#                     categorical = categorical[mask]
#                     current_batch_size = len(samples)
#                     if current_batch_size == 0:
#                         logging.warning("All fraud samples filtered out in this batch. Continuing...")
#                         continue
                
#                 if c_val == 0:
#                     legitimate_samples.append((samples, categorical))
#                 else:
#                     fraud_samples.append((samples, categorical))
                
#                 num_generated += current_batch_size
                
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
        
#         numerical_data = []
#         categorical_data = []
        
#         if legitimate_samples:
#             legitimate_numerical = np.vstack([s[0] for s in legitimate_samples])[:num_legitimate]
#             legitimate_categorical = np.vstack([s[1] for s in legitimate_samples])[:num_legitimate]
#             numerical_data.append(legitimate_numerical)
#             categorical_data.append(legitimate_categorical)
        
#         if fraud_samples:
#             fraud_numerical = np.vstack([s[0] for s in fraud_samples])[:num_fraud]
#             fraud_categorical = np.vstack([s[1] for s in fraud_samples])[:num_fraud]
#             numerical_data.append(fraud_numerical)
#             categorical_data.append(fraud_categorical)
        
#         if not numerical_data:
#             raise ValueError("No samples generated. Check num_legitimate and num_fraud parameters.")
        
#         numerical_data = np.vstack(numerical_data)
#         categorical_data = np.vstack(categorical_data)
        
#         synthetic_df = self._inverse_transform_data(numerical_data, categorical_data)
#         synthetic_df = synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
#         return synthetic_df

# Commented on 28 Aug 25, 11:30 PM

import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List
import logging
import os
from dataset import TabularDataset
from models import MLPDiffusion
import optuna

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class OptimizedTabDDPM(nn.Module):
    """
    Optimized TabDDPM model for conditional synthetic data generation with Quantile Loss.
    Used for generating high-quality fraud samples by conditioning on the fraud class.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 4,
                 num_timesteps: int = 1000, device: str = 'auto', 
                 seed: int = 42, lr: float = 1e-4, batch_size: int = 512, quantile: float = 0.5, quantile_weight: float = 1.0,
                 cosine_s: float = 0.008, schedule_type: str = 'cosine'):
        super(OptimizedTabDDPM, self).__init__()
        self.device = self._get_device(device)
        self.input_dim = input_dim
        self.num_classes = 2
        self.num_timesteps = num_timesteps
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.quantile = quantile
        self.quantile_weight = quantile_weight
        self.cosine_s = cosine_s
        self.schedule_type = schedule_type
        torch.manual_seed(self.seed)
        
        logging.info(f"Initializing TabDDPM on {self.device} with input_dim={input_dim}, "
                     f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
                     f"lr={lr}, batch_size={batch_size}, seed={seed}, quantile={quantile}, "
                     f"cosine_s={cosine_s}, schedule_type={schedule_type}")
        
        d_layers = [hidden_dim] * num_layers
        self.model = MLPDiffusion(
            d_in=input_dim + self.num_classes,
            d_layers=d_layers,
            d_out=input_dim,
            dropout=0.1
        ).to(self.device)
        
        self._setup_noise_schedule()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = None
        self.numerical_scalers = {}
        self.categorical_cols = ['Class']
        self.numerical_cols = None
        self.best_val_loss = float('inf')
        self.patience = 200
        self.patience_counter = 0
        self.best_model_state = None
        self.best_epoch = 0
        self.current_epoch = 0

    def _get_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        return device

    def _setup_noise_schedule(self):
        """Setup noise schedule based on schedule_type"""
        if self.schedule_type == 'cosine':
            betas = self._cosine_beta_schedule(self.num_timesteps, self.cosine_s)
        elif self.schedule_type == 'linear':
            betas = self._linear_beta_schedule(self.num_timesteps)
        elif self.schedule_type == 'sigmoid':
            betas = self._sigmoid_beta_schedule(self.num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        betas = betas.to(self.device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar = torch.clamp(alpha_bar, 1e-8, 0.9999)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))
        self.register_buffer('sqrt_alpha', torch.sqrt(alphas))
        self.register_buffer('sqrt_beta', torch.sqrt(betas))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """Cosine noise schedule (from Improved DDPM paper)"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 1e-4, 0.999)

    def _linear_beta_schedule(self, timesteps: int):
        """Linear noise schedule"""
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps, device=self.device)

    def _sigmoid_beta_schedule(self, timesteps: int):
        """Sigmoid noise schedule"""
        timesteps_tensor = torch.arange(0, timesteps, dtype=torch.float32, device=self.device)
        sigmoid = 1 / (1 + torch.exp(-10 * (timesteps_tensor / timesteps - 0.5)))
        betas = sigmoid * (0.999 - 1e-4) + 1e-4
        return betas.clamp(min=1e-4, max=0.999)

    def _q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_t = self.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        # Gaussian
        noise = torch.randn_like(x)
        # With Student-t sampling:
        # gaussian = torch.randn_like(x)
        # chi2 = torch.distributions.Chi2(df=3.0).sample((x.shape[0],)).to(self.device).unsqueeze(-1)
        # noise = gaussian / torch.sqrt(chi2 / 3.0)

        x_t = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
        return x_t, noise

    def _mse_loss(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
            return torch.mean((true - pred) ** 2)
    
    def _quantile_loss(self, true: torch.Tensor, pred: torch.Tensor, quantile: float) -> torch.Tensor:
        errors = true - pred
        loss = torch.max(quantile * errors, (quantile - 1) * errors)
        return torch.mean(loss)

    def _predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
            sqrt_alpha_t = self.sqrt_alpha_bar[t].view(-1, 1)
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
            x0_hat = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            x0_hat = torch.clamp(x0_hat, -10.0, 10.0)
            return x0_hat

    def _loss(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device)
            x_t, noise = self._q_sample(x, t)
    
            if c.dim() > 1:
                c = c.squeeze(-1)
            c = c.long()  # one_hot requires LongTensor
    
            c_embed = torch.nn.functional.one_hot(c, num_classes=self.num_classes).float()
    
            if c_embed.dim() != 2 or x_t.dim() != 2:
                raise ValueError(f"Shape mismatch: c_embed {c_embed.shape}, x_t {x_t.shape}")
    
            input_x = torch.cat([x_t, c_embed], dim=-1)
            predicted_noise = self.model(input_x)  # single forward pass, reused for both terms
            x0_hat = self._predict_x0(x_t, t, predicted_noise)
    
            mse_loss = self._mse_loss(noise, predicted_noise)
            #quantile_loss = self._quantile_loss(noise, predicted_noise, self.quantile)
            quantile_loss = self._quantile_loss(x, x0_hat, self.quantile)
    
            total_loss = mse_loss + self.quantile_weight * quantile_loss   
    
            return total_loss, mse_loss, quantile_loss
    

    # def _quantile_loss(self, true: torch.Tensor, pred: torch.Tensor, quantile: float) -> torch.Tensor:
    #     errors = true - pred
    #     loss = torch.max(quantile * errors, (quantile - 1) * errors)
    #     return torch.mean(loss)

    # def _loss(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device)
    #     x_t, noise = self._q_sample(x, t)
        
    #     if c.dim() > 1:
    #         c = c.squeeze(-1)
        
    #     c_embed = torch.nn.functional.one_hot(c, num_classes=self.num_classes).float()
        
    #     if c_embed.dim() != 2 or x_t.dim() != 2:
    #         raise ValueError(f"Shape mismatch: c_embed {c_embed.shape}, x_t {x_t.shape}")
        
    #     input_x = torch.cat([x_t, c_embed], dim=-1)
    #     predicted_noise = self.model(input_x)
        
    #     quantile_loss = self._quantile_loss(noise, predicted_noise, self.quantile)
        
    #     return quantile_loss, quantile_loss, torch.tensor(0.0, device=self.device)

    def _get_numerical_categorical_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numerical_cols = [col for col in data.columns if col != 'Class' and data[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        categorical_cols = ['Class']
        return numerical_cols, categorical_cols

    def preprocess_data(self, data: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.numerical_cols is None:
            self.numerical_cols, self.categorical_cols = self._get_numerical_categorical_columns(data)
        
        logging.info(f"Preprocessing data with {len(self.numerical_cols)} numerical columns: {self.numerical_cols}")
        
        if data[self.numerical_cols].isna().any().any():
            data[self.numerical_cols] = data[self.numerical_cols].fillna(data[self.numerical_cols].median())
        
        numerical_data = data[self.numerical_cols].copy()
        
        if fit_scaler:
            self.numerical_scalers = {}
            for col in self.numerical_cols:
                scaler = RobustScaler()
                self.numerical_scalers[col] = scaler.fit(numerical_data[[col]])
        
        for col in self.numerical_cols:
            if col not in self.numerical_scalers:
                raise ValueError(f"Scaler for {col} not initialized. Run with fit_scaler=True first.")
            numerical_data[[col]] = self.numerical_scalers[col].transform(numerical_data[[col]])
        
        numerical_data = numerical_data.values
        numerical_data = np.clip(numerical_data, -10, 10)
        
        categorical_data = data[self.categorical_cols].values.flatten().astype(np.int64)
        
        return numerical_data, categorical_data

    def _inverse_transform_data(self, numerical_data: np.ndarray, categorical_data: np.ndarray) -> pd.DataFrame:
        numerical_data = np.clip(numerical_data, -10, 10)
        transformed_data = pd.DataFrame(numerical_data, columns=self.numerical_cols)
        
        for col in self.numerical_cols:
            if col in self.numerical_scalers:
                transformed_data[[col]] = self.numerical_scalers[col].inverse_transform(transformed_data[[col]])
        
        if 'Amount' in transformed_data.columns:
            transformed_data['Amount'] = transformed_data['Amount'].clip(lower=0)
        
        transformed_data[self.categorical_cols[0]] = categorical_data.flatten()
        
        if transformed_data.isna().any().any():
            logging.warning("Inverse transformed data contains NaN. Replacing with 0.")
            transformed_data = transformed_data.fillna(0)
        
        return transformed_data

    def _train_with_mixed_precision(self, train_loader: DataLoader, val_loader: Optional[DataLoader], epochs: int) -> float:
        scaler = torch.amp.GradScaler('cuda') if self.device == 'cuda' else None
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.model.train()
            train_loss = 0
            num_batches = 0
            
            for batch_idx, (x, c) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                x, c = x.to(self.device), c.to(self.device)
                total_loss, quantile_loss, _ = self._loss(x, c)
                
                if not torch.isfinite(total_loss):
                    logging.warning(f"Loss is not finite at epoch {epoch+1}, batch {batch_idx+1}: {total_loss.item()}")
                    continue
                
                if scaler:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                train_loss += total_loss.item()
                num_batches += 1
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / max(num_batches, 1)
            
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                val_batches = 0
                with torch.no_grad():
                    for x, c in val_loader:
                        x, c = x.to(self.device), c.to(self.device)
                        total_loss, _, _ = self._loss(x, c)
                        val_loss += total_loss.item()
                        val_batches += 1
                val_loss = val_loss / max(val_batches, 1)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logging.info(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                        break
                
                logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return best_val_loss

    def train(self, train_data: pd.DataFrame, target_col: str = 'Class', epochs: int = 1000,   # epochs: int = 1000
              batch_size: int = 512, val_data: Optional[pd.DataFrame] = None) -> None:
        logging.info(f"Starting training with batch_size={batch_size}, epochs={epochs}")
        
        logging.info(f"Training data class distribution: Non-fraud = {len(train_data[train_data['Class'] == 0])}, Fraud = {len(train_data[train_data['Class'] == 1])}")
        if val_data is not None:
            logging.info(f"Validation data class distribution: Non-fraud = {len(val_data[val_data['Class'] == 0])}, Fraud = {len(val_data[val_data['Class'] == 1])}")
        
        X_train, y_train = self.preprocess_data(train_data, fit_scaler=True)
        train_dataset = TabularDataset(X_train, y_train)
        
        fraud_ratio = y_train.mean()
        class_weights = torch.tensor([1.0 / (1.0 - fraud_ratio), 1.0 / fraud_ratio], dtype=torch.float32)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True
        )
        
        val_loader = None
        if val_data is not None:
            X_val, y_val = self.preprocess_data(val_data, fit_scaler=False)
            val_dataset = TabularDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * 0.1)
        annealing_steps = max(total_steps - warmup_steps, 1)
        logging.info(f"Scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}, annealing_steps={annealing_steps}")
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step / warmup_steps, 1.0) if step < warmup_steps else 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / annealing_steps))
        )
        
        best_val_loss = self._train_with_mixed_precision(train_loader, val_loader, epochs)
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            torch.save({
                'model_state': self.best_model_state,
                'epoch': self.best_epoch,
                'val_loss': self.best_val_loss
            }, 'final_model.pth')
            logging.info(f"Training completed. Best val loss: {self.best_val_loss:.6f}")
        
        return best_val_loss

    # def optimize_hyperparameters(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
    #                             n_trials: int = 30, epochs: int = 150) -> Dict:   # n_trials: int = 30, epochs: int = 150)
    #     study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
                 
    #     def build_model(params: Dict, seed: int) -> "OptimizedTabDDPM":
    #         return OptimizedTabDDPM(
    #             input_dim=self.input_dim,
    #             hidden_dim=params['hidden_dim'],
    #             num_layers=params['num_layers'],
    #             num_timesteps=params['num_timesteps'],
    #             device=self.device,
    #             seed=seed,
    #             lr=params['lr'],
    #             batch_size=params['batch_size'],
    #             quantile=params['quantile'],
    #             cosine_s=params.get('cosine_s', 0.008),
    #             schedule_type=params.get('schedule_type', 'cosine')
    #         )

    #     def objective(trial):
    #         trial_seed = self.seed + trial.number
    #         torch.manual_seed(trial_seed)
    #         np.random.seed(trial_seed)

    #         # Original hyperparameters
    #         params = {
    #             'hidden_dim': trial.suggest_int('hidden_dim', 128, 1024, step=128),
    #             'num_layers': trial.suggest_int('num_layers', 2, 8, step=2),
    #             'num_timesteps': trial.suggest_categorical('num_timesteps', [100, 1000]),
    #             'lr': trial.suggest_float('lr', 1e-5, 3e-3, log=True),
    #             'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 4096]),
    #             'proportion_samples': trial.suggest_categorical('proportion_samples', [0.25, 0.5, 1.0, 2.0]),
    #             'quantile': trial.suggest_float('quantile', 0.6, 0.9, step=0.05)
    #         }
            
    #         # New noise schedule hyperparameters
    #         schedule_type = trial.suggest_categorical('schedule_type', ['cosine', 'linear', 'sigmoid'])
    #         params['schedule_type'] = schedule_type
            
    #         # Only add cosine_s parameter if using cosine schedule
    #         if schedule_type == 'cosine':
    #             params['cosine_s'] = trial.suggest_float('cosine_s', 0.001, 0.1, log=True)
    #         else:
    #             params['cosine_s'] = 0.008  # Default value for non-cosine schedules

    #         logging.info(f"Trial {trial.number} parameters: {params}")

    #         model = build_model(params, seed=trial_seed)
    #         model.train(train_data, val_data=val_data, epochs=epochs, batch_size=params['batch_size'])

    #         non_fraud_count = len(train_data[train_data['Class'] == 0])
    #         fraud_count = len(train_data[train_data['Class'] == 1])
    #         total_samples = int(non_fraud_count * params['proportion_samples'])
    #         total_fraud_needed = int(total_samples * 0.1)
    #         synthetic_fraud_needed = max(total_fraud_needed - fraud_count, 0)

    #         synthetic_data = model.generate_synthetic_data(
    #             num_legitimate=0, num_fraud=synthetic_fraud_needed, batch_size=64
    #         )
    #         augmented_data = pd.concat([train_data, synthetic_data], ignore_index=True)

    #         X_train = augmented_data.drop(columns=['Class']).values
    #         y_train = augmented_data['Class'].values
    #         X_val = val_data.drop(columns=['Class']).values
    #         y_val = val_data['Class'].values

    #         clf = XGBClassifier(
    #             n_estimators=300, max_depth=5, learning_rate=0.1,
    #             scale_pos_weight=(1.0 - y_train.mean()) / y_train.mean() * 1.5,
    #             eval_metric='logloss', random_state=trial_seed
    #         )
    #         clf.fit(X_train, y_train)
    #         y_pred_proba = clf.predict_proba(X_val)[:, 1]

    #         thresholds = np.linspace(0.1, 0.9, 20)
    #         best_f1 = 0.0
    #         for thresh in thresholds:
    #             y_pred = (y_pred_proba >= thresh).astype(int)
    #             f1 = f1_score(y_val, y_pred, zero_division=0)
    #             if f1 > best_f1:
    #                 best_f1 = f1

    #         logging.info(f"Trial {trial.number}: F1-score = {best_f1:.4f} (Schedule: {schedule_type})")
    #         return best_f1

    #     study = optuna.create_study(direction='maximize')
    #     study.optimize(objective, n_trials=n_trials)

    #     best_params = study.best_params
    #     best_value = study.best_value

    #     logging.info(f"Best hyperparameters: {best_params}")
    #     logging.info(f"Best F1-score: {best_value:.6f}")

    #     csv_file = 'best_optuna_params.csv'
    #     with open(csv_file, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['Parameter', 'Value'])
    #         for key, value in best_params.items():
    #             writer.writerow([key, value])
    #         writer.writerow(['best_f1_score', best_value])
    #     logging.info(f"Saved best hyperparameters to {csv_file}")

    #     best_model = build_model(best_params, seed=self.seed)
    #     self.__dict__.update(best_model.__dict__)

    #     return best_params

    
    def optimize_hyperparameters(self, train_data: pd.DataFrame, val_data: pd.DataFrame, n_trials: int = 30, epochs: int = 150) -> Dict:

        def build_model(params: Dict, seed: int) -> "OptimizedTabDDPM":
            return OptimizedTabDDPM(
                input_dim=self.input_dim,
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                num_timesteps=params['num_timesteps'],
                device=self.device,
                seed=seed,
                lr=params['lr'],
                batch_size=params['batch_size'],
                quantile=params['quantile'],
                quantile_weight=params['quantile_weight'],
                cosine_s=params.get('cosine_s', 0.008),
                schedule_type=params.get('schedule_type', 'cosine')
            )

        def objective(trial):
            trial_seed = self.seed + trial.number
            torch.manual_seed(trial_seed)
            np.random.seed(trial_seed)
            params = {
                'hidden_dim': trial.suggest_int('hidden_dim', 128, 1024, step=128),
                'num_layers': trial.suggest_int('num_layers', 2, 8, step=2),
                'num_timesteps': trial.suggest_categorical('num_timesteps', [100, 1000]),
                'lr': trial.suggest_float('lr', 1e-5, 3e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 4096]),
                'proportion_samples': trial.suggest_categorical('proportion_samples', [0.25, 0.5, 1.0, 2.0]),
                'quantile': trial.suggest_float('quantile', 0.6, 0.9, step=0.05),
                'quantile_weight': trial.suggest_float('quantile_weight', 0.1, 1.0, log=True)
            }
            schedule_type = trial.suggest_categorical('schedule_type', ['cosine', 'linear', 'sigmoid'])
            params['schedule_type'] = schedule_type
            if schedule_type == 'cosine':
                params['cosine_s'] = trial.suggest_float('cosine_s', 0.001, 0.1, log=True)
            else:
                params['cosine_s'] = 0.008
            logging.info(f"Trial {trial.number} parameters: {params}")
            model = build_model(params, seed=trial_seed)
            model.train(train_data, val_data=val_data, epochs=epochs, batch_size=params['batch_size'])
            non_fraud_count = len(train_data[train_data['Class'] == 0])
            fraud_count = len(train_data[train_data['Class'] == 1])
            total_samples = int(non_fraud_count * params['proportion_samples'])
            total_fraud_needed = int(total_samples * 0.1)
            synthetic_fraud_needed = max(total_fraud_needed - fraud_count, 0)
            synthetic_data = model.generate_synthetic_data(
                num_legitimate=0, num_fraud=synthetic_fraud_needed, batch_size=64
            )
            augmented_data = pd.concat([train_data, synthetic_data], ignore_index=True)
            X_train = augmented_data.drop(columns=['Class']).values
            y_train = augmented_data['Class'].values
            X_val = val_data.drop(columns=['Class']).values
            y_val = val_data['Class'].values
            clf = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=(1.0 - y_train.mean()) / y_train.mean() * 1.5,
                eval_metric='logloss',
                random_state=trial_seed
            )
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_val)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 20)
            best_f1 = 0.0
            for thresh in thresholds:
                y_pred = (y_pred_proba >= thresh).astype(int)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
            logging.info(f"Trial {trial.number}: F1-score = {best_f1:.4f} (Schedule: {schedule_type})")
            return best_f1

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_value = study.best_value
        logging.info(f"Best hyperparameters: {best_params}")
        logging.info(f"Best F1-score: {best_value:.6f}")
        csv_file = 'best_optuna_params.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Parameter', 'Value'])
            for key, value in best_params.items():
                writer.writerow([key, value])
            writer.writerow(['best_f1_score', best_value])
        logging.info(f"Saved best hyperparameters to {csv_file}")
        best_model = build_model(best_params, seed=self.seed)
        self.__dict__.update(best_model.__dict__)
        return best_params
    

    def generate_synthetic_data(self, num_legitimate: int = 0, num_fraud: int = 1000, 
                              batch_size: int = 64, fraud_median_amt: float = None) -> pd.DataFrame:
        self.model.eval()
        logging.info(f"Generating {num_legitimate} legitimate samples and {num_fraud} fraud samples")
        
        torch.manual_seed(self.seed)

        legitimate_samples = []
        fraud_samples = []
        
        for c_val, num_samples in [(0, num_legitimate), (1, num_fraud)]:
            if num_samples == 0:
                continue
            num_generated = 0
            while num_generated < num_samples:
                current_batch_size = min(batch_size, num_samples - num_generated)
                # Gaussian prior
                x_t = torch.randn((current_batch_size, self.input_dim), device=self.device)
                # With Student-t sampling:
                # gaussian = torch.randn((current_batch_size, self.input_dim), device=self.device)
                # chi2 = torch.distributions.Chi2(df=3.0).sample((current_batch_size,)).to(self.device).unsqueeze(-1)
                # x_t = gaussian / torch.sqrt(chi2 / 3.0)
                c_tensor = torch.full((current_batch_size,), c_val, dtype=torch.long, device=self.device)
                
                for t in reversed(range(self.num_timesteps)):
                    t_tensor = torch.full((current_batch_size,), t, device=self.device)
                    with torch.no_grad():
                        c_embed = torch.nn.functional.one_hot(c_tensor, num_classes=self.num_classes).float()
                        input_x = torch.cat([x_t, c_embed], dim=-1)
                        predicted_noise = self.model(input_x)
                        alpha_t = self.alphas[t]
                        beta_t = self.betas[t]
                        sqrt_alpha_t = self.sqrt_alpha[t]
                        sigma_t = torch.sqrt(self.betas[t]) if t > 0 else 0
                        # Gaussian
                        noise = torch.randn_like(x_t) if t > 0 else 0
                        # With Student-t sampling:
                        # if t > 0:
                        #     g = torch.randn_like(x_t)
                        #     chi2 = torch.distributions.Chi2(df=3.0).sample((x_t.shape[0],)).to(self.device).unsqueeze(-1)
                        #     noise = g / torch.sqrt(chi2 / 3.0)
                        # else:
                        #     noise = 0
                        x_t = (1 / sqrt_alpha_t) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - self.alpha_bar[t])) * predicted_noise) + sigma_t * noise
                        x_t = torch.clamp(x_t, -3, 3)
                
                samples = x_t.cpu().numpy()
                categorical = np.full((current_batch_size, 1), c_val)
                
                if c_val == 1 and fraud_median_amt is not None:
                    temp_df = self._inverse_transform_data(samples, categorical)
                    mask = temp_df['Amount'] >= fraud_median_amt * 0.7
                    samples = samples[mask]
                    categorical = categorical[mask]
                    current_batch_size = len(samples)
                    if current_batch_size == 0:
                        logging.warning("All fraud samples filtered out in this batch. Continuing...")
                        continue
                
                if c_val == 0:
                    legitimate_samples.append((samples, categorical))
                else:
                    fraud_samples.append((samples, categorical))
                
                num_generated += current_batch_size
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        numerical_data = []
        categorical_data = []
        
        if legitimate_samples:
            legitimate_numerical = np.vstack([s[0] for s in legitimate_samples])[:num_legitimate]
            legitimate_categorical = np.vstack([s[1] for s in legitimate_samples])[:num_legitimate]
            numerical_data.append(legitimate_numerical)
            categorical_data.append(legitimate_categorical)
        
        if fraud_samples:
            fraud_numerical = np.vstack([s[0] for s in fraud_samples])[:num_fraud]
            fraud_categorical = np.vstack([s[1] for s in fraud_samples])[:num_fraud]
            numerical_data.append(fraud_numerical)
            categorical_data.append(fraud_categorical)
        
        if not numerical_data:
            raise ValueError("No samples generated. Check num_legitimate and num_fraud parameters.")
        
        numerical_data = np.vstack(numerical_data)
        categorical_data = np.vstack(categorical_data)
        
        synthetic_df = self._inverse_transform_data(numerical_data, categorical_data)
        synthetic_df = synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return synthetic_df
