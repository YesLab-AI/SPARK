import os
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5EncoderModel, AutoTokenizer
from peft import PeftModel, PeftConfig
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

try:
    plt.rcParams['font.family'] = 'Arial'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False 

BASE_MODEL_NAME = "ankh3-large"
ADAPTER_PATH = Path("models/ankh3_adalora_finetune") 
REGRESSION_DATA_PATH = Path("data/regression_data.csv")
OUTPUT_DIR = Path("results/final_oracle") 

BATCH_SIZE = 32
HIDDEN_DIM = 1536      
MAX_EPOCHS = 100
PATIENCE = 10          

LEARNING_RATE_HEAD = 5e-4  
WEIGHT_DECAY = 1e-2        

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def calculate_metrics(y_true, y_pred):
    if y_pred.ndim > 1: y_pred = y_pred.flatten()
    y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_true))
    
    if np.std(y_pred) < 1e-6:
        pcc, spearman = 0.0, 0.0
    else:
        pcc = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
        
    if np.isnan(pcc): pcc = 0.0
    if np.isnan(spearman): spearman = 0.0
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2)) 
    
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    denominator = var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8
    ccc = (2 * pcc * sd_true * sd_pred) / denominator
    if np.isnan(ccc): ccc = 0.0
    
    return {'PCC': pcc, 'Spearman': spearman, 'CCC': ccc, 'MAE': mae, 'RMSE': rmse}

def compute_hybrid_oracle_score(sequence: str, neural_network_score: float) -> float:
    seq_clean = re.sub(r'[^A-Z]', '', sequence.upper())
    
    if len(seq_clean) < 10:
        return 0.0
        
    penalty = 0.0
    
    if len(seq_clean) > 2:
        if seq_clean[1] not in ['K', 'R'] or seq_clean[2] not in ['K', 'R']:
            penalty -= 0.2  

    if len(seq_clean) >= 3:
        if seq_clean[-3] != 'A' or seq_clean[-1] != 'A':
            penalty -= 0.2  
            
    final_score = neural_network_score + penalty
    
    return max(0.0, min(1.0, final_score))

class CnnPoolingLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        
    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask == 0, -1e4)
        return torch.max(x, dim=2)[0]

class SwiGLUHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = 1024
        self.w_gate = nn.Linear(input_dim, hidden_dim)
        self.w_transform = nn.Linear(input_dim, hidden_dim)
        self.ln_glu = nn.LayerNorm(hidden_dim)
        self.w_out = nn.Linear(hidden_dim, 256)
        self.ln_out = nn.LayerNorm(256)
        self.predictor = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)
        initialize_weights(self)
        
    def forward(self, x):
        gate = self.w_gate(x).float()
        transform = self.w_transform(x).float()
        glu_out = F.silu(gate) * transform
        glu_out = self.ln_glu(glu_out.to(x.dtype)) 
        h = self.dropout(F.gelu(self.ln_out(self.w_out(self.dropout(glu_out)))))
        return self.predictor(h)

class FrozenEncoderOracle(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.pooler = CnnPoolingLayer(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = SwiGLUHead(hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = encoder_outputs.last_hidden_state
        pooled_features = self.pooler(hidden_states, attention_mask)
        normalized_pooled = self.norm(pooled_features)
        return self.head(normalized_pooled)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_STATE)

    df = pd.read_csv(REGRESSION_DATA_PATH)
    sequences = df['sequence'].tolist()
    labels = df['normalized_value'].values.astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, legacy=False)
    
    encodings = tokenizer(sequences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    X_ids = encodings['input_ids']
    X_masks = encodings['attention_mask']

    train_idx, temp_idx = train_test_split(np.arange(len(labels)), test_size=0.20, random_state=RANDOM_STATE)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_STATE)
    
    X_ids_train, X_masks_train, y_train_raw = X_ids[train_idx], X_masks[train_idx], labels[train_idx]
    X_ids_val, X_masks_val, y_val_raw = X_ids[val_idx], X_masks[val_idx], labels[val_idx]
    X_ids_test, X_masks_test, y_test_raw = X_ids[test_idx], X_masks[test_idx], labels[test_idx]

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
    joblib.dump(scaler_y, OUTPUT_DIR / "scaler_y.pkl")

    dl_train = DataLoader(TensorDataset(X_ids_train, X_masks_train, torch.tensor(y_train).float()), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl_val = DataLoader(TensorDataset(X_ids_val, X_masks_val, torch.tensor(y_val).float()), batch_size=BATCH_SIZE, shuffle=False)
    dl_test = DataLoader(TensorDataset(X_ids_test, X_masks_test, torch.tensor(y_test).float()), batch_size=BATCH_SIZE, shuffle=False)

    base_model = T5EncoderModel.from_pretrained(BASE_MODEL_NAME)
    
    if ADAPTER_PATH.exists():
        peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
        peft_config.task_type = "FEATURE_EXTRACTION"
        model_with_prior = PeftModel.from_pretrained(base_model, ADAPTER_PATH, config=peft_config)
        merged_base_model = model_with_prior.merge_and_unload()
    else:
        merged_base_model = base_model

    oracle = FrozenEncoderOracle(merged_base_model, HIDDEN_DIM).to(DEVICE)
    
    trainable_params = list(oracle.pooler.parameters()) + list(oracle.norm.parameters()) + list(oracle.head.parameters())

    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )
    mse_criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    save_path = OUTPUT_DIR / "final_e2e_oracle.pth"
    
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(MAX_EPOCHS):
        oracle.train()
        train_loss = 0
        valid_batches = 0
        
        for bx_ids, bx_masks, by in tqdm(dl_train, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} Train"):
            optimizer.zero_grad()
            
            preds = oracle(bx_ids.to(DEVICE), bx_masks.to(DEVICE)).squeeze(-1)
            loss = mse_criterion(preds.float(), by.to(DEVICE).float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            loss_val = loss.item()
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                train_loss += loss_val
                valid_batches += 1
                
        oracle.eval()
        val_loss = 0
        with torch.no_grad():
            for bx_ids, bx_masks, by in dl_val:
                preds = oracle(bx_ids.to(DEVICE), bx_masks.to(DEVICE)).squeeze(-1)
                loss = mse_criterion(preds.float(), by.to(DEVICE).float())
                val_loss += loss.item()
                
        avg_train_loss = train_loss / valid_batches if valid_batches > 0 else float('inf')
        avg_val_loss = val_loss / len(dl_val)
        
        scheduler.step()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            head_state = {k: v for k, v in oracle.state_dict().items() if not k.startswith("encoder.")}
            torch.save(head_state, save_path)
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:03d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f} | Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            break
            
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train MSE', marker='o', color='#7FBCE6', linewidth=2)
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Validation MSE', marker='o', color='#F06055', linewidth=2)
    
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = min(history['val_loss'])
    plt.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch, best_val, color='red', s=100, zorder=5)

    plt.title('Training and Validation Loss Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('MSE Loss', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=300)
    plt.close()

    print("\n--- Evaluating Best Model on Independent Test Set ---")

    oracle.load_state_dict(torch.load(save_path, map_location=DEVICE), strict=False)
    oracle.eval()
    
    test_preds_raw = []
    test_trues = []
    
    with torch.no_grad():
        for bx_ids, bx_masks, by in dl_test:
            preds = oracle(bx_ids.to(DEVICE), bx_masks.to(DEVICE)).squeeze(-1)
            test_preds_raw.extend(preds.cpu().numpy())
            test_trues.extend(by.cpu().numpy())
            
    test_sequences = [sequences[i] for i in test_idx]
    
    test_preds_inv = scaler_y.inverse_transform(np.array(test_preds_raw).reshape(-1, 1)).flatten()
    test_trues_inv = scaler_y.inverse_transform(np.array(test_trues).reshape(-1, 1)).flatten()
    
    metrics = calculate_metrics(test_trues_inv, test_preds_inv)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)
    
    print("Final NN Test Metrics (Excluding Hybrid Penalties):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    test_preds_hybrid_scaled = []
    test_preds_hybrid_inv = []
    for seq, pred_raw in zip(test_sequences, test_preds_raw):
        hybrid_scaled = compute_hybrid_oracle_score(seq, float(pred_raw))
        test_preds_hybrid_scaled.append(hybrid_scaled)
        hybrid_inv = scaler_y.inverse_transform(np.array([[hybrid_scaled]]))[0, 0]
        test_preds_hybrid_inv.append(hybrid_inv)

    results_df = pd.DataFrame({
        'Sequence': test_sequences,
        'True_Value': test_trues_inv,
        'NN_Raw_Score_Scaled': test_preds_raw,
        'NN_Predicted_Value': test_preds_inv,
        'Hybrid_Score_Scaled': test_preds_hybrid_scaled,
        'Hybrid_Predicted_Value': test_preds_hybrid_inv
    })
    results_df.to_csv(OUTPUT_DIR / "test_predictions_hybrid.csv", index=False)

if __name__ == "__main__":
    main()