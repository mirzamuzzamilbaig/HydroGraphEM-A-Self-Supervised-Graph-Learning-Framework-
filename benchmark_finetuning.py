import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, f1_score, accuracy_score
import os
import pickle
from scalers import FeatureScaler

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, is_classifier=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        self.is_classifier = is_classifier

    def forward(self, x):
        return self.net(x)

def train_and_eval_head(X_train, y_train, X_val, y_val, out_dim, is_classifier=False, epochs=10):
    # Flatten inputs for standard MLP training
    # Shape: [N_samples * N_reaches, hidden_dim]
    X_tr_flat = X_train.reshape(-1, X_train.shape[-1])
    y_tr_flat = y_train.reshape(-1, out_dim) if (out_dim > 1 and not is_classifier) else y_train.ravel()
    
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    y_val_flat = y_val.reshape(-1, out_dim) if (out_dim > 1 and not is_classifier) else y_val.ravel()
    
    # Remove NaNs if any
    mask_tr = ~np.isnan(y_tr_flat).any(axis=1) if (out_dim > 1 and not is_classifier) else ~np.isnan(y_tr_flat)
    X_tr_flat = X_tr_flat[mask_tr]
    y_tr_flat = y_tr_flat[mask_tr]
    
    mask_val = ~np.isnan(y_val_flat).any(axis=1) if (out_dim > 1 and not is_classifier) else ~np.isnan(y_val_flat)
    X_val_flat = X_val_flat[mask_val]
    y_val_flat = y_val_flat[mask_val]

    # Target scaling for regression to prevent gradient explosion
    y_scaler = None
    y_val_flat_target_scale = y_val_flat.copy()
    if not is_classifier:
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        y_tr_flat_2d = y_tr_flat.reshape(-1, 1) if (out_dim == 1) else y_tr_flat
        y_tr_flat_scaled = y_scaler.fit_transform(y_tr_flat_2d)
        y_tr_flat = y_tr_flat_scaled.ravel() if (out_dim == 1) else y_tr_flat_scaled
        
        y_val_flat_2d = y_val_flat.reshape(-1, 1) if (out_dim == 1) else y_val_flat
        y_val_flat_scaled = y_scaler.transform(y_val_flat_2d)
        y_val_flat = y_val_flat_scaled.ravel() if (out_dim == 1) else y_val_flat_scaled

    # Convert to PyTorch tensors
    t_X_tr = torch.tensor(X_tr_flat, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr_flat, dtype=torch.float32 if not is_classifier else torch.long)
    t_X_val = torch.tensor(X_val_flat, dtype=torch.float32)
    
    head = MLPHead(X_train.shape[-1], out_dim, is_classifier=is_classifier)
    optimizer = optim.Adam(head.parameters(), lr=0.01)
    
    if is_classifier:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
        
    # Train MLP
    head.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        outputs = head(t_X_tr)
        if out_dim == 1 and not is_classifier:
            outputs = outputs.squeeze(-1)
        loss = criterion(outputs, t_y_tr)
        loss.backward()
        optimizer.step()
        
    # Eval
    head.eval()
    with torch.no_grad():
        pred_val = head(t_X_val)
        if is_classifier:
            pred_classes = torch.argmax(pred_val, dim=1).numpy()
            acc = accuracy_score(y_val_flat, pred_classes)
            f1 = f1_score(y_val_flat, pred_classes, zero_division=0)
            return {'Accuracy': acc, 'F1-Score': f1}
        else:
            if out_dim == 1:
                pred_val = pred_val.squeeze(-1)
            pred_val = pred_val.numpy()
            
            # Inverse transform predictions to original scale
            pred_val_2d = pred_val.reshape(-1, 1) if (out_dim == 1) else pred_val
            pred_val_orig = y_scaler.inverse_transform(pred_val_2d)
            pred_val = pred_val_orig.ravel() if (out_dim == 1) else pred_val_orig
            
            r2 = r2_score(y_val_flat_target_scale, pred_val)
            mse = np.mean((y_val_flat_target_scale - pred_val)**2)
            return {'R2': r2, 'MSE': mse}

# HST-PAGT Encoder definition for loading weights
from train_pretrain import HST_PAGT_Encoder, compute_positional_encodings

def main():
    print("=== Fine-Tuning downstream benchmark heads on HydroGraphEM Encoder ===")
    
    # 1. Load reaches metadata and pretraining dataset to extract encoder inputs
    df_meta = pd.read_csv('metadata_reaches.csv')
    df_pre = pd.read_parquet('hydrographfm_pretrain_full.parquet')
    
    num_reaches = 483
    edges_fwd = []
    edges_bwd = []
    for idx, row in df_meta.iterrows():
        u = int(row['Reach_ID'])
        d = row['Downstream_ID']
        if not pd.isna(d):
            v = int(d)
            if 1 <= v <= num_reaches:
                edges_fwd.append((u - 1, v - 1))
                edges_bwd.append((v - 1, u - 1))
    edge_index = torch.tensor(edges_fwd + edges_bwd, dtype=torch.long).t().contiguous()
    pos_enc = compute_positional_encodings(df_meta, df_pre)
    
    meta_cols = ['Sample_ID', 'Time_Lag', 'Date', 'Reach_ID', 'Subbasin_ID', 'Upstream_ID', 'Downstream_ID']
    feat_cols = [c for c in df_pre.columns if c not in meta_cols]
    
    # Load and apply feature scaling
    print("  Loading feature scaler...")
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.scale_features_matrix(feat_cols, df_pre)
    
    num_samples = 205
    # Reshape to [num_samples, num_reaches, 24, D]
    X_train_full = X_scaled.reshape(num_reaches, num_samples, 24, -1)
    X_train_full = np.transpose(X_train_full, (1, 0, 2, 3))
    
    # 2. Initialize GAT Encoder and load weights
    print("\nLoading Pretrained Model A (GATConv) Encoder...")
    encoder = HST_PAGT_Encoder(len(feat_cols), 64, edge_index, pos_enc, conv_type='gat')
    if os.path.exists('hydrographfm_encoder_v1.pt'):
        encoder.load_state_dict(torch.load('hydrographfm_encoder_v1.pt'))
        print("  Encoder weights loaded from 'hydrographfm_encoder_v1.pt'")
    else:
        print("  WARNING: Encoder weights file not found! Using initialized weights.")
        
    encoder.eval()
    
    # 3. Extract last step embeddings for all sequences
    print("Extracting reach embeddings...")
    emb_list = []
    with torch.no_grad():
        for b in range(int(np.ceil(num_samples / 16))):
            batch_data = X_train_full[b*16 : (b+1)*16]
            if len(batch_data) == 0:
                continue
            x = torch.tensor(batch_data, dtype=torch.float32)
            h = encoder(x) # [B, N, T, hidden_dim]
            # Slice at time step 23
            emb_list.append(h[:, :, 23, :].numpy())
            
    # Embeddings shape: [num_samples, num_reaches, 64]
    H = np.concatenate(emb_list, axis=0)
    print(f"  Reach embeddings extracted: shape={H.shape}")
    
    # Load targets from Parquets
    print("\nLoading benchmark targets...")
    p1 = pd.read_parquet('benchmark_p1_flood.parquet')
    p2 = pd.read_parquet('benchmark_p2_drought.parquet')
    p3 = pd.read_parquet('benchmark_p3_water_quality.parquet')
    p4 = pd.read_parquet('benchmark_p4_event_prediction.parquet')
    
    # Match the sample dates (monthly dates 23 to 227)
    # Target dates correspond to step 23 of each sample (which is start_date + 23 months)
    # Since df_pre contains the chronological sequences, the target dates of step 23 of each sample
    # are exactly the sorted unique dates in the dataset starting from month 23 (after the 24-month lag).
    unique_dates = sorted(df_pre['Date'].unique())
    target_dates = np.array(unique_dates[23:]) # 205 target dates (string YYYY-MM-DD)
    
    # --- P1 Flood Target ---
    print("Aligning P1 targets...")
    p1_filtered = p1[p1['Date'].isin(target_dates)].sort_values(['Date', 'Reach_ID']).reset_index(drop=True)
    y_p1 = p1_filtered['Flood_Score'].values.reshape(num_samples, num_reaches)
    
    # --- P2 Drought Target ---
    print("Aligning P2 targets...")
    p2_filtered = p2[p2['Date'].isin(target_dates)].sort_values(['Date', 'Subbasin_ID']).reset_index(drop=True)
    y_p2 = p2_filtered['Drought_Index'].values.reshape(num_samples, num_reaches)
    
    # --- P3 Water Quality Targets (NO3, NH4, Sediment) ---
    print("Aligning P3 targets...")
    p3_filtered = p3[p3['Date'].isin(target_dates)].sort_values(['Date', 'Reach_ID']).reset_index(drop=True)
    y_p3_no3 = p3_filtered['NO3'].values.reshape(num_samples, num_reaches)
    y_p3_nh4 = p3_filtered['NH4'].values.reshape(num_samples, num_reaches)
    y_p3_sed = p3_filtered['Sediment'].values.reshape(num_samples, num_reaches)
    
    # --- P4 Event Targets ---
    # Shifted target dates (exclude last 6 months since P4 event prediction only runs up to month 221)
    print("Aligning P4 targets...")
    p4_dates = target_dates[:199] # Only 199 samples have valid shifted targets
    p4_filtered = p4[p4['Date'].isin(p4_dates)].sort_values(['Date', 'Reach_ID']).reset_index(drop=True)
    y_p4_flood = p4_filtered['Flood_Event_6m'].values.reshape(199, num_reaches)
    y_p4_drought = p4_filtered['Drought_Event_6m'].values.reshape(199, num_reaches)

    # 4. Spatiotemporal Train-Test Split (Standard 80/20 temporal split)
    split_idx = int(num_samples * 0.8)
    split_idx_p4 = int(199 * 0.8)
    
    print("\n--- Training Benchmark heads ---")
    
    # P1 Flood evaluation
    res_p1 = train_and_eval_head(H[:split_idx], y_p1[:split_idx], H[split_idx:], y_p1[split_idx:], out_dim=1, epochs=15)
    print(f"P1 Flood Head: R2={res_p1['R2']:.4f}, MSE={res_p1['MSE']:.4f}")
    
    # P2 Drought evaluation
    res_p2 = train_and_eval_head(H[:split_idx], y_p2[:split_idx], H[split_idx:], y_p2[split_idx:], out_dim=1, epochs=15)
    print(f"P2 Drought Head: R2={res_p2['R2']:.4f}, MSE={res_p2['MSE']:.4f}")
    
    # P3 Water Quality (NO3, NH4, Sediment)
    res_p3_no3 = train_and_eval_head(H[:split_idx], y_p3_no3[:split_idx], H[split_idx:], y_p3_no3[split_idx:], out_dim=1, epochs=15)
    res_p3_nh4 = train_and_eval_head(H[:split_idx], y_p3_nh4[:split_idx], H[split_idx:], y_p3_nh4[split_idx:], out_dim=1, epochs=15)
    res_p3_sed = train_and_eval_head(H[:split_idx], y_p3_sed[:split_idx], H[split_idx:], y_p3_sed[split_idx:], out_dim=1, epochs=15)
    print(f"P3 NO3 Head: R2={res_p3_no3['R2']:.4f}, MSE={res_p3_no3['MSE']:.4f}")
    print(f"P3 NH4 Head: R2={res_p3_nh4['R2']:.4f}, MSE={res_p3_nh4['MSE']:.4f}")
    print(f"P3 Sediment Head: R2={res_p3_sed['R2']:.4f}, MSE={res_p3_sed['MSE']:.4f}")
    
    # P4 Events (6-Month ahead classification)
    H_p4 = H[:199]
    res_p4_flood = train_and_eval_head(H_p4[:split_idx_p4], y_p4_flood[:split_idx_p4], H_p4[split_idx_p4:], y_p4_flood[split_idx_p4:], out_dim=2, is_classifier=True, epochs=15)
    res_p4_drought = train_and_eval_head(H_p4[:split_idx_p4], y_p4_drought[:split_idx_p4], H_p4[split_idx_p4:], y_p4_drought[split_idx_p4:], out_dim=2, is_classifier=True, epochs=15)
    print(f"P4 6m Flood Event: Accuracy={res_p4_flood['Accuracy']:.4f}, F1-Score={res_p4_flood['F1-Score']:.4f}")
    print(f"P4 6m Drought Event: Accuracy={res_p4_drought['Accuracy']:.4f}, F1-Score={res_p4_drought['F1-Score']:.4f}")
    
    print("\n=== FINE-TUNING AND EVALUATION PROCESS COMPLETE ===")

if __name__ == '__main__':
    main()
