import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import pickle
from train_pretrain import HST_PAGT_Encoder, compute_positional_encodings

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def main():
    print("=== MC-Dropout Evaluation ===")
    
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
    
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.scale_features_matrix(feat_cols, df_pre)
    
    num_samples = 205
    X_train_full = X_scaled.reshape(num_reaches, num_samples, 24, -1)
    X_train_full = np.transpose(X_train_full, (1, 0, 2, 3))
    
    unique_dates = sorted(df_pre['Date'].unique())
    target_dates = np.array(unique_dates[23:])
    
    train_dates_idx = np.where(target_dates < '2016-01-01')[0]
    val_dates_idx = np.where((target_dates >= '2016-01-01') & (target_dates < '2019-01-01'))[0]
    test_dates_idx = np.where(target_dates >= '2019-01-01')[0]
    
    p1 = pd.read_parquet('benchmark_p1_flood.parquet')
    p1_filtered = p1[p1['Date'].isin(target_dates)].sort_values(['Date', 'Reach_ID']).reset_index(drop=True)
    y_p1 = p1_filtered['Flood_Score'].values.reshape(num_samples, num_reaches)
    
    p3 = pd.read_parquet('benchmark_p3_water_quality.parquet')
    p3_filtered = p3[p3['Date'].isin(target_dates)].sort_values(['Date', 'Reach_ID']).reset_index(drop=True)
    y_p3_no3 = p3_filtered['NO3'].values.reshape(num_samples, num_reaches)

    encoder_gat = HST_PAGT_Encoder(len(feat_cols), 64, edge_index, pos_enc, conv_type='gat')
    encoder_gat.load_state_dict(torch.load('hydrographfm_encoder_v1.pt'))
    encoder_gat.eval()
    
    emb_list = []
    with torch.no_grad():
        for b in range(int(np.ceil(num_samples / 16))):
            batch_data = X_train_full[b*16 : (b+1)*16]
            if len(batch_data) == 0:
                continue
            x = torch.tensor(batch_data, dtype=torch.float32)
            h = encoder_gat(x)
            emb_list.append(h[:, :, 23, :].numpy())
    H = np.concatenate(emb_list, axis=0)
    
    def train_and_eval_mcdropout(H_all, y_all):
        X_tr = H_all[train_dates_idx].reshape(-1, H_all.shape[-1])
        y_tr = y_all[train_dates_idx].ravel()
        
        X_te = H_all[test_dates_idx].reshape(-1, H_all.shape[-1])
        y_te = y_all[test_dates_idx].ravel()
        
        mask_tr = ~np.isnan(y_tr)
        X_tr = X_tr[mask_tr]
        y_tr = y_tr[mask_tr]
        
        mask_te = ~np.isnan(y_te)
        X_te = X_te[mask_te]
        y_te = y_te[mask_te]
        
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        y_tr_scaled = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
        
        t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr_scaled, dtype=torch.float32)
        t_X_te = torch.tensor(X_te, dtype=torch.float32)
        
        head = MLPHead(H_all.shape[-1], 1)
        optimizer = optim.Adam(head.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        head.train()
        for ep in range(15):
            optimizer.zero_grad()
            outputs = head(t_X_tr).squeeze(-1)
            loss = criterion(outputs, t_y_tr)
            loss.backward()
            optimizer.step()
            
        head.eval()
        enable_dropout(head)
        
        preds_list = []
        with torch.no_grad():
            for _ in range(50):
                pred = head(t_X_te).squeeze(-1).numpy()
                pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
                preds_list.append(pred)
                
        preds = np.stack(preds_list, axis=0)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred
        
        picp = np.mean((y_te >= lower) & (y_te <= upper)) * 100
        y_range = y_te.max() - y_te.min()
        mpiw = np.mean(upper - lower) / (y_range + 1e-8)
        
        return picp, mpiw

    picp_p1, mpiw_p1 = train_and_eval_mcdropout(H, y_p1)
    print(f"P1 Flood: PICP={picp_p1:.2f}%, MPIW={mpiw_p1:.4f}")
    
    picp_p3, mpiw_p3 = train_and_eval_mcdropout(H, y_p3_no3)
    print(f"P3 NO3: PICP={picp_p3:.2f}%, MPIW={mpiw_p3:.4f}")
    
    with open('mcdropout_results.txt', 'w') as f:
        f.write(f"P1_PICP={picp_p1:.2f}\n")
        f.write(f"P1_MPIW={mpiw_p1:.4f}\n")
        f.write(f"P3_PICP={picp_p3:.2f}\n")
        f.write(f"P3_MPIW={mpiw_p3:.4f}\n")

if __name__ == '__main__':
    main()
