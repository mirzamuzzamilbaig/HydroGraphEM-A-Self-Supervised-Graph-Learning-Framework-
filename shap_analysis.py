import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from train_pretrain import HST_PAGT_Encoder, compute_positional_encodings
from benchmark_generalization import MLPHead

def main():
    print("=== TASK 006: Global Permutation Importance & Gradient Saliency ===")
    
    # 0. Setup directories
    os.makedirs('paper_figures', exist_ok=True)
    
    # 1. Load reaches metadata and pretraining dataset
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
    num_features = len(feat_cols)
    
    # Load feature scaler
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.scale_features_matrix(feat_cols, df_pre)
    
    num_samples = 205
    # Reshape to [num_samples, num_reaches, 24, num_features]
    X_train_full = X_scaled.reshape(num_reaches, num_samples, 24, -1)
    X_train_full = np.transpose(X_train_full, (1, 0, 2, 3))
    
    unique_dates = sorted(df_pre['Date'].unique())
    target_dates = np.array(unique_dates[23:])
    
    train_dates_idx = np.where(target_dates < '2016-01-01')[0]
    test_dates_idx = np.where(target_dates >= '2019-01-01')[0]
    
    # Load targets
    p1 = pd.read_parquet('benchmark_p1_flood.parquet')
    y_p1 = p1[p1['Date'].isin(target_dates)].sort_values(['Date', 'Reach_ID'])['Flood_Score'].values.reshape(num_samples, num_reaches)
    
    p2 = pd.read_parquet('benchmark_p2_drought.parquet')
    y_p2 = p2[p2['Date'].isin(target_dates)].sort_values(['Date', 'Subbasin_ID'])['Drought_Index'].values.reshape(num_samples, num_reaches)
    
    p3 = pd.read_parquet('benchmark_p3_water_quality.parquet')
    y_p3_no3 = p3[p3['Date'].isin(target_dates)].sort_values(['Date', 'Reach_ID'])['NO3'].values.reshape(num_samples, num_reaches)
    
    # Load Transformer SSL Encoder
    print("Loading Transformer Conv SSL Encoder...")
    encoder = HST_PAGT_Encoder(num_features, 64, edge_index, pos_enc, conv_type='transformer')
    encoder.load_state_dict(torch.load('hydrographfm_encoder_transformer.pt'))
    encoder.eval()
    
    # Helper to extract embeddings
    def get_embeddings(X_matrix):
        emb_list = []
        with torch.no_grad():
            for b in range(int(np.ceil(len(X_matrix) / 16))):
                batch_data = X_matrix[b*16 : (b+1)*16]
                if len(batch_data) == 0:
                    continue
                x = torch.tensor(batch_data, dtype=torch.float32)
                h = encoder(x) # [B, N, T, 64]
                emb_list.append(h[:, :, 23, :].numpy())
        return np.concatenate(emb_list, axis=0) # [B, N, 64]

    # Pre-extract base embeddings
    H_all = get_embeddings(X_train_full)
    H_tr = H_all[train_dates_idx]
    H_te = H_all[test_dates_idx]
    
    # Let's train downstream heads for Flood, Drought, and Water Quality (NO3)
    def train_downstream_head(H_train, y_train, H_test, y_test):
        # Flatten
        X_tr_flat = H_train.reshape(-1, H_train.shape[-1])
        y_tr_flat = y_train.ravel()
        X_te_flat = H_test.reshape(-1, H_test.shape[-1])
        y_te_flat = y_test.ravel()
        
        # Remove NaNs
        mask_tr = ~np.isnan(y_tr_flat)
        X_tr_flat = X_tr_flat[mask_tr]
        y_tr_flat = y_tr_flat[mask_tr]
        
        mask_te = ~np.isnan(y_te_flat)
        X_te_flat = X_te_flat[mask_te]
        y_te_flat = y_te_flat[mask_te]
        
        y_scaler = StandardScaler()
        y_tr_flat_scaled = y_scaler.fit_transform(y_tr_flat.reshape(-1, 1)).ravel()
        
        t_X_tr = torch.tensor(X_tr_flat, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr_flat_scaled, dtype=torch.float32)
        
        head = MLPHead(64, 1, is_classifier=False)
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
        return head, y_scaler

    print("Training downstream heads...")
    head_flood, scaler_flood = train_downstream_head(H_tr, y_p1[train_dates_idx], H_te, y_p1[test_dates_idx])
    head_drought, scaler_drought = train_downstream_head(H_tr, y_p2[train_dates_idx], H_te, y_p2[test_dates_idx])
    head_no3, scaler_no3 = train_downstream_head(H_tr, y_p3_no3[train_dates_idx], H_te, y_p3_no3[test_dates_idx])
    
    # Baseline Test Metrics
    def eval_model(head, y_scaler, H_test, y_test):
        X_te_flat = H_test.reshape(-1, H_test.shape[-1])
        y_te_flat = y_test.ravel()
        mask_te = ~np.isnan(y_te_flat)
        X_te_flat = X_te_flat[mask_te]
        y_te_flat = y_te_flat[mask_te]
        
        with torch.no_grad():
            t_X_te = torch.tensor(X_te_flat, dtype=torch.float32)
            pred = head(t_X_te).squeeze(-1).numpy()
            pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
            
        return mean_squared_error(y_te_flat, pred)

    base_mse_flood = eval_model(head_flood, scaler_flood, H_te, y_p1[test_dates_idx])
    base_mse_drought = eval_model(head_drought, scaler_drought, H_te, y_p2[test_dates_idx])
    base_mse_no3 = eval_model(head_no3, scaler_no3, H_te, y_p3_no3[test_dates_idx])
    
    # 2. Compute Permutation Feature Importance
    print("Computing Permutation Feature Importance...")
    importance_flood = {}
    importance_drought = {}
    importance_no3 = {}
    
    X_test_baseline = X_train_full[test_dates_idx]
    
    for f_idx, feat in enumerate(feat_cols):
        # Create a shuffled test set
        X_shuffled = X_test_baseline.copy()
        
        # Shuffle feature f_idx across dates and reaches (temporal/spatial dimensions)
        flat_feat_vals = X_shuffled[:, :, :, f_idx].copy().ravel()
        np.random.shuffle(flat_feat_vals)
        X_shuffled[:, :, :, f_idx] = flat_feat_vals.reshape(X_shuffled.shape[0], X_shuffled.shape[1], X_shuffled.shape[2])
        
        # Get shuffled embeddings
        H_shuffled = get_embeddings(X_shuffled)
        
        # Evaluate
        mse_f = eval_model(head_flood, scaler_flood, H_shuffled, y_p1[test_dates_idx])
        mse_dr = eval_model(head_drought, scaler_drought, H_shuffled, y_p2[test_dates_idx])
        mse_n = eval_model(head_no3, scaler_no3, H_shuffled, y_p3_no3[test_dates_idx])
        
        importance_flood[feat] = max(0.0, mse_f - base_mse_flood)
        importance_drought[feat] = max(0.0, mse_dr - base_mse_drought)
        importance_no3[feat] = max(0.0, mse_n - base_mse_no3)
        
    print("  Permutation analysis completed.")
    
    # 3. Compute Gradient-Based Saliency
    print("Computing Gradient Saliency...")
    # Select a representative subset of test samples to save compute time
    t_X_samples = torch.tensor(X_test_baseline[:12], dtype=torch.float32, requires_grad=True)
    
    # Pass through model
    h_out = encoder(t_X_samples)
    h_last = h_out[:, :, 23, :] # Get last sequence representation: [B, N, 64]
    
    # Get saliency for Flood
    pred_fl = head_flood(h_last.reshape(-1, 64))
    encoder.zero_grad()
    head_flood.zero_grad()
    pred_fl.sum().backward(retain_graph=True)
    saliency_flood = t_X_samples.grad.abs().mean(dim=(0, 1, 2)).numpy()
    
    # Reset grad and compute for Drought
    t_X_samples.grad.zero_()
    pred_dr = head_drought(h_last.reshape(-1, 64))
    encoder.zero_grad()
    head_drought.zero_grad()
    pred_dr.sum().backward(retain_graph=True)
    saliency_drought = t_X_samples.grad.abs().mean(dim=(0, 1, 2)).numpy()
    
    # Reset grad and compute for NO3
    t_X_samples.grad.zero_()
    pred_n = head_no3(h_last.reshape(-1, 64))
    encoder.zero_grad()
    head_no3.zero_grad()
    pred_n.sum().backward()
    saliency_no3 = t_X_samples.grad.abs().mean(dim=(0, 1, 2)).numpy()
    
    # Combine results into DataFrame for plotting
    df_imp = pd.DataFrame({
        'Feature': feat_cols,
        'Flood_Permutation': [importance_flood[f] for f in feat_cols],
        'Drought_Permutation': [importance_drought[f] for f in feat_cols],
        'NO3_Permutation': [importance_no3[f] for f in feat_cols],
        'Flood_Saliency': saliency_flood,
        'Drought_Saliency': saliency_drought,
        'NO3_Saliency': saliency_no3
    })
    
    # Normalize importances for visualization
    for col in df_imp.columns[1:]:
        s = df_imp[col].sum()
        if s > 0:
            df_imp[col] = df_imp[col] / s
            
    df_imp.to_csv('paper_figures/feature_importance_metrics.csv', index=False)
    print("  Importances saved to csv.")
    
    # 4. Generate high-quality explainability bar charts
    fig, axes = plt.subplots(3, 1, figsize=(11, 14))
    
    # Sort and plot Flood
    df_fl = df_imp.sort_values('Flood_Permutation', ascending=False).head(10)
    sns.barplot(data=df_fl, x='Flood_Permutation', y='Feature', palette='Oranges_r', ax=axes[0])
    axes[0].set_title('A. Flood Susceptibility (P1) Variable Importance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Relative Importance (Normalized MSE Increase)')
    axes[0].set_ylabel('')
    
    # Sort and plot Drought
    df_dr = df_imp.sort_values('Drought_Permutation', ascending=False).head(10)
    sns.barplot(data=df_dr, x='Drought_Permutation', y='Feature', palette='Blues_r', ax=axes[1])
    axes[1].set_title('B. Regional Drought Index (P2) Variable Importance', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Relative Importance (Normalized MSE Increase)')
    axes[1].set_ylabel('')
    
    # Sort and plot Nitrate (NO3)
    df_n = df_imp.sort_values('NO3_Permutation', ascending=False).head(10)
    sns.barplot(data=df_n, x='NO3_Permutation', y='Feature', palette='Greens_r', ax=axes[2])
    axes[2].set_title('C. Nitrate (NO3) Loading (P3) Variable Importance', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Relative Importance (Normalized MSE Increase)')
    axes[2].set_ylabel('')
    
    plt.suptitle('HydroGraphEM: Physical Hydrological Knowledge Saliency Map', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig('paper_figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. SHAP notebook compatibility test
    # We create shap_flood.png and others using KernelExplainer on representative subset
    import shap
    print("Computing downstream SHAP summary plot...")
    
    # Take background and test samples of embeddings
    bg_embs = H_tr.reshape(-1, 64)[:100]
    test_embs = H_te.reshape(-1, 64)[:50]
    
    explainer = shap.KernelExplainer(lambda x: head_flood(torch.tensor(x, dtype=torch.float32)).detach().numpy(), bg_embs)
    shap_values = explainer.shap_values(test_embs)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_embs, show=False)
    plt.title('SHAP Summary Plot for Flood Susceptibility (Embedding Space)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper_figures/shap_flood.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  shap_flood.png saved.")
    
    print("=== TASK 006: Explainability Analysis Completed Successfully ===")

if __name__ == '__main__':
    main()
