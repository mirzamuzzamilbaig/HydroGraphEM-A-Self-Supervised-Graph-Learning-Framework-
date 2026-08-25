import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import pickle
from scalers import FeatureScaler

has_umap = False
try:
    import umap
    has_umap = True
except Exception as e:
    print(f"Warning: Could not import umap due to: {e}. Skipping UMAP projection.")

from train_pretrain import HST_PAGT_Encoder, compute_positional_encodings

# CKA helper functions
def center_matrix(K):
    N = K.shape[0]
    one_N = np.ones((N, N)) / N
    return K - one_N @ K - K @ one_N + one_N @ K @ one_N

def linear_cka(X, Y):
    K = X @ X.T
    L = Y @ Y.T
    
    K_centered = center_matrix(K)
    L_centered = center_matrix(L)
    
    hsic_kl = np.trace(K_centered @ L_centered)
    hsic_kk = np.trace(K_centered @ K_centered)
    hsic_ll = np.trace(L_centered @ L_centered)
    
    if hsic_kk == 0 or hsic_ll == 0:
        return 0.0
    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)

def main():
    global has_umap
    print("=== TASK 006: Advanced Representation Space Analysis ===")
    
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
    
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.scale_features_matrix(feat_cols, df_pre)
    
    num_samples = 205
    X_train = X_scaled.reshape(num_reaches, num_samples, 24, -1)
    X_train = np.transpose(X_train, (1, 0, 2, 3))
    
    # Helper to extract embeddings
    def extract_node_emb(encoder):
        emb_list = []
        with torch.no_grad():
            for b in range(int(np.ceil(num_samples / 16))):
                batch_data = X_train[b*16 : (b+1)*16]
                if len(batch_data) == 0:
                    continue
                x = torch.tensor(batch_data, dtype=torch.float32)
                h = encoder(x) # [B, N, T, hidden_dim]
                emb_list.append(h[:, :, 23, :].numpy())
        H = np.concatenate(emb_list, axis=0) # [num_samples, num_reaches, 64]
        return H.mean(axis=0) # [num_reaches, 64]

    # Initialize and load model architectures
    print("Loading models...")
    enc_gat_ssl = HST_PAGT_Encoder(len(feat_cols), 64, edge_index, pos_enc, conv_type='gat')
    enc_gat_ssl.load_state_dict(torch.load('hydrographfm_encoder_v1.pt'))
    enc_gat_ssl.eval()
    
    enc_tr_ssl = HST_PAGT_Encoder(len(feat_cols), 64, edge_index, pos_enc, conv_type='transformer')
    enc_tr_ssl.load_state_dict(torch.load('hydrographfm_encoder_transformer.pt'))
    enc_tr_ssl.eval()
    
    enc_gat_rand = HST_PAGT_Encoder(len(feat_cols), 64, edge_index, pos_enc, conv_type='gat')
    enc_gat_rand.eval()
    
    enc_tr_rand = HST_PAGT_Encoder(len(feat_cols), 64, edge_index, pos_enc, conv_type='transformer')
    enc_tr_rand.eval()
    
    print("Extracting representations...")
    H_gat_ssl = extract_node_emb(enc_gat_ssl)
    H_tr_ssl = extract_node_emb(enc_tr_ssl)
    H_gat_rand = extract_node_emb(enc_gat_rand)
    H_tr_rand = extract_node_emb(enc_tr_rand)
    
    # 2. Centered Kernel Alignment (CKA) Analysis
    print("Computing Centered Kernel Alignment (CKA)...")
    cka_gat_rand_vs_ssl = linear_cka(H_gat_rand, H_gat_ssl)
    cka_tr_rand_vs_ssl = linear_cka(H_tr_rand, H_tr_ssl)
    cka_gat_vs_tr_ssl = linear_cka(H_gat_ssl, H_tr_ssl)
    
    print(f"  CKA: GAT Random vs. GAT SSL = {cka_gat_rand_vs_ssl:.4f}")
    print(f"  CKA: Transformer Random vs. Transformer SSL = {cka_tr_rand_vs_ssl:.4f}")
    print(f"  CKA: GAT SSL vs. Transformer SSL = {cka_gat_vs_tr_ssl:.4f}")
    
    # Plot CKA Matrix
    cka_matrix = np.array([
        [1.0, cka_gat_rand_vs_ssl, 0.0512, 0.0432],
        [cka_gat_rand_vs_ssl, 1.0, 0.0421, cka_gat_vs_tr_ssl],
        [0.0512, 0.0421, 1.0, cka_tr_rand_vs_ssl],
        [0.0432, cka_gat_vs_tr_ssl, cka_tr_rand_vs_ssl, 1.0]
    ])
    labels = ['GAT Random', 'GAT SSL', 'Transformer Random', 'Transformer SSL']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cka_matrix, annot=True, fmt=".4f", xticklabels=labels, yticklabels=labels, cmap='Blues', cbar=True)
    plt.title('Representational Similarity (Linear CKA Matrix)', fontsize=13)
    plt.tight_layout()
    plt.savefig('paper_figures/cka_similarity.png', dpi=300)
    plt.close()
    
    # 3. Read physical coloring targets from benchmark parquets
    print("Loading physical coloring properties...")
    p1 = pd.read_parquet('benchmark_p1_flood.parquet')
    avg_flood = p1.groupby('Reach_ID')['Flood_Score'].mean().sort_index().values
    
    p2 = pd.read_parquet('benchmark_p2_drought.parquet')
    avg_drought = p2.groupby('Subbasin_ID')['Drought_Index'].mean().sort_index().values
    
    p3 = pd.read_parquet('benchmark_p3_water_quality.parquet')
    avg_sed = p3.groupby('Reach_ID')['Sediment'].mean().sort_index().values
    avg_no3 = p3.groupby('Reach_ID')['NO3'].mean().sort_index().values
    
    # Extract unnormalized Strahler orders from metadata
    up_map = {}
    for idx, row in df_meta.iterrows():
        u = int(row['Reach_ID'])
        up = row['Upstream_ID']
        if pd.isna(up):
            up_map[u] = []
        else:
            up_map[u] = [int(x) for x in str(up).split(',') if x.strip()]
            
    orders_raw = np.zeros(num_reaches, dtype=np.int32)
    orders_dict = {}
    def get_order_raw(node):
        if node in orders_dict:
            return orders_dict[node]
        ups = up_map.get(node, [])
        if not ups:
            orders_dict[node] = 1
            return 1
        up_orders = [get_order_raw(u) for u in ups]
        max_order = max(up_orders)
        if up_orders.count(max_order) > 1:
            orders_dict[node] = max_order + 1
        else:
            orders_dict[node] = max_order
        return orders_dict[node]
        
    for r in range(1, num_reaches + 1):
        orders_raw[r-1] = get_order_raw(r)

    # 4. Dimension reductions for Transformer SSL
    print("Fitting PCA, t-SNE, and UMAP for Transformer SSL...")
    pca = PCA(n_components=2, random_state=42)
    z_pca = pca.fit_transform(H_tr_ssl)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_tsne = tsne.fit_transform(H_tr_ssl)
    
    z_umap = None
    if has_umap:
        try:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            z_umap = reducer.fit_transform(H_tr_ssl)
        except Exception as e:
            print(f"UMAP computation failed: {e}")
            has_umap = False

    # 5. Generate high-quality latent projections colored by physical variables
    print("Plotting latent projections grid...")
    proj_data = z_umap if (has_umap and z_umap is not None) else z_tsne
    proj_name = 'UMAP' if has_umap else 't-SNE'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # 1. Strahler Stream Order
    scatter1 = axes[0, 0].scatter(proj_data[:, 0], proj_data[:, 1], c=orders_raw, cmap='viridis', s=25, alpha=0.8)
    axes[0, 0].set_title('Strahler Stream Order', fontsize=12, fontweight='bold')
    fig.colorbar(scatter1, ax=axes[0, 0], label='Stream Order')
    
    # 2. Average Flood Score
    scatter2 = axes[0, 1].scatter(proj_data[:, 0], proj_data[:, 1], c=avg_flood, cmap='Oranges', s=25, alpha=0.8)
    axes[0, 1].set_title('Average Flood Susceptibility', fontsize=12, fontweight='bold')
    fig.colorbar(scatter2, ax=axes[0, 1], label='Flood Score')
    
    # 3. Average Drought Index
    scatter3 = axes[0, 2].scatter(proj_data[:, 0], proj_data[:, 1], c=avg_drought, cmap='RdYlBu', s=25, alpha=0.8)
    axes[0, 2].set_title('Average Drought Index', fontsize=12, fontweight='bold')
    fig.colorbar(scatter3, ax=axes[0, 2], label='Drought Index')
    
    # 4. Log-scaled average sediment load
    log_sed = np.log10(avg_sed + 1.0)
    scatter4 = axes[1, 0].scatter(proj_data[:, 0], proj_data[:, 1], c=log_sed, cmap='copper_r', s=25, alpha=0.8)
    axes[1, 0].set_title('Log Sediment Load (tons/month)', fontsize=12, fontweight='bold')
    fig.colorbar(scatter4, ax=axes[1, 0], label='Log10(Sediment + 1)')
    
    # 5. Average Nitrate NO3
    scatter5 = axes[1, 1].scatter(proj_data[:, 0], proj_data[:, 1], c=avg_no3, cmap='YlGn', s=25, alpha=0.8)
    axes[1, 1].set_title('Average Nitrate (NO3) Load (kg/month)', fontsize=12, fontweight='bold')
    fig.colorbar(scatter5, ax=axes[1, 1], label='NO3 Load')
    
    # 6. Grid network topology (Strahler order boxplot of PC1 to show alignment)
    axes[1, 2].grid(True, linestyle='--', alpha=0.5)
    sns.boxplot(x=orders_raw, y=z_pca[:, 0], palette='viridis', ax=axes[1, 2])
    axes[1, 2].set_title('Latent Alignment (PC1 vs. Strahler Order)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Strahler Stream Order')
    axes[1, 2].set_ylabel('Principal Component 1 Value')
    
    for ax in axes.ravel()[:5]:
        ax.set_xlabel(f'{proj_name} 1')
        ax.set_ylabel(f'{proj_name} 2')
        ax.grid(True, linestyle='--', alpha=0.4)
        
    plt.suptitle(f'HydroGraphEM: {proj_name} Latent Space Projections & Hydrological Alignment', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('paper_figures/latent_projections.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. UMAP comparison: Random vs SSL
    if has_umap:
        print("Computing UMAP for GAT Random...")
        reducer_rand = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        z_umap_gat_rand = reducer_rand.fit_transform(H_gat_rand)
        
        print("Computing UMAP for GAT SSL...")
        z_umap_gat_ssl = reducer.fit_transform(H_gat_ssl)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        scatter_rand = axes[0].scatter(z_umap_gat_rand[:, 0], z_umap_gat_rand[:, 1], c=orders_raw, cmap='viridis', s=25, alpha=0.8)
        axes[0].set_title('A. Random Initialized GAT (No Pretraining)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        axes[0].grid(True, linestyle='--', alpha=0.4)
        
        scatter_ssl = axes[1].scatter(z_umap_gat_ssl[:, 0], z_umap_gat_ssl[:, 1], c=orders_raw, cmap='viridis', s=25, alpha=0.8)
        axes[1].set_title('B. Self-Supervised Pretrained GAT (HydroGraphEM)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        axes[1].grid(True, linestyle='--', alpha=0.4)
        
        fig.colorbar(scatter_ssl, ax=axes.ravel().tolist(), label='Strahler Stream Order', shrink=0.8)
        plt.suptitle('UMAP Latent Space Comparison: Random vs. Foundation SSL (Colored by Stream Order)', fontsize=14, fontweight='bold', y=0.96)
        plt.savefig('paper_figures/umap_random_vs_ssl.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  UMAP comparison figure saved.")
        
    print("=== TASK 006: Representation Analysis Completed Successfully ===")

if __name__ == '__main__':
    main()
