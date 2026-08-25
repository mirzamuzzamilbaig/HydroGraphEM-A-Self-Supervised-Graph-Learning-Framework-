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

# Encoder definition for loading
from train_pretrain import HST_PAGT_Encoder, compute_positional_encodings

def main():
    global has_umap
    print("=== Analyzing Reach Latent Embeddings (Task 004D) ===")
    
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
    
    # Compute positional encodings
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
    X_train = X_scaled.reshape(num_reaches, num_samples, 24, -1)
    X_train = np.transpose(X_train, (1, 0, 2, 3))
    
    # 2. Initialize GAT Encoder and load weights
    encoder = HST_PAGT_Encoder(len(feat_cols), 64, edge_index, pos_enc, conv_type='gat')
    if os.path.exists('hydrographfm_encoder_v1.pt'):
        encoder.load_state_dict(torch.load('hydrographfm_encoder_v1.pt'))
        print("  Encoder weights loaded successfully.")
    else:
        print("  WARNING: Encoder weights file not found! Using initialized weights.")
        
    encoder.eval()
    
    # 3. Extract embeddings
    print("Extracting reach embeddings...")
    emb_list = []
    with torch.no_grad():
        for b in range(int(np.ceil(num_samples / 16))):
            batch_data = X_train[b*16 : (b+1)*16]
            if len(batch_data) == 0:
                continue
            x = torch.tensor(batch_data, dtype=torch.float32)
            h = encoder(x) # [B, N, T, hidden_dim]
            emb_list.append(h[:, :, 23, :].numpy())
            
    # Embeddings shape: [num_samples, num_reaches, 64]
    H = np.concatenate(emb_list, axis=0)
    
    # Mean pool embeddings over all temporal samples to get a static representation for each reach node
    H_nodes = H.mean(axis=0) # Shape: [483, 64]
    print(f"  Aggregated reach node embeddings: shape={H_nodes.shape}")
    
    # 4. Read coloring variables
    # Color reaches by Strahler stream order (which we computed earlier)
    orders = pos_enc[:, 0].numpy()
    # Or color by average streamflow
    mean_flow = df_pre.groupby('Reach_ID')['FLOW_OUT'].mean().values
    
    # 5. Dimensionality Reduction Projections
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    z_pca = pca.fit_transform(H_nodes)
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_tsne = tsne.fit_transform(H_nodes)
    
    if has_umap:
        print("Computing UMAP projection...")
        try:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            z_umap = reducer.fit_transform(H_nodes)
        except Exception as e:
            print(f"Warning: UMAP computation failed: {e}")
            has_umap = False
            
    # 6. Plotting
    print("Generating embedding plots...")
    num_plots = 3 if has_umap else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
        
    # Normalize orders for legend mapping (convert back to actual integers)
    # Re-extract unnormalized Strahler orders from metadata for clearer labels
    # We can reconstruct them from pos_enc column 0 by undoing scaling
    # Or we can just calculate them from metadata directly:
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
        
    # PCA Plot
    sns.scatterplot(x=z_pca[:, 0], y=z_pca[:, 1], hue=orders_raw, palette='viridis', legend='full', ax=axes[0])
    axes[0].set_title('PCA Projection of Reach Embeddings')
    axes[0].set_xlabel('PC 1')
    axes[0].set_ylabel('PC 2')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # t-SNE Plot
    sns.scatterplot(x=z_tsne[:, 0], y=z_tsne[:, 1], hue=orders_raw, palette='viridis', legend='full', ax=axes[1])
    axes[1].set_title('t-SNE Projection of Reach Embeddings')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # UMAP Plot
    if has_umap:
        sns.scatterplot(x=z_umap[:, 0], y=z_umap[:, 1], hue=orders_raw, palette='viridis', legend='full', ax=axes[2])
        axes[2].set_title('UMAP Projection of Reach Embeddings')
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
        axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('HydroGraphEM: Reach Embedding Space Analysis (Colored by Strahler Stream Order)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('reach_embeddings_analysis.png', dpi=300, bbox_inches='tight')
    print("  reach_embeddings_analysis.png saved.")
    plt.close()

if __name__ == '__main__':
    main()
