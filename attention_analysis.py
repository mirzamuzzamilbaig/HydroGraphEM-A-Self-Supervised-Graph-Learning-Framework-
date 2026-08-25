import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from train_pretrain import HST_PAGT_Encoder, compute_positional_encodings

def main():
    print("=== TASK 006: GNN Spatial & Temporal Attention Analysis ===")
    
    # 0. Setup directories
    os.makedirs('paper_figures', exist_ok=True)
    
    # 1. Load reaches metadata and pretraining dataset
    df_meta = pd.read_csv('metadata_reaches.csv')
    df_pre = pd.read_parquet('hydrographfm_pretrain_full.parquet')
    
    num_reaches = 483
    edges_fwd = []
    edges_bwd = []
    up_map = {}
    for idx, row in df_meta.iterrows():
        u = int(row['Reach_ID'])
        d = row['Downstream_ID']
        if pd.isna(row['Upstream_ID']):
            up_map[u] = []
        else:
            up_map[u] = [int(x) for x in str(row['Upstream_ID']).split(',') if x.strip()]
            
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
    X_train_full = X_scaled.reshape(num_reaches, num_samples, 24, -1)
    X_train_full = np.transpose(X_train_full, (1, 0, 2, 3))
    
    # Load Transformer SSL Encoder
    print("Loading Transformer Conv SSL Encoder for Attention Analysis...")
    encoder = HST_PAGT_Encoder(num_features, 64, edge_index, pos_enc, conv_type='transformer')
    encoder.load_state_dict(torch.load('hydrographfm_encoder_transformer.pt'))
    encoder.eval()
    
    # 2. Extract Temporal Self-Attention Matrix
    print("Extracting Temporal Self-Attention weights...")
    # X_train_full shape: [num_samples, num_reaches, 24, num_features]
    # We will pass a batch of samples through the model
    with torch.no_grad():
        x = torch.tensor(X_train_full[:16], dtype=torch.float32)
        B, N, T, D = x.shape
        pos_emb = encoder.pos_proj(encoder.pos_enc)
        
        # We need the inputs to the temporal transformer:
        spatial_outs = []
        E = encoder.edge_index.shape[1]
        edge_index_batched = torch.zeros((2, E * B), dtype=torch.long, device=x.device)
        for i in range(B):
            edge_index_batched[:, i*E : (i+1)*E] = encoder.edge_index + i * N
            
        for t in range(T):
            x_t = x[:, :, t, :]
            h_t = encoder.feat_proj(x_t) + pos_emb.unsqueeze(0)
            h_t_flat = h_t.view(B * N, -1)
            h_t_conv = encoder.conv(h_t_flat, edge_index_batched)
            spatial_outs.append(h_t_conv.view(B, N, -1))
            
        h_spatial = torch.stack(spatial_outs, dim=2) # [B, N, T, 64]
        h_spatial_flat = h_spatial.view(B * N, T, -1) # [B * N, T, 64]
        
        # Access temporal MultiheadAttention layer parameters
        self_attn = encoder.temporal_transformer.layers[0].self_attn
        in_proj_weight = self_attn.in_proj_weight
        in_proj_bias = self_attn.in_proj_bias
        
        # Project
        qkv = h_spatial_flat @ in_proj_weight.t() + in_proj_bias
        q, k, v = qkv.chunk(3, dim=-1) # Each of shape [B * N, T, 64]
        
        # Head splitting for 2 heads (d_head = 32)
        num_heads = 2
        d_head = 32
        
        # Reshape to [B*N, T, num_heads, d_head] -> transpose to [B*N, num_heads, T, d_head]
        q_heads = q.view(B * N, T, num_heads, d_head).transpose(1, 2)
        k_heads = k.view(B * N, T, num_heads, d_head).transpose(1, 2)
        
        # Attention scores: Softmax(Q K.T / sqrt(d_head))
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / np.sqrt(d_head)
        attn_weights = torch.softmax(scores, dim=-1) # [B * N, num_heads, T, T]
        
        # Average over batch, reaches and heads to get [T, T] global attention map
        mean_attn = attn_weights.mean(dim=(0, 1)).numpy() # Shape [24, 24]
        
    print(f"  Temporal attention weights extracted: shape={mean_attn.shape}")
    
    # 3. Analyze Spatial Attention Weights (TransformerConv)
    print("Extracting Spatial Attention weights from TransformerConv...")
    with torch.no_grad():
        # Get query and key linear layers from the TransformerConv
        conv = encoder.conv
        # In PyTorch Geometric, TransformerConv weights are:
        # conv.lin_query.weight, conv.lin_key.weight, conv.lin_edge.weight
        # Let's project node features at t=23 (last time step) for sample 0
        h_last = h_spatial[0, :, 23, :] # [N, 64]
        
        # Manual projection matching PyG TransformerConv attention coefficients
        # lin_query: [out_channels, in_channels]
        q_proj = conv.lin_query(h_last) # [N, 64]
        k_proj = conv.lin_key(h_last)   # [N, 64]
        
        # Compute scores for all edges
        u_idx = edge_index[0].numpy() # Source node indices
        v_idx = edge_index[1].numpy() # Target node indices
        
        q_v = q_proj[v_idx] # target nodes queries
        k_u = k_proj[u_idx] # source nodes keys
        
        # Dot product scores for each edge
        edge_scores = (q_v * k_u).sum(dim=-1).numpy() / np.sqrt(64)
        
        # Softmax over incoming edges for each target node
        spatial_attn_list = np.zeros(len(edge_scores))
        for target_node in range(num_reaches):
            incoming_edges = np.where(v_idx == target_node)[0]
            if len(incoming_edges) > 0:
                scores_t = edge_scores[incoming_edges]
                max_score = np.max(scores_t) # Stability trick
                exp_scores = np.exp(scores_t - max_score)
                softmax_scores = exp_scores / np.sum(exp_scores)
                spatial_attn_list[incoming_edges] = softmax_scores
                
    print(f"  Spatial attention weights computed: total edges={len(spatial_attn_list)}")
    
    # Analyze topology correlation (Strahler order flow)
    # Reconstruct Strahler stream order
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

    # Check if downstream nodes (larger stream order) pay higher attention to upstream nodes
    up_attn_vals = []
    down_attn_vals = []
    for edge_id, (u, v) in enumerate(zip(u_idx, v_idx)):
        # u is source, v is target (GNN conv message flows source -> target)
        u_order = orders_raw[u]
        v_order = orders_raw[v]
        
        if u_order < v_order:
            # message flowing downstream (headwaters to river channel)
            down_attn_vals.append(spatial_attn_list[edge_id])
        else:
            # message flowing upstream or sideways
            up_attn_vals.append(spatial_attn_list[edge_id])
            
    print(f"  Average attention downstream (tributary -> river): {np.mean(down_attn_vals):.4f}")
    print(f"  Average attention other directions: {np.mean(up_attn_vals):.4f}")
    
    # 4. Generate Attention Visualization Plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # A. Temporal Attention Heatmap
    sns.heatmap(mean_attn, cmap='plasma', ax=axes[0], xticklabels=5, yticklabels=5)
    axes[0].set_title('A. Temporal Self-Attention Weight Matrix', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Key Time Step (months)')
    axes[0].set_ylabel('Query Time Step (months)')
    axes[0].invert_yaxis()
    
    # B. Spatial Attention by Strahler Order Flow
    axes[1].grid(True, linestyle='--', alpha=0.5)
    sns.barplot(x=['Tributary -> River (Downstream)', 'Sideways/Upstream Flow'], 
                y=[np.mean(down_attn_vals), np.mean(up_attn_vals)], 
                palette='viridis', ax=axes[1])
    axes[1].set_title('B. Average Spatial Message Routing Attention', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Softmax Attention Coefficient')
    
    plt.suptitle('HydroGraphEM: Spatial-Temporal Attention Mechanism Diagnostics', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('paper_figures/attention_patterns.png', dpi=300)
    plt.close()
    
    print("=== TASK 006: Attention Analysis Completed Successfully ===")

if __name__ == '__main__':
    main()
