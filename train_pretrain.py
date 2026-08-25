import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, TransformerConv, GCNConv
import os
import pickle
from scalers import FeatureScaler

# Helper to normalize series
def scale_array(arr):
    std = arr.std()
    mean = arr.mean()
    if std == 0:
        return arr - mean
    return (arr - mean) / (std + 1e-8)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # radius of Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def compute_positional_encodings(df_meta, df_pre):
    print("Computing static Reach Positional Encodings (Geometric + Hydrological)...")
    num_reaches = 483
    
    # 1. Parse Upstream and Downstream connections
    up_map = {}
    down_map = {}
    area_map = {}
    lat_map = {}
    lon_map = {}
    
    for idx, row in df_meta.iterrows():
        u = int(row['Reach_ID'])
        d = row['Downstream_ID']
        up = row['Upstream_ID']
        down_map[u] = int(d) if not pd.isna(d) and int(d) != 0 else None
        area_map[u] = float(row['Area'])
        lat_map[u] = float(row['latitude'])
        lon_map[u] = float(row['longitude'])
        if pd.isna(up):
            up_map[u] = []
        else:
            up_map[u] = [int(x) for x in str(up).split(',') if x.strip()]

    # 2. Strahler Stream Order
    orders = {}
    def get_order(node):
        if node in orders:
            return orders[node]
        ups = up_map.get(node, [])
        if not ups:
            orders[node] = 1
            return 1
        up_orders = [get_order(u) for u in ups]
        max_order = max(up_orders)
        if up_orders.count(max_order) > 1:
            orders[node] = max_order + 1
        else:
            orders[node] = max_order
        return orders[node]
    
    for r in range(1, num_reaches + 1):
        get_order(r)

    # 3. Distance to Outlet (hops)
    dist_to_outlet = {}
    for r in range(1, num_reaches + 1):
        curr = r
        hops = 0
        visited = set()
        while curr in down_map and down_map[curr] is not None:
            curr = down_map[curr]
            hops += 1
            if curr in visited:
                break
            visited.add(curr)
        dist_to_outlet[r] = hops

    # 4. Coastal Outlets & Distance to Coast
    outlets = [r for r, d in down_map.items() if d is None]
    dist_to_coast = {}
    for r in range(1, num_reaches + 1):
        lat_r, lon_r = lat_map[r], lon_map[r]
        min_dist = float('inf')
        for out in outlets:
            d = haversine(lat_r, lon_r, lat_map[out], lon_map[out])
            if d < min_dist:
                min_dist = d
        dist_to_coast[r] = min_dist

    # 5. Accumulated Upstream Area
    acc_area = {}
    def get_acc_area(node):
        if node in acc_area:
            return acc_area[node]
        ups = up_map.get(node, [])
        tot = area_map[node] + sum(get_acc_area(u) for u in ups)
        acc_area[node] = tot
        return tot
    
    for r in range(1, num_reaches + 1):
        get_acc_area(r)

    # 6. Hydrological position (Means of FLOW_OUT, WYLDmm, SURQmm)
    # df_pre is sorted by Reach_ID and Date
    mean_stats = df_pre.groupby('Reach_ID')[['FLOW_OUT', 'WYLDmm', 'SURQmm']].mean().reset_index()
    flow_mean = mean_stats.set_index('Reach_ID')['FLOW_OUT'].to_dict()
    wyld_mean = mean_stats.set_index('Reach_ID')['WYLDmm'].to_dict()
    surq_mean = mean_stats.set_index('Reach_ID')['SURQmm'].to_dict()

    # Compile positional encodings matrix
    pos_enc = np.zeros((num_reaches, 7), dtype=np.float32)
    for r in range(1, num_reaches + 1):
        idx = r - 1
        pos_enc[idx, 0] = orders[r]
        pos_enc[idx, 1] = dist_to_outlet[r]
        pos_enc[idx, 2] = dist_to_coast[r]
        pos_enc[idx, 3] = acc_area[r]
        pos_enc[idx, 4] = flow_mean.get(r, 0.0)
        pos_enc[idx, 5] = wyld_mean.get(r, 0.0)
        pos_enc[idx, 6] = surq_mean.get(r, 0.0)

    # Normalize each column
    for col in range(7):
        pos_enc[:, col] = scale_array(pos_enc[:, col])
        
    print(f"  Positional encodings computed: shape={pos_enc.shape}")
    return torch.tensor(pos_enc, dtype=torch.float32)

class HST_PAGT_Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_index, pos_enc, conv_type='gat'):
        super().__init__()
        self.feat_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_proj = nn.Linear(7, hidden_dim)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('pos_enc', pos_enc)
        self.conv_type = conv_type
        
        if conv_type == 'gat':
            self.conv = GATConv(hidden_dim, hidden_dim, heads=1)
        elif conv_type == 'gcn':
            self.conv = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv = TransformerConv(hidden_dim, hidden_dim, heads=1)
            
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim*2, batch_first=True),
            num_layers=1
        )
        
    def forward(self, x):
        # x shape: [B, N, T, D] where N = 483, T = 24
        B, N, T, D = x.shape
        
        # Project positional encodings
        pos_emb = self.pos_proj(self.pos_enc) # [N, hidden_dim]
        
        # Build batched edge index for PyG
        # edge_index shape: [2, E]. Batched shape: [2, E * B]
        E = self.edge_index.shape[1]
        edge_index_batched = torch.zeros((2, E * B), dtype=torch.long, device=x.device)
        for i in range(B):
            edge_index_batched[:, i*E : (i+1)*E] = self.edge_index + i * N

        # We will loop over time steps to apply spatial convolution to the batched graph
        spatial_outs = []
        for t in range(T):
            x_t = x[:, :, t, :] # [B, N, D]
            # Project features and add positional embeddings
            h_t = self.feat_proj(x_t) + pos_emb.unsqueeze(0) # [B, N, hidden_dim]
            h_t_flat = h_t.view(B * N, -1) # [B * N, hidden_dim]
            # GNN conv
            h_t_conv = self.conv(h_t_flat, edge_index_batched) # [B * N, hidden_dim]
            h_t_spatial = h_t_conv.view(B, N, -1) # [B, N, hidden_dim]
            spatial_outs.append(h_t_spatial)

        # Stack spatial features: [B, N, T, hidden_dim]
        h_spatial = torch.stack(spatial_outs, dim=2)

        # Reshape for Temporal Transformer: [B * N, T, hidden_dim]
        h_spatial_flat = h_spatial.view(B * N, T, -1)
        h_temporal_flat = self.temporal_transformer(h_spatial_flat) # [B * N, T, hidden_dim]
        
        # Reshape back to [B, N, T, hidden_dim]
        h_out = h_temporal_flat.view(B, N, T, -1)
        return h_out

    # Forward for prefix sequence of length T_prefix (Stage B future prediction)
    def forward_prefix(self, x_prefix):
        B, N, T_pref, D = x_prefix.shape
        pos_emb = self.pos_proj(self.pos_enc)
        E = self.edge_index.shape[1]
        edge_index_batched = torch.zeros((2, E * B), dtype=torch.long, device=x_prefix.device)
        for i in range(B):
            edge_index_batched[:, i*E : (i+1)*E] = self.edge_index + i * N

        spatial_outs = []
        for t in range(T_pref):
            x_t = x_prefix[:, :, t, :]
            h_t = self.feat_proj(x_t) + pos_emb.unsqueeze(0)
            h_t_flat = h_t.view(B * N, -1)
            h_t_conv = self.conv(h_t_flat, edge_index_batched)
            spatial_outs.append(h_t_conv.view(B, N, -1))

        h_spatial = torch.stack(spatial_outs, dim=2)
        h_spatial_flat = h_spatial.view(B * N, T_pref, -1)
        h_temporal_flat = self.temporal_transformer(h_spatial_flat)
        return h_temporal_flat.view(B, N, T_pref, -1)


class HST_PAGT_SSL_Model(nn.Module):
    def __init__(self, encoder, hidden_dim, swat_dim, clim_dim, eo_dim):
        super().__init__()
        self.encoder = encoder
        
        # Task Decoders
        self.recon_head = nn.Linear(hidden_dim, 4)  # FLOW_OUT, NO3, Sediment, GW_Qmm
        self.future_head = nn.Linear(hidden_dim, 3) # Horizons t+1, t+3, t+6
        self.impute_head = nn.Linear(hidden_dim, 5) # FLOW_OUT, SURQmm, GW_Qmm, SWmm, WYLDmm
        
        # Link Prediction project head
        self.link_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-Modal Alignment projection heads
        self.proj_swat = nn.Linear(swat_dim, hidden_dim)
        self.proj_clim = nn.Linear(clim_dim, hidden_dim)
        self.proj_eo = nn.Linear(eo_dim, hidden_dim)
        
        # Alignment node projector
        self.proj_node = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.encoder(x)


def compute_alignment_loss(h_out, x_swat, x_clim, x_eo, model, swat_idx, clim_idx, eo_idx):
    B, N, T, _ = h_out.shape
    # Flatten everything to shape [B * N * T, hidden_dim]
    h_flat = model.proj_node(h_out.reshape(B * N * T, -1))
    
    # Project modalities
    swat_flat = x_swat.reshape(B * N * T, -1)
    clim_flat = x_clim.reshape(B * N * T, -1)
    eo_flat = x_eo.reshape(B * N * T, -1)
    
    z_swat = model.proj_swat(swat_flat)
    z_clim = model.proj_clim(clim_flat)
    z_eo = model.proj_eo(eo_flat)
    
    # Standardize/Normalize embeddings
    h_flat = h_flat / (h_flat.norm(dim=-1, keepdim=True) + 1e-8)
    z_swat = z_swat / (z_swat.norm(dim=-1, keepdim=True) + 1e-8)
    z_clim = z_clim / (z_clim.norm(dim=-1, keepdim=True) + 1e-8)
    z_eo = z_eo / (z_eo.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Compute InfoNCE contrastive alignment loss over a subset to save memory/speed on CPU
    # Select a subset of size 512 randomly
    indices = torch.randperm(B * N * T)[:512]
    h_s = h_flat[indices]
    zs_swat = z_swat[indices]
    zs_clim = z_clim[indices]
    zs_eo = z_eo[indices]
    
    tau = 0.07
    # SWAT to EO alignment
    sim_swat_eo = torch.matmul(h_s, zs_eo.t()) / tau
    labels = torch.arange(len(indices), device=h_out.device)
    loss1 = nn.CrossEntropyLoss()(sim_swat_eo, labels)
    
    # SWAT to Clim alignment
    sim_swat_clim = torch.matmul(h_s, zs_clim.t()) / tau
    loss2 = nn.CrossEntropyLoss()(sim_swat_clim, labels)
    
    return loss1 + loss2


def run_pretraining(X_train, edge_index, pos_enc, feat_cols, conv_type='gat'):
    print(f"\n--- Initiating Pretraining Loop for Model ({conv_type.upper()}) ---")
    
    num_samples = X_train.shape[0]
    in_dim = X_train.shape[3]
    hidden_dim = 64
    batch_size = 16
    
    # Setup programmatical feature indices
    flow_out_idx = feat_cols.index('FLOW_OUT')
    
    recon_vars = ['FLOW_OUT', 'NO3', 'Sediment', 'GW_Qmm']
    recon_indices = [feat_cols.index(v) for v in recon_vars]
    
    impute_vars = ['FLOW_OUT', 'SURQmm', 'GW_Qmm', 'SWmm', 'WYLDmm']
    impute_indices = [feat_cols.index(v) for v in impute_vars]
    
    swat_vars = ['FLOW_IN', 'FLOW_OUT', 'Channel_Evap', 'Transmission_Loss', 'Sediment', 'TN', 'TP', 'NO3', 'NH4', 'Temp_Water', 'SURQmm', 'SWmm', 'GW_Qmm', 'WYLDmm', 'ETmm']
    clim_vars = ['Rainfall', 'Temp_Mean', 'Soil_Moist', 'Evaporation']
    eo_vars = ['NDVI', 'NDWI', 'MNDWI', 'EVI', 'JRC_Occurrence', 'JRC_Seasonality', 'JRC_Recurrence', 'Cropland_pct', 'Wetland_pct', 'Builtup_pct', 'Forest_pct', 'Water_pct']
    
    swat_indices = [feat_cols.index(v) for v in swat_vars if v in feat_cols]
    clim_indices = [feat_cols.index(v) for v in clim_vars if v in feat_cols]
    eo_indices = [feat_cols.index(v) for v in eo_vars if v in feat_cols]

    # Initialize model
    encoder = HST_PAGT_Encoder(in_dim, hidden_dim, edge_index, pos_enc, conv_type=conv_type)
    model = HST_PAGT_SSL_Model(encoder, hidden_dim, len(swat_indices), len(clim_indices), len(eo_indices))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Move model to CPU
    model.train()
    
    metrics = []
    
    # 30 epochs pretraining on CPU
    epochs_a = 10  # Reconstruction (S1 & S5)
    epochs_b = 10  # Future State Prediction (S2)
    epochs_c = 10  # Joint SSL (S1-S5)
    
    num_batches = int(np.ceil(num_samples / batch_size))
    
    for epoch in range(1, epochs_a + epochs_b + epochs_c + 1):
        # Shuffle samples
        perm = np.random.permutation(num_samples)
        X_train_shuffled = X_train[perm]
        
        epoch_loss = 0.0
        epoch_loss_recon = 0.0
        epoch_loss_future = 0.0
        epoch_loss_link = 0.0
        epoch_loss_align = 0.0
        epoch_loss_impute = 0.0
        
        # Decide Stage
        if epoch <= epochs_a:
            stage = "Stage A (Reconstruction)"
        elif epoch <= epochs_a + epochs_b:
            stage = "Stage B (Future Prediction)"
        else:
            stage = "Stage C (Joint SSL)"
            
        for b in range(num_batches):
            batch_data = X_train_shuffled[b*batch_size : (b+1)*batch_size]
            if len(batch_data) < 2: # Skip very small boundary batches to prevent shape errors
                continue
            
            # Convert to PyTorch Tensor
            x = torch.tensor(batch_data, dtype=torch.float32)
            B, N, T, D = x.shape
            
            loss = torch.tensor(0.0)
            
            # --- TASK S1 & S5: Masked Reconstruction / Imputation ---
            if stage in ["Stage A (Reconstruction)", "Stage C (Joint SSL)"]:
                # Mask 30% of inputs for S1 variables
                x_masked = x.clone()
                mask_s1 = torch.rand((B, N, T, len(recon_indices))) < 0.3
                for idx, feat_idx in enumerate(recon_indices):
                    x_masked[:, :, :, feat_idx] = torch.where(mask_s1[:, :, :, idx], 0.0, x_masked[:, :, :, feat_idx])
                
                # Mask 30% of inputs for S5 variables
                mask_s5 = torch.rand((B, N, T, len(impute_indices))) < 0.3
                for idx, feat_idx in enumerate(impute_indices):
                    x_masked[:, :, :, feat_idx] = torch.where(mask_s5[:, :, :, idx], 0.0, x_masked[:, :, :, feat_idx])
                
                # Forward call
                h_out = model(x_masked)
                
                # S1 Loss
                pred_recon = model.recon_head(h_out)
                target_recon = x[:, :, :, recon_indices]
                loss_s1 = ((pred_recon - target_recon) ** 2)[mask_s1].mean()
                if torch.isnan(loss_s1):
                    loss_s1 = torch.tensor(0.0)
                
                # S5 Loss
                pred_impute = model.impute_head(h_out)
                target_impute = x[:, :, :, impute_indices]
                loss_s5 = ((pred_impute - target_impute) ** 2)[mask_s5].mean()
                if torch.isnan(loss_s5):
                    loss_s5 = torch.tensor(0.0)
                
                loss = loss + loss_s1 + loss_s5
                epoch_loss_recon += loss_s1.item()
                epoch_loss_impute += loss_s5.item()

            # --- TASK S2: Future Prediction ---
            if stage in ["Stage B (Future Prediction)", "Stage C (Joint SSL)"]:
                # Pass only prefix of length 17
                x_prefix = x[:, :, :17, :]
                h_prefix = model.encoder.forward_prefix(x_prefix) # [B, N, 17, hidden_dim]
                h_last = h_prefix[:, :, 16, :] # [B, N, hidden_dim]
                
                pred_future = model.future_head(h_last) # [B, N, 3]
                target_future = torch.stack([x[:, :, 17, flow_out_idx], x[:, :, 19, flow_out_idx], x[:, :, 22, flow_out_idx]], dim=-1)
                loss_s2 = ((pred_future - target_future) ** 2).mean()
                
                # Scale loss factor
                loss = loss + 0.5 * loss_s2
                epoch_loss_future += loss_s2.item()

            # --- TASK S3 & S4: Link prediction & Alignment ---
            if stage == "Stage C (Joint SSL)":
                # Link Prediction (S3): edge connectivity
                # Bilinear prediction of 50 positive and 50 negative edges in batch
                E = edge_index.shape[1]
                pos_edges = edge_index[:, torch.randperm(E)[:50]]
                neg_src = torch.randint(0, N, (50,))
                neg_dst = torch.randint(0, N, (50,))
                neg_edges = torch.stack([neg_src, neg_dst], dim=0)
                
                # Select intermediate node embeddings
                h_recon = h_out[:, :, 12, :] # Slice at step 12
                # Project
                h_proj = model.link_proj(h_recon) # [B, N, hidden_dim]
                
                # Compute scores
                # Let's average score over batch for simplicity
                pos_scores = torch.sigmoid((h_proj[:, pos_edges[0]] * h_proj[:, pos_edges[1]]).sum(-1)).mean()
                neg_scores = torch.sigmoid((h_proj[:, neg_edges[0]] * h_proj[:, neg_edges[1]]).sum(-1)).mean()
                loss_s3 = -torch.log(pos_scores + 1e-8) - torch.log(1.0 - neg_scores + 1e-8)
                
                # Cross-Modal Alignment (S4)
                x_swat = x[:, :, :, swat_indices]
                x_clim = x[:, :, :, clim_indices]
                x_eo = x[:, :, :, eo_indices]
                loss_s4 = compute_alignment_loss(h_out, x_swat, x_clim, x_eo, model, swat_indices, clim_indices, eo_indices)
                
                loss = loss + 0.25 * loss_s3 + 1.0 * loss_s4
                epoch_loss_link += loss_s3.item()
                epoch_loss_align += loss_s4.item()

            # Backpropagation
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
        # Normalize stats
        epoch_loss /= num_batches
        epoch_loss_recon /= num_batches
        epoch_loss_future /= num_batches
        epoch_loss_link /= num_batches
        epoch_loss_align /= num_batches
        epoch_loss_impute /= num_batches
        
        print(f"Epoch {epoch:02d}/{epochs_a+epochs_b+epochs_c:02d} | Stage: {stage} | Loss: {epoch_loss:.4f} (Recon={epoch_loss_recon:.4f}, Impute={epoch_loss_impute:.4f}, Future={epoch_loss_future:.4f}, Link={epoch_loss_link:.4f}, Align={epoch_loss_align:.4f})")
        metrics.append({
            'Epoch': epoch,
            'Stage': stage,
            'Total_Loss': epoch_loss,
            'Recon_Loss': epoch_loss_recon,
            'Impute_Loss': epoch_loss_impute,
            'Future_Loss': epoch_loss_future,
            'Link_Loss': epoch_loss_link,
            'Align_Loss': epoch_loss_align
        })
        
    return model.encoder, pd.DataFrame(metrics)


# Main compilation flow
def main():
    print("=== Spatial Network Adjacency Setup ===")
    df_meta = pd.read_csv('metadata_reaches.csv')
    df_pre = pd.read_parquet('hydrographfm_pretrain_full.parquet')
    
    # Construct adjacency list (bi-directed)
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
    print(f"  Graph Adjacency constructed: {edge_index.shape[1]} total edges (bi-directed).")

    # Compute Positional Encodings
    pos_enc = compute_positional_encodings(df_meta, df_pre)
    
    # Process sequence features
    print("\nPreparing pretraining features from Parquet...")
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
    print(f"  Training features prepared (scaled): shape={X_train.shape}")

    # ==================== RUN ABLATION STUDY ====================
    print("\n================== ABLATION STUDY SETUP ==================")
    
    # Run pretraining for Model A (GATConv)
    gat_encoder, gat_metrics = run_pretraining(X_train, edge_index, pos_enc, feat_cols, conv_type='gat')
    
    # Run pretraining for Model B (TransformerConv)
    tr_encoder, tr_metrics = run_pretraining(X_train, edge_index, pos_enc, feat_cols, conv_type='transformer')
    
    # Save Model A Encoder Weights (frozen representation layer)
    print("\nSaving Primary Model A (GATConv) Encoder weights...")
    torch.save(gat_encoder.state_dict(), 'hydrographfm_encoder_v1.pt')
    print("  Saved to 'hydrographfm_encoder_v1.pt'")
    
    # Save Model B Encoder Weights for completeness
    torch.save(tr_encoder.state_dict(), 'hydrographfm_encoder_transformer.pt')
    print("  Saved to 'hydrographfm_encoder_transformer.pt'")
    
    # Write pretraining metrics (Model A)
    gat_metrics.to_csv('pretraining_metrics.csv', index=False)
    print("  pretraining_metrics.csv exported.")
    
    # Run a simple linear probe validation check to compare representation quality (ablation study)
    print("\n--- Running Downstream Ablation Evaluation (Linear Probing) ---")
    # Define a simple linear probe task: predict downstream flow_out at step 23 given step 23 embeddings
    gat_encoder.eval()
    tr_encoder.eval()
    
    # Extract step 23 embeddings for all samples
    # We batch samples to prevent memory overflow
    emb_gat_list = []
    emb_tr_list = []
    flow_targets = []
    
    with torch.no_grad():
        for b in range(int(np.ceil(num_samples / 16))):
            batch_data = X_train[b*16 : (b+1)*16]
            if len(batch_data) == 0:
                continue
            x = torch.tensor(batch_data, dtype=torch.float32)
            
            # Get temporal embeddings
            h_gat = gat_encoder(x) # [B, N, T, hidden_dim]
            h_tr = tr_encoder(x)
            
            # Slice at step 23 (last step)
            emb_gat_list.append(h_gat[:, :, 23, :].numpy())
            emb_tr_list.append(h_tr[:, :, 23, :].numpy())
            flow_targets.append(batch_data[:, :, 23, feat_cols.index('FLOW_OUT')])
            
    emb_gat = np.concatenate(emb_gat_list, axis=0) # [num_samples, num_reaches, hidden_dim]
    emb_tr = np.concatenate(emb_tr_list, axis=0)
    y_flow = np.concatenate(flow_targets, axis=0)  # [num_samples, num_reaches]
    
    # Flatten across samples and reaches for regression check
    X_gat = emb_gat.reshape(-1, 64)
    X_tr = emb_tr.reshape(-1, 64)
    y_flat = y_flow.ravel()
    
    # Standard regression: Solve via least squares to evaluate downstream linear predictability
    # Add bias column
    X_gat_bias = np.hstack([X_gat, np.ones((X_gat.shape[0], 1))])
    X_tr_bias = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
    
    w_gat, _, _, _ = np.linalg.lstsq(X_gat_bias, y_flat, rcond=None)
    w_tr, _, _, _ = np.linalg.lstsq(X_tr_bias, y_flat, rcond=None)
    
    # Predict
    pred_gat = X_gat_bias @ w_gat
    pred_tr = X_tr_bias @ w_tr
    
    # Calculate R2
    ss_tot = np.sum((y_flat - y_flat.mean())**2)
    r2_gat = 1 - np.sum((y_flat - pred_gat)**2) / (ss_tot + 1e-8)
    r2_tr = 1 - np.sum((y_flat - pred_tr)**2) / (ss_tot + 1e-8)
    
    mse_gat = np.mean((y_flat - pred_gat)**2)
    mse_tr = np.mean((y_flat - pred_tr)**2)
    
    print(f"Model A (GATConv) Downstream Linear Probe R2: {r2_gat:.4f} | MSE: {mse_gat:.4f}")
    print(f"Model B (TransformerConv) Downstream Linear Probe R2: {r2_tr:.4f} | MSE: {mse_tr:.4f}")
    
    # Export Ablation CSV
    df_ablation = pd.DataFrame({
        'Model': ['Model A (GATConv)', 'Model B (TransformerConv)'],
        'Spatial_Layer': ['GATConv', 'TransformerConv'],
        'Pretrain_Warmup_Loss': [gat_metrics['Total_Loss'].iloc[9], tr_metrics['Total_Loss'].iloc[9]],
        'Pretrain_Final_Loss': [gat_metrics['Total_Loss'].iloc[-1], tr_metrics['Total_Loss'].iloc[-1]],
        'Linear_Probe_R2_Flow': [r2_gat, r2_tr],
        'Linear_Probe_MSE_Flow': [mse_gat, mse_tr]
    })
    df_ablation.to_csv('hydrographfm_ablation_graph_layers.csv', index=False)
    print("  hydrographfm_ablation_graph_layers.csv exported.")
    
    print("\n=== PRETRAINING AND ABLATION COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
