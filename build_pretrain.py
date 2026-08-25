import pandas as pd
import numpy as np
import os

def get_norm(series):
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(0.0, index=series.index)
    return (series - s_min) / (s_max - s_min + 1e-8)

def main():
    print("=== HydroGraphEM Pretraining Dataset & Benchmark Suite Builder (Task 003) ===")

    # 1. Load Aligned Datasets
    print("Loading aligned reach datasets...")
    df_full = pd.read_parquet('hydrographfm_dataset_v1_full.parquet')
    df_modern = pd.read_parquet('hydrographfm_dataset_v1_modern.parquet')
    print(f"  Loaded full dataset: {df_full.shape} rows.")
    print(f"  Loaded modern dataset: {df_modern.shape} rows.")

    # 2. Load SWAT Subbasins Output for Runoff and Water Budget features
    print("Loading SWAT subbasins Parquet...")
    df_sub = pd.read_parquet('subbasins.parquet')
    # Filter for monthly records
    df_sub = df_sub[df_sub['type'] == 'monthly']
    print(f"  Loaded {len(df_sub)} monthly subbasin records.")

    # Select subbasin columns needed for benchmarks and pretraining features
    sub_cols = ['date', 'SUB', 'SURQmm', 'SWmm', 'GW_Qmm', 'WYLDmm', 'ETmm']
    df_sub_clean = df_sub[sub_cols].copy()
    df_sub_clean.rename(columns={'date': 'Date', 'SUB': 'Subbasin_ID'}, inplace=True)
    df_sub_clean['Date'] = pd.to_datetime(df_sub_clean['Date']).dt.strftime('%Y-%m-%d')

    # Ensure keys are matching types
    df_full['Reach_ID'] = df_full['Reach_ID'].astype(int)
    df_full['Subbasin_ID'] = df_full['Subbasin_ID'].astype(int)
    df_modern['Reach_ID'] = df_modern['Reach_ID'].astype(int)
    df_modern['Subbasin_ID'] = df_modern['Subbasin_ID'].astype(int)
    df_sub_clean['Subbasin_ID'] = df_sub_clean['Subbasin_ID'].astype(int)

    # 3. Merge Reach and Subbasin dataframes
    print("Merging reach features with subbasin features...")
    df_full_merged = df_full.merge(df_sub_clean, on=['Date', 'Subbasin_ID'], how='left')
    df_modern_merged = df_modern.merge(df_sub_clean, on=['Date', 'Subbasin_ID'], how='left')
    print(f"  Full merged shape: {df_full_merged.shape}")
    print(f"  Modern merged shape: {df_modern_merged.shape}")

    # Sort dataframes by Reach_ID and Date to ensure correct temporal order
    df_full_merged = df_full_merged.sort_values(['Reach_ID', 'Date']).reset_index(drop=True)
    df_modern_merged = df_modern_merged.sort_values(['Reach_ID', 'Date']).reset_index(drop=True)

    # 4. Compute P1 Flood Score
    # Formula: F = Norm(JRC_Occurrence) * Norm(Rainfall + FLOW_OUT + SURQmm) * (1 - Norm(Slope)) * Norm(TWI)
    print("Computing P1 Flood Score...")
    jrc_norm = get_norm(df_full_merged['JRC_Occurrence'].fillna(0))
    hydro_sum = df_full_merged['Rainfall'].fillna(0) + df_full_merged['FLOW_OUT'].fillna(0) + df_full_merged['SURQmm'].fillna(0)
    hydro_norm = get_norm(hydro_sum)
    slope_norm = get_norm(df_full_merged['Slope'].fillna(0))
    twi_norm = get_norm(df_full_merged['TWI'].fillna(0))

    df_full_merged['Flood_Score'] = jrc_norm * hydro_norm * (1.0 - slope_norm) * twi_norm
    print(f"  P1 Flood Score range: min={df_full_merged['Flood_Score'].min():.6f}, max={df_full_merged['Flood_Score'].max():.6f}, mean={df_full_merged['Flood_Score'].mean():.6f}")

    # 5. Compute P2 Drought Index
    # Formula: SDI = 0.2*Z_Rainfall + 0.2*Z_Soil_Moist + 0.2*Z_SWmm + 0.1*Z_Evaporation + 0.15*Z_GW_Qmm + 0.15*Z_WYLDmm
    print("Computing P2 Drought Index...")
    df_full_merged['Month'] = pd.to_datetime(df_full_merged['Date']).dt.month
    drought_vars = ['Rainfall', 'Soil_Moist', 'SWmm', 'Evaporation', 'GW_Qmm', 'WYLDmm']
    
    # Store standard Z-scores
    z_scores = {}
    for col in drought_vars:
        # Calculate mean and std for each Reach/Subbasin and calendar month over the 19-year period (2002-2020)
        stats = df_full_merged.groupby(['Subbasin_ID', 'Month'])[col].agg(['mean', 'std']).reset_index()
        stats.rename(columns={'mean': f'{col}_clim_mean', 'std': f'{col}_clim_std'}, inplace=True)
        
        # Merge climatology back to full dataframe
        df_full_merged = df_full_merged.merge(stats, on=['Subbasin_ID', 'Month'], how='left')
        
        # Compute Z-score
        mean_s = df_full_merged[f'{col}_clim_mean']
        std_s = df_full_merged[f'{col}_clim_std']
        df_full_merged[f'Z_{col}'] = (df_full_merged[col] - mean_s) / (std_s + 1e-8)
        
    df_full_merged['Drought_Index'] = (
        0.20 * df_full_merged['Z_Rainfall'] +
        0.20 * df_full_merged['Z_Soil_Moist'] +
        0.20 * df_full_merged['Z_SWmm'] +
        0.10 * df_full_merged['Z_Evaporation'] +
        0.15 * df_full_merged['Z_GW_Qmm'] +
        0.15 * df_full_merged['Z_WYLDmm']
    )
    print(f"  P2 Drought Index range: min={df_full_merged['Drought_Index'].min():.6f}, max={df_full_merged['Drought_Index'].max():.6f}, mean={df_full_merged['Drought_Index'].mean():.6f}")

    # 6. Compute P4 Event Prediction (6-month ahead prediction)
    print("Computing P4 Event Prediction Targets (6-month ahead shift)...")
    # Shift targets forward by 6 months (negative shift groups future values to current rows)
    df_full_merged['Flood_Score_6m'] = df_full_merged.groupby('Reach_ID')['Flood_Score'].shift(-6)
    df_full_merged['Drought_Index_6m'] = df_full_merged.groupby('Reach_ID')['Drought_Index'].shift(-6)

    # Establish event thresholds on the full period distributions
    flood_threshold = df_full_merged['Flood_Score'].quantile(0.90)
    drought_threshold = df_full_merged['Drought_Index'].quantile(0.10)
    print(f"  Flood Threshold (90th percentile of Flood_Score): {flood_threshold:.6f}")
    print(f"  Drought Threshold (10th percentile of Drought_Index): {drought_threshold:.6f}")

    # Construct binary indicators. Set to NaN if shifted target is NaN (i.e. last 6 months of dataset)
    df_full_merged['Flood_Event_6m'] = np.where(df_full_merged['Flood_Score_6m'].isna(), np.nan, (df_full_merged['Flood_Score_6m'] > flood_threshold).astype(float))
    df_full_merged['Drought_Event_6m'] = np.where(df_full_merged['Drought_Index_6m'].isna(), np.nan, (df_full_merged['Drought_Index_6m'] < drought_threshold).astype(float))

    # 7. Map Computed Targets to Modern Dataset
    # Since modern is a temporal subset, we merge the computed Flood_Score and Drought_Index columns
    print("Mapping computed targets to the modern dataset (2015-2020)...")
    targets_to_map = df_full_merged[['Reach_ID', 'Date', 'Flood_Score', 'Drought_Index']].copy()
    df_modern_merged = df_modern_merged.merge(targets_to_map, on=['Reach_ID', 'Date'], how='left')

    # 8. Export Benchmark Datasets
    print("Exporting Benchmark Datasets...")
    # P1: Flood Susceptibility
    df_p1 = df_full_merged[['Reach_ID', 'Date', 'Flood_Score']].dropna().copy()
    df_p1.to_parquet('benchmark_p1_flood.parquet', index=False)
    print(f"  benchmark_p1_flood.parquet saved: {df_p1.shape}")

    # P2: Drought
    df_p2 = df_full_merged[['Subbasin_ID', 'Date', 'Drought_Index']].dropna().copy()
    df_p2.to_parquet('benchmark_p2_drought.parquet', index=False)
    print(f"  benchmark_p2_drought.parquet saved: {df_p2.shape}")

    # P3: Water Quality (NO3, NH4, Sediment)
    df_p3 = df_full_merged[['Reach_ID', 'Date', 'NO3', 'NH4', 'Sediment']].dropna().copy()
    df_p3.to_parquet('benchmark_p3_water_quality.parquet', index=False)
    print(f"  benchmark_p3_water_quality.parquet saved: {df_p3.shape}")

    # P4: Event Prediction (6-month ahead)
    df_p4 = df_full_merged[['Reach_ID', 'Date', 'Flood_Event_6m', 'Drought_Event_6m']].dropna().copy()
    df_p4['Flood_Event_6m'] = df_p4['Flood_Event_6m'].astype(np.int32)
    df_p4['Drought_Event_6m'] = df_p4['Drought_Event_6m'].astype(np.int32)
    df_p4.to_parquet('benchmark_p4_event_prediction.parquet', index=False)
    print(f"  benchmark_p4_event_prediction.parquet saved: {df_p4.shape}")

    # 9. Build Rolling 24-Month Sequences for Pretraining Datasets
    print("\n--- Constructing Rolling 24-Month Sequences ---")
    
    # Seasonality Embeddings helper
    def add_seasonality_embeddings(df):
        dates_dt = pd.to_datetime(df['Date'])
        df['Month_of_Year'] = dates_dt.dt.month.astype(np.int32)
        df['Year'] = dates_dt.dt.year.astype(np.int32)
        # Monsoon Flag: 1 for June to September (6, 7, 8, 9), 0 otherwise
        df['Monsoon_Flag'] = df['Month_of_Year'].isin([6, 7, 8, 9]).astype(np.int32)
        return df

    # Prepare features for Full Pretraining Dataset
    # Clean up climatology stats and temp columns
    drop_cols = [c for c in df_full_merged.columns if '_clim_' in c or 'Z_' in c or '6m' in c or c == 'Month']
    df_full_features = df_full_merged.drop(columns=drop_cols)
    df_full_features = add_seasonality_embeddings(df_full_features)

    # Prepare features for Modern Pretraining Dataset
    # Modern doesn't have the climatology columns built-in, but check if there are any target columns from merge
    df_modern_features = df_modern_merged.copy()
    df_modern_features = add_seasonality_embeddings(df_modern_features)

    # Create sequence indexes using vectorized numpy sliding window
    def build_sequences(df_sorted, n_months):
        reach_ids = df_sorted['Reach_ID'].unique()
        num_reaches = len(reach_ids)
        
        # Verify that each reach has exactly the same number of months
        # Group by Reach_ID and count
        counts = df_sorted.groupby('Reach_ID').size()
        N = counts.iloc[0]
        if not (counts == N).all():
            raise ValueError("Reaches do not have the same number of time steps!")
            
        num_samples_per_reach = N - 23 # 24 months sequence length, so lag of 23
        
        print(f"  Building sequences: Reaches={num_reaches}, Steps per reach={N}, Samples per reach={num_samples_per_reach}")
        
        # Setup index offsets
        reach_offsets = np.arange(num_reaches)[:, None] * N
        sample_offsets = np.arange(num_samples_per_reach)[None, :]
        start_indices = reach_offsets + sample_offsets # Shape: [num_reaches, num_samples_per_reach]
        
        step_offsets = np.arange(24)[None, None, :]
        all_indices = start_indices[:, :, None] + step_offsets # Shape: [num_reaches, num_samples_per_reach, 24]
        
        flat_indices = all_indices.ravel()
        
        # Slice the dataframe
        df_seq = df_sorted.iloc[flat_indices].copy()
        
        # Construct Sample_ID and Time_Lag
        reach_ids_expanded = np.repeat(reach_ids, num_samples_per_reach)
        sample_ids_base = reach_ids_expanded * 1000 + np.tile(np.arange(num_samples_per_reach), num_reaches)
        
        df_seq['Sample_ID'] = np.repeat(sample_ids_base, 24).astype(np.int64)
        df_seq['Time_Lag'] = np.tile(np.arange(24), num_reaches * num_samples_per_reach).astype(np.int32)
        
        # Reorder columns to put Sample_ID and Time_Lag first
        meta_cols = ['Sample_ID', 'Time_Lag', 'Date', 'Reach_ID', 'Subbasin_ID', 'Month_of_Year', 'Year', 'Monsoon_Flag']
        other_cols = [c for c in df_seq.columns if c not in meta_cols]
        df_seq = df_seq[meta_cols + other_cols]
        
        return df_seq

    print("Building Full Pretraining Sequences (2002-2020)...")
    df_pretrain_full = build_sequences(df_full_features, 228)
    print(f"  Full sequence shape: {df_pretrain_full.shape}")
    
    print("Building Modern Pretraining Sequences (2015-2020)...")
    df_pretrain_modern = build_sequences(df_modern_features, 72)
    print(f"  Modern sequence shape: {df_pretrain_modern.shape}")

    # 10. Write Pretraining Parquet files
    print("\nWriting Pretraining Parquets...")
    df_pretrain_full.to_parquet('hydrographfm_pretrain_full.parquet', index=False)
    print("  hydrographfm_pretrain_full.parquet saved.")
    
    # Save a version of full dataset to hydrographfm_pretrain_v1.parquet to satisfy exact request
    df_pretrain_full.to_parquet('hydrographfm_pretrain_v1.parquet', index=False)
    print("  hydrographfm_pretrain_v1.parquet saved.")

    df_pretrain_modern.to_parquet('hydrographfm_pretrain_modern.parquet', index=False)
    print("  hydrographfm_pretrain_modern.parquet saved.")

    print("\n=== DATASET SEQUENTIAL PRETRAINING BUILDER COMPLETE ===")

if __name__ == '__main__':
    main()
