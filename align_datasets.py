import pandas as pd
import numpy as np

def main():
    print("=== Spatial-Temporal Dataset Alignment (Task 002) ===")
    
    # 1. Load SWAT reaches monthly outputs
    print("Loading SWAT reaches Parquet...")
    df_swat = pd.read_parquet('reaches.parquet')
    
    # Keep only monthly rows (exclude annual summaries and multi-year averages)
    df_swat = df_swat[df_swat['type'] == 'monthly']
    print(f"  Loaded {len(df_swat)} monthly SWAT records.")
    
    # 2. Load GEE reach features
    print("Loading GEE reach features Parquet...")
    df_gee = pd.read_parquet('gee_reach_features.parquet')
    print(f"  Loaded {len(df_gee)} monthly GEE records.")
    
    # 3. Load reach metadata (contains topology and coordinates)
    print("Loading reaches metadata CSV...")
    df_meta = pd.read_csv('metadata_reaches.csv')
    
    # 4. Merge SWAT and GEE
    # SWAT key: RCH (Reach ID), date (string like '2002-01-01')
    # GEE key: Reach_ID, Date
    print("Merging SWAT and GEE monthly data...")
    df_swat['RCH'] = df_swat['RCH'].astype(int)
    df_gee['Reach_ID'] = df_gee['Reach_ID'].astype(int)
    
    df_merged = df_swat.merge(
        df_gee,
        left_on=['RCH', 'date'],
        right_on=['Reach_ID', 'Date'],
        how='inner'
    )
    print(f"  Merged data contains {len(df_merged)} records.")
    
    # 5. Merge with Metadata
    print("Merging with spatial and graph metadata...")
    df_meta['Reach_ID'] = df_meta['Reach_ID'].astype(int)
    df_final = df_merged.merge(df_meta, on='Reach_ID', how='inner')
    
    # Ensure date columns are formatted consistently
    df_final['Date'] = pd.to_datetime(df_final['Date'])
    
    # 6. Extract Dataset A: Full Period (2002-2020)
    print("\nCompiling Dataset A (Full Period: 2002-2020)...")
    # Date bounds: 2002-01-01 to 2020-12-31
    df_full = df_final[
        (df_final['Date'] >= '2002-01-01') & 
        (df_final['Date'] <= '2020-12-31')
    ].copy()
    
    # Select columns for Dataset A (No S1 radar backscatter)
    cols_full = [
        # IDs & Temporal
        'Date', 'Reach_ID', 'Subbasin_ID',
        
        # Hydrology Features (from SWAT)
        'FLOW_INcms', 'FLOW_OUTcms', 'EVAPcms', 'TLOSScms', 'SED_OUTtons', 
        'TOT_Nkg', 'TOT_Pkg', 'NO3_OUTkg', 'NH4_OUTkg', 'WTMPdegc',
        
        # Climate Features (from GEE)
        'Rainfall', 'Temp_Mean', 'Soil_Moist', 'Evaporation',
        
        # Remote Sensing Features (from GEE - Optical Landsat/Sentinel-2 only)
        'NDVI', 'NDWI', 'MNDWI', 'EVI',
        
        # Terrain Features (from GEE/DEM)
        'Elevation', 'Slope', 'TWI',
        
        # Surface Water Features (from GRC/GSW)
        'JRC_Occurrence', 'JRC_Seasonality', 'JRC_Recurrence',
        
        # Land Cover Features (from ESA/Dynamic World)
        'Cropland_pct', 'Wetland_pct', 'Builtup_pct', 'Forest_pct', 'Water_pct',
        
        # Graph & Topological Features
        'Upstream_ID', 'Downstream_ID', 'Area'
    ]
    
    # Rename columns to standard, clean schema names
    schema_map = {
        'FLOW_INcms': 'FLOW_IN',
        'FLOW_OUTcms': 'FLOW_OUT',
        'EVAPcms': 'Channel_Evap',
        'TLOSScms': 'Transmission_Loss',
        'SED_OUTtons': 'Sediment',
        'TOT_Nkg': 'TN',
        'TOT_Pkg': 'TP',
        'NO3_OUTkg': 'NO3',
        'NH4_OUTkg': 'NH4',
        'WTMPdegc': 'Temp_Water',
        'Area': 'Reach_Area'
    }
    
    df_full_clean = df_full[cols_full].rename(columns=schema_map)
    df_full_clean['Date'] = df_full_clean['Date'].dt.strftime('%Y-%m-%d')
    df_full_clean.to_parquet('hydrographfm_dataset_v1_full.parquet', index=False)
    print(f"  hydrographfm_dataset_v1_full.parquet generated: {df_full_clean.shape} rows/columns.")
    
    # 7. Extract Dataset B: Modern Period (2015-2020)
    print("\nCompiling Dataset B (Modern Period: 2015-2020)...")
    # Date bounds: 2015-01-01 to 2020-12-31
    df_modern = df_final[
        (df_final['Date'] >= '2015-01-01') & 
        (df_final['Date'] <= '2020-12-31')
    ].copy()
    
    # Select columns for Dataset B (Includes Sentinel-1 radar backscatter: VV, VH, ratio)
    cols_modern = [
        # IDs & Temporal
        'Date', 'Reach_ID', 'Subbasin_ID',
        
        # Hydrology Features (from SWAT)
        'FLOW_INcms', 'FLOW_OUTcms', 'EVAPcms', 'TLOSScms', 'SED_OUTtons', 
        'TOT_Nkg', 'TOT_Pkg', 'NO3_OUTkg', 'NH4_OUTkg', 'WTMPdegc',
        
        # Climate Features (from GEE)
        'Rainfall', 'Temp_Mean', 'Soil_Moist', 'Evaporation',
        
        # Remote Sensing Features (from GEE - Sentinel-1 SAR & Optical Sentinel-2)
        'VV', 'VH', 'VV_VH_ratio', 'NDVI', 'NDWI', 'MNDWI', 'EVI',
        
        # Terrain Features (from GEE/DEM)
        'Elevation', 'Slope', 'TWI',
        
        # Surface Water Features (from GRC/GSW)
        'JRC_Occurrence', 'JRC_Seasonality', 'JRC_Recurrence',
        
        # Land Cover Features (from ESA/Dynamic World)
        'Cropland_pct', 'Wetland_pct', 'Builtup_pct', 'Forest_pct', 'Water_pct',
        
        # Graph & Topological Features
        'Upstream_ID', 'Downstream_ID', 'Area'
    ]
    
    df_modern_clean = df_modern[cols_modern].rename(columns=schema_map)
    df_modern_clean['Date'] = df_modern_clean['Date'].dt.strftime('%Y-%m-%d')
    df_modern_clean.to_parquet('hydrographfm_dataset_v1_modern.parquet', index=False)
    print(f"  hydrographfm_dataset_v1_modern.parquet generated: {df_modern_clean.shape} rows/columns.")
    
    print("\n--- STAGE 2: DATASET ALIGNMENT AND EXPORT COMPLETE ---")

if __name__ == '__main__':
    main()
