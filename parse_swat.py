import os
import re
import pandas as pd
import numpy as np

def parse_cio(cio_path):
    """
    Parses file.cio to extract:
    - NBYR: Number of years simulated
    - IYR: Beginning year of simulation
    - IDAF: Beginning julian day
    - IDAL: Ending julian day
    - IPRINT: Print code (0=monthly, 1=daily, 2=yearly)
    - NYSKIP: Number of years to skip output printing
    """
    print(f"Parsing CIO file: {cio_path}")
    cio_data = {}
    with open(cio_path, 'r') as f:
        for line in f:
            if '|' in line:
                val_part, desc_part = line.split('|', 1)
                val_str = val_part.strip()
                desc = desc_part.strip()
                
                # Check for NBYR, IYR, IPRINT, NYSKIP, etc.
                if 'NBYR' in desc:
                    cio_data['NBYR'] = int(val_str)
                elif 'IYR' in desc:
                    cio_data['IYR'] = int(val_str)
                elif 'IDAF' in desc:
                    cio_data['IDAF'] = int(val_str)
                elif 'IDAL' in desc:
                    cio_data['IDAL'] = int(val_str)
                elif 'IPRINT' in desc:
                    cio_data['IPRINT'] = int(val_str)
                elif 'NYSKIP' in desc:
                    cio_data['NYSKIP'] = int(val_str)
                    
    # Fill in defaults if not found
    cio_data.setdefault('NBYR', 21)
    cio_data.setdefault('IYR', 2000)
    cio_data.setdefault('NYSKIP', 2)
    cio_data.setdefault('IPRINT', 0)
    
    start_year = cio_data['IYR'] + cio_data['NYSKIP']
    end_year = cio_data['IYR'] + cio_data['NBYR'] - 1
    years = list(range(start_year, end_year + 1))
    total_years_count = float(len(years))
    
    frequency = 'monthly' if cio_data['IPRINT'] == 0 else 'daily' if cio_data['IPRINT'] == 1 else 'yearly'
    
    print(f"Auto-detected settings:")
    print(f"  Simulation Years: {cio_data['NBYR']}")
    print(f"  Starting Year:    {cio_data['IYR']}")
    print(f"  Skipped Years:    {cio_data['NYSKIP']}")
    print(f"  Output Start Yr:  {start_year}")
    print(f"  Output End Yr:    {end_year}")
    print(f"  Output Interval:  {frequency}")
    
    cio_data['start_year'] = start_year
    cio_data['end_year'] = end_year
    cio_data['years'] = years
    cio_data['total_years_count'] = total_years_count
    cio_data['frequency'] = frequency
    
    return cio_data

def parse_fig(path):
    """
    Parses fig.fig to build downstream and upstream reach routing topology.
    Returns:
      reach_downstream: dict, reach_id -> downstream_reach_id
      reach_upstream: dict, reach_id -> list of upstream_reach_ids
    """
    print(f"Parsing FIG file for routing topology: {path}")
    hydro_sources = {}
    reach_downstream = {}
    reach_upstream = {}
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        parts = line.split()
        if not parts:
            i += 1
            continue
        cmd = parts[0].lower()
        
        if cmd == 'subbasin':
            try:
                sub_id = int(parts[2])
                hydro_sources[sub_id] = {sub_id}
            except (IndexError, ValueError):
                pass
            i += 2 # skip the .sub file name line
            
        elif cmd == 'route':
            try:
                out_hydro = int(parts[2])
                reach_id = int(parts[3])
                in_hydro = int(parts[4])
                
                hydro_sources[out_hydro] = {reach_id}
                
                sources = hydro_sources.get(in_hydro, set())
                for src_reach in sources:
                    if src_reach != reach_id:
                        reach_downstream[src_reach] = reach_id
                        reach_upstream.setdefault(reach_id, set()).add(src_reach)
            except (IndexError, ValueError):
                pass
            i += 2 # skip the .rte file name line
            
        elif cmd == 'add':
            try:
                out_hydro = int(parts[2])
                in_hydro1 = int(parts[3])
                in_hydro2 = int(parts[4])
                
                sources1 = hydro_sources.get(in_hydro1, set())
                sources2 = hydro_sources.get(in_hydro2, set())
                hydro_sources[out_hydro] = sources1.union(sources2)
            except (IndexError, ValueError):
                pass
            i += 1
            
        else:
            i += 1
            
    # Convert sets to sorted lists for representation
    reach_upstream_list = {k: sorted(list(v)) for k, v in reach_upstream.items()}
    return reach_downstream, reach_upstream_list

def generate_steps_metadata(path, total_nodes, years, total_years_count, frequency):
    """
    Scans the file to detect the number of blocks and maps each block index
    to its metadata (type, year, month/day, raw MON string).
    """
    print(f"Scanning blocks in {os.path.basename(path)}...")
    steps = []
    first_node_id = None
    
    is_rch = 'output.rch' in path.lower()
    is_sub = 'output.sub' in path.lower()
    is_hru = 'output.hru' in path.lower()
    
    with open(path, 'r') as f:
        # Skip 9 header lines
        for _ in range(9):
            next(f)
            
        for line in f:
            if not line.strip():
                continue
            
            try:
                if is_rch:
                    node_id = int(float(line[6:11]))
                    mon_str = line[20:26].strip()
                elif is_sub:
                    node_id = int(float(line[6:11]))
                    mon_str = line[20:25].strip()
                elif is_hru:
                    node_id = int(float(line[4:9]))
                    mon_str = line[29:34].strip()
            except (ValueError, IndexError):
                continue
                
            if first_node_id is None:
                first_node_id = node_id
                
            if node_id == first_node_id:
                # Started a new time step block
                steps.append(mon_str)
                
    print(f"Found {len(steps)} blocks in the file.")
    
    # Map raw MON strings to years/months
    steps_metadata = []
    year_idx = 0
    current_year = years[0] if years else None
    
    for i, mon_str in enumerate(steps):
        # Clean MON string (can be integer like '1', float like '19.0', or year like '2002')
        val = float(mon_str)
        
        # Determine step type
        if val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
            # Monthly step
            # If MON is 1 and it's not the first step, we check if we transitioned to a new year
            if val == 1.0 and i > 0 and steps_metadata[-1]['type'] != 'monthly':
                # Transitioned to a new year after annual summary
                if year_idx < len(years) - 1:
                    year_idx += 1
                    current_year = years[year_idx]
            
            steps_metadata.append({
                'type': 'monthly',
                'year': current_year,
                'month': int(val),
                'date': f"{current_year}-{int(val):02d}-01",
                'raw': mon_str
            })
        elif val >= 1900.0:
            # Annual summary row
            steps_metadata.append({
                'type': 'annual',
                'year': int(val),
                'month': None,
                'date': None,
                'raw': mon_str
            })
        else:
            # Multi-year average row (e.g. 19.0)
            steps_metadata.append({
                'type': 'average',
                'year': None,
                'month': None,
                'date': None,
                'raw': mon_str
            })
            
    return steps_metadata

def parse_float(val_str):
    val_str = val_str.strip()
    if not val_str or '*' in val_str or 'nan' in val_str.lower():
        return 0.0
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def parse_line_rch(line):
    rch_type = line[0:6].strip()
    rch = int(float(line[6:11]))
    gis = int(float(line[11:20]))
    mon = float(line[20:26])
    vals = [rch_type, rch, gis, mon]
    idx = 26
    for _ in range(47):
        val_str = line[idx:idx+12]
        vals.append(parse_float(val_str))
        idx += 12
    return vals

def parse_line_sub(line):
    sub_type = line[0:6].strip()
    sub = int(float(line[6:11]))
    gis = int(float(line[11:20]))
    mon = float(line[20:25])
    area = parse_float(line[25:35])
    vals = [sub_type, sub, gis, mon, area]
    idx = 35
    for _ in range(24):
        val_str = line[idx:idx+10]
        vals.append(parse_float(val_str))
        idx += 10
    return vals

def parse_line_hru(line):
    lulc = line[0:4].strip()
    hru = int(float(line[4:9]))
    gis = line[9:19].strip()
    sub = int(float(line[19:24]))
    mgt = int(float(line[24:29]))
    mon = float(line[29:34])
    area = parse_float(line[34:44])
    vals = [lulc, hru, gis, sub, mgt, mon, area]
    idx = 44
    for _ in range(79):
        val_str = line[idx:idx+10]
        vals.append(parse_float(val_str))
        idx += 10
    return vals

def parse_output_rch(path, steps_meta, reach_downstream, reach_upstream, out_parquet):
    print(f"\nParsing output.rch from {path}...")
    
    columns = [
        'RCH', 'GIS', 'MON', 'AREAkm2', 'FLOW_INcms', 'FLOW_OUTcms', 'EVAPcms', 'TLOSScms', 
        'SED_INtons', 'SED_OUTtons', 'SEDCONCmg/L', 'ORGN_INkg', 'ORGN_OUTkg', 'ORGP_INkg', 
        'ORGP_OUTkg', 'NO3_INkg', 'NO3_OUTkg', 'NH4_INkg', 'NH4_OUTkg', 'NO2_INkg', 'NO2_OUTkg', 
        'MINP_INkg', 'MINP_OUTkg', 'CHLA_INkg', 'CHLA_OUTkg', 'CBOD_INkg', 'CBOD_OUTkg', 
        'DISOX_INkg', 'DISOX_OUTkg', 'SOLPST_INmg', 'SOLPST_OUTmg', 'SORPST_INmg', 'SORPST_OUTmg', 
        'REACTPSTmg', 'VOLPSTmg', 'SETTLPSTmg', 'RESUSP_PSTmg', 'DIFFUSEPSTmg', 'REACBEDPSTmg', 
        'BURYPSTmg', 'BED_PSTmg', 'BACTP_OUTct', 'BACTLP_OUTct', 'CMETAL#1kg', 'CMETAL#2kg', 
        'CMETAL#3kg', 'TOT_Nkg', 'TOT_Pkg', 'NO3ConcMg/l', 'WTMPdegc'
    ]
    
    rows_data = []
    first_node_id = None
    current_step_idx = -1
    
    with open(path, 'r') as f:
        for _ in range(9):
            next(f)
            
        for line in f:
            if not line.strip():
                continue
            vals = parse_line_rch(line)
            node_id = vals[1]
            if first_node_id is None:
                first_node_id = node_id
                
            if node_id == first_node_id:
                current_step_idx += 1
                
            meta = steps_meta[current_step_idx]
            float_vals = vals[4:]
            
            row = [
                node_id,
                meta['type'],
                meta['year'],
                meta['month'],
                meta['date']
            ] + float_vals
            
            rows_data.append(row)
            
    df_cols = ['RCH', 'type', 'year', 'month', 'date'] + columns[3:]
    df = pd.DataFrame(rows_data, columns=df_cols)
    
    print(f"Saving to Parquet: {out_parquet}")
    df.to_parquet(out_parquet, index=False)
    
    # Extract metadata reaches
    print("Generating metadata_reaches.csv...")
    # Get first step (which is monthly step from the first year)
    df_m1 = df[(df['type'] == 'monthly') & (df['month'] == 1) & (df['year'] == df['year'].min())]
    metadata_rows = []
    for _, row in df_m1.iterrows():
        reach_id = int(row['RCH'])
        area = row['AREAkm2']
        
        downstream_id = reach_downstream.get(reach_id, np.nan)
        upstream_list = reach_upstream.get(reach_id, [])
        upstream_str = ",".join(map(str, upstream_list)) if upstream_list else np.nan
        
        metadata_rows.append({
            'Reach_ID': reach_id,
            'Subbasin_ID': reach_id, # In SWAT subbasin matches reach
            'Upstream_ID': upstream_str,
            'Downstream_ID': downstream_id,
            'Latitude': np.nan,
            'Longitude': np.nan,
            'Area': area
        })
        
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv("metadata_reaches.csv", index=False)
    
    return df

def parse_output_sub(path, steps_meta, reach_downstream, out_parquet):
    print(f"\nParsing output.sub from {path}...")
    
    columns = [
        'SUB', 'GIS', 'MON', 'AREAkm2', 'PRECIPmm', 'SNOMELTmm', 'PETmm', 'ETmm', 'SWmm', 
        'PERCmm', 'SURQmm', 'GW_Qmm', 'WYLDmm', 'SYLDt_ha', 'ORGNkg_ha', 'ORGPkg_ha', 
        'NSURQkg_ha', 'SOLPkg_ha', 'SEDPkg_ha', 'LATQ_mm', 'LATNO3kg_h', 'GWNO3kg_ha', 
        'CHOLAmic_L', 'CBODU_mg_L', 'DOXQ_mg_L', 'TNO3kg_ha', 'QTILEmm', 'TVAPkg_ha'
    ]
    
    rows_data = []
    first_node_id = None
    current_step_idx = -1
    
    with open(path, 'r') as f:
        for _ in range(9):
            next(f)
            
        for line in f:
            if not line.strip():
                continue
            vals = parse_line_sub(line)
            node_id = vals[1]
            if first_node_id is None:
                first_node_id = node_id
                
            if node_id == first_node_id:
                current_step_idx += 1
                
            meta = steps_meta[current_step_idx]
            area_val = vals[4]
            float_vals = vals[5:]
            
            row = [
                node_id,
                meta['type'],
                meta['year'],
                meta['month'],
                meta['date'],
                area_val
            ] + float_vals
            
            rows_data.append(row)
            
    df_cols = ['SUB', 'type', 'year', 'month', 'date', 'AREAkm2'] + columns[4:]
    df = pd.DataFrame(rows_data, columns=df_cols)
    
    print(f"Saving to Parquet: {out_parquet}")
    df.to_parquet(out_parquet, index=False)
    
    # Extract metadata subbasins
    print("Generating metadata_subbasins.csv...")
    df_m1 = df[(df['type'] == 'monthly') & (df['month'] == 1) & (df['year'] == df['year'].min())]
    metadata_rows = []
    for _, row in df_m1.iterrows():
        sub_id = int(row['SUB'])
        area = row['AREAkm2']
        downstream_sub_id = reach_downstream.get(sub_id, np.nan)
        
        metadata_rows.append({
            'Subbasin_ID': sub_id,
            'Downstream_Subbasin_ID': downstream_sub_id,
            'Latitude': np.nan,
            'Longitude': np.nan,
            'Area': area
        })
        
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv("metadata_subbasins.csv", index=False)
    
    return df

def parse_output_hru(path, steps_meta, out_parquet):
    print(f"\nParsing output.hru from {path}...")
    
    columns_float = [
        'PRECIPmm', 'SNOFALLmm', 'SNOMELTmm', 'IRRmm', 'PETmm', 'ETmm', 'SW_INITmm', 'SW_ENDmm', 
        'PERCmm', 'GW_RCHGmm', 'DA_RCHGmm', 'REVAPmm', 'SA_IRRmm', 'DA_IRRmm', 'SA_STmm', 'DA_STmm', 
        'SURQ_GENmm', 'SURQ_CNTmm', 'TLOSSmm', 'LATQGENmm', 'GW_Qmm', 'WYLDmm', 'DAILYCN', 
        'TMP_AVdgC', 'TMP_MXdgC', 'TMP_MNdgC', 'SOL_TMPdgC', 'SOLARMJ_m2', 'SYLDt_ha', 'USLEt_ha', 
        'N_APPkg_ha', 'P_APPkg_ha', 'NAUTOkg_ha', 'PAUTOkg_ha', 'NGRZkg_ha', 'PGRZkg_ha', 
        'NCFRTkg_ha', 'PCFRTkg_ha', 'NRAINkg_ha', 'NFIXkg_ha', 'F_MNkg_ha', 'A_MNkg_ha', 
        'A_SNkg_ha', 'F_MPkg_ha', 'AO_LPkg_ha', 'L_APkg_ha', 'A_SPkg_ha', 'DNITkg_ha', 
        'NUPkg_ha', 'PUPkg_ha', 'ORGNkg_ha', 'ORGPkg_ha', 'SEDPkg_ha', 'NSURQkg_ha', 
        'NLATQkg_ha', 'NO3Lkg_ha', 'NO3GWkg_ha', 'SOLPkg_ha', 'P_GWkg_ha', 'W_STRS', 
        'TMP_STRS', 'N_STRS', 'P_STRS', 'BIOMt_ha', 'LAI', 'YLDt_ha', 'BACTPct', 'BACTLPct', 
        'WTAB_CLIm', 'WTAB_SOLm', 'SNOmm', 'CMUPkg_ha', 'CMTOTkg_ha', 'QTILEmm', 'TNO3kg_ha', 
        'LNO3kg_ha', 'GW_Q_Dmm', 'LATQCNTmm', 'TVAPkg_ha'
    ]
    
    rows_data = []
    first_node_id = None
    current_step_idx = -1
    
    with open(path, 'r') as f:
        for _ in range(9):
            next(f)
            
        for line in f:
            if not line.strip():
                continue
            vals = parse_line_hru(line)
            node_id = vals[1] # Local HRU index
            if first_node_id is None:
                first_node_id = node_id
                
            if node_id == first_node_id:
                current_step_idx += 1
                
            meta = steps_meta[current_step_idx]
            
            lulc = vals[0]
            gis_id = vals[2]
            sub_id = vals[3]
            mgt_id = vals[4]
            area_val = vals[6]
            float_vals = vals[7:]
            
            row = [
                lulc,
                node_id,
                gis_id,
                sub_id,
                mgt_id,
                meta['type'],
                meta['year'],
                meta['month'],
                meta['date'],
                area_val
            ] + float_vals
            
            rows_data.append(row)
            
    df_cols = ['LULC', 'HRU', 'GIS', 'SUB', 'MGT', 'type', 'year', 'month', 'date', 'AREAkm2'] + columns_float
    df = pd.DataFrame(rows_data, columns=df_cols)
    
    print(f"Saving to Parquet: {out_parquet}")
    df.to_parquet(out_parquet, index=False)
    
    # Extract metadata HRUs
    print("Generating metadata_hrus.csv...")
    df_m1 = df[(df['type'] == 'monthly') & (df['month'] == 1) & (df['year'] == df['year'].min())]
    metadata_rows = []
    for _, row in df_m1.iterrows():
        sub_id = int(row['SUB'])
        hru_local = int(row['HRU'])
        hru_id = sub_id * 10000 + hru_local
        gis_id = row['GIS']
        lulc = row['LULC']
        mgt_id = int(row['MGT'])
        area = row['AREAkm2']
        
        metadata_rows.append({
            'HRU_ID': hru_id,
            'Subbasin_ID': sub_id,
            'GIS_ID': gis_id,
            'LULC': lulc,
            'MGT_ID': mgt_id,
            'Latitude': np.nan,
            'Longitude': np.nan,
            'Area': area
        })
        
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv("metadata_hrus.csv", index=False)
    
    return df

def generate_feature_dictionaries():
    print("\nGenerating feature dictionary files...")
    
    rch_features = [
        {"variable": "RCH", "unit": "None", "description": "Reach segment numeric identifier"},
        {"variable": "AREAkm2", "unit": "km2", "description": "Drainage area contributing to the reach segment"},
        {"variable": "FLOW_INcms", "unit": "m3/s", "description": "Streamflow volume routing into the reach channel segment"},
        {"variable": "FLOW_OUTcms", "unit": "m3/s", "description": "Streamflow volume routing out of the reach channel segment"},
        {"variable": "EVAPcms", "unit": "m3/s", "description": "Evaporation losses from the stream reach surface"},
        {"variable": "TLOSScms", "unit": "m3/s", "description": "Transmission losses from the stream reach channel bed"},
        {"variable": "SED_INtons", "unit": "metric tons", "description": "Sediment mass routing into the reach segment"},
        {"variable": "SED_OUTtons", "unit": "metric tons", "description": "Sediment mass routing out of the reach segment"},
        {"variable": "SEDCONCmg/L", "unit": "mg/L", "description": "Suspended sediment concentration in reach streamflow"},
        {"variable": "ORGN_INkg", "unit": "kg", "description": "Organic Nitrogen load entering the reach"},
        {"variable": "ORGN_OUTkg", "unit": "kg", "description": "Organic Nitrogen load leaving the reach"},
        {"variable": "ORGP_INkg", "unit": "kg", "description": "Organic Phosphorus load entering the reach"},
        {"variable": "ORGP_OUTkg", "unit": "kg", "description": "Organic Phosphorus load leaving the reach"},
        {"variable": "NO3_INkg", "unit": "kg N", "description": "Nitrate (NO3) mass load entering the reach segment"},
        {"variable": "NO3_OUTkg", "unit": "kg N", "description": "Nitrate (NO3) mass load leaving the reach segment"},
        {"variable": "NH4_INkg", "unit": "kg N", "description": "Ammonium (NH4) mass load entering the reach segment"},
        {"variable": "NH4_OUTkg", "unit": "kg N", "description": "Ammonium (NH4) mass load leaving the reach segment"},
        {"variable": "NO2_INkg", "unit": "kg N", "description": "Nitrite (NO2) mass load entering the reach segment"},
        {"variable": "NO2_OUTkg", "unit": "kg N", "description": "Nitrite (NO2) mass load leaving the reach segment"},
        {"variable": "MINP_INkg", "unit": "kg P", "description": "Mineral/soluble Phosphorus mass load entering the reach segment"},
        {"variable": "MINP_OUTkg", "unit": "kg P", "description": "Mineral/soluble Phosphorus mass load leaving the reach segment"},
        {"variable": "CHLA_INkg", "unit": "kg", "description": "Chlorophyll-a load entering the reach segment"},
        {"variable": "CHLA_OUTkg", "unit": "kg", "description": "Chlorophyll-a load leaving the reach segment"},
        {"variable": "CBOD_INkg", "unit": "kg O2", "description": "Carbonaceous Biochemical Oxygen Demand entering the reach"},
        {"variable": "CBOD_OUTkg", "unit": "kg O2", "description": "Carbonaceous Biochemical Oxygen Demand leaving the reach"},
        {"variable": "DISOX_INkg", "unit": "kg O2", "description": "Dissolved Oxygen load entering the reach segment"},
        {"variable": "DISOX_OUTkg", "unit": "kg O2", "description": "Dissolved Oxygen load leaving the reach segment"},
        {"variable": "TOT_Nkg", "unit": "kg N", "description": "Total Nitrogen loading transported out of the reach segment"},
        {"variable": "TOT_Pkg", "unit": "kg P", "description": "Total Phosphorus loading transported out of the reach segment"},
        {"variable": "NO3ConcMg/l", "unit": "mg N/L", "description": "Mean Nitrate concentration in reach streamflow"},
        {"variable": "WTMPdegc", "unit": "Celsius", "description": "Average water temperature in reach channel segment"}
    ]
    pd.DataFrame(rch_features).to_csv("feature_dictionary_reaches.csv", index=False)
    
    sub_features = [
        {"variable": "SUB", "unit": "None", "description": "Subbasin numeric identifier"},
        {"variable": "AREAkm2", "unit": "km2", "description": "Subbasin physical surface area"},
        {"variable": "PRECIPmm", "unit": "mm", "description": "Total precipitation sum over the subbasin"},
        {"variable": "SNOMELTmm", "unit": "mm H2O", "description": "Snowmelt volume contributing to subbasin runoff"},
        {"variable": "PETmm", "unit": "mm H2O", "description": "Potential Evapotranspiration over the subbasin surface area"},
        {"variable": "ETmm", "unit": "mm H2O", "description": "Actual Evapotranspiration over the subbasin surface area"},
        {"variable": "SWmm", "unit": "mm H2O", "description": "Soil water volume retained in the subbasin soil columns"},
        {"variable": "PERCmm", "unit": "mm H2O", "description": "Water percolation out of the root zone"},
        {"variable": "SURQmm", "unit": "mm H2O", "description": "Surface runoff contribution from the subbasin"},
        {"variable": "GW_Qmm", "unit": "mm H2O", "description": "Groundwater discharge into reach routing channels"},
        {"variable": "WYLDmm", "unit": "mm H2O", "description": "Total Water Yield leaving the subbasin"},
        {"variable": "SYLDt_ha", "unit": "metric tons/ha", "description": "Sediment yield leaving the subbasin area"}
    ]
    pd.DataFrame(sub_features).to_csv("feature_dictionary_subbasins.csv", index=False)
    
    hru_features = [
        {"variable": "HRU", "unit": "None", "description": "HRU local index within the subbasin"},
        {"variable": "GIS", "unit": "None", "description": "Zero-padded GIS identification string of the HRU"},
        {"variable": "SUB", "unit": "None", "description": "Subbasin index containing the HRU"},
        {"variable": "LULC", "unit": "None", "description": "Land use classification category string identifier"},
        {"variable": "AREAkm2", "unit": "km2", "description": "HRU physical surface area"},
        {"variable": "SW_INITmm", "unit": "mm H2O", "description": "Initial soil water content at step start"},
        {"variable": "SW_ENDmm", "unit": "mm H2O", "description": "Final soil water content at step end"},
        {"variable": "PERCmm", "unit": "mm", "description": "Water percolating out of soil profile zone"},
        {"variable": "GW_Qmm", "unit": "mm", "description": "Groundwater flow contribution from HRU to subbasin outlet"},
        {"variable": "WYLDmm", "unit": "mm", "description": "Total water yield from the HRU"},
        {"variable": "LAI", "unit": "None", "description": "Leaf Area Index of vegetation in the HRU"},
        {"variable": "BIOMt_ha", "unit": "metric tons/ha", "description": "Biomass accumulation in the HRU"}
    ]
    pd.DataFrame(hru_features).to_csv("feature_dictionary_hrus.csv", index=False)
    print("Feature dictionaries generated successfully.")

def main():
    cio_path = r"d:\all in one\Desktop\new_PhD_project\swat1\TxtInOut\file.cio"
    fig_path = r"d:\all in one\Desktop\new_PhD_project\swat1\TxtInOut\fig.fig"
    rch_path = r"d:\all in one\Desktop\new_PhD_project\swat1\TxtInOut\output.rch"
    sub_path = r"d:\all in one\Desktop\new_PhD_project\swat1\TxtInOut\output.sub"
    hru_path = r"d:\all in one\Desktop\new_PhD_project\swat1\TxtInOut\output.hru"
    
    # 1. Parse Master Control File (file.cio)
    cio_data = parse_cio(cio_path)
    
    # 2. Parse Routing configuration (fig.fig)
    reach_downstream, reach_upstream = parse_fig(fig_path)
    
    # 3. Extract step metadata arrays
    rch_meta = generate_steps_metadata(rch_path, 483, cio_data['years'], cio_data['total_years_count'], cio_data['frequency'])
    sub_meta = generate_steps_metadata(sub_path, 483, cio_data['years'], cio_data['total_years_count'], cio_data['frequency'])
    hru_meta = generate_steps_metadata(hru_path, 1857, cio_data['years'], cio_data['total_years_count'], cio_data['frequency'])
    
    # 4. Parse and convert files
    parse_output_rch(rch_path, rch_meta, reach_downstream, reach_upstream, "reaches.parquet")
    parse_output_sub(sub_path, sub_meta, reach_downstream, "subbasins.parquet")
    parse_output_hru(hru_path, hru_meta, "hrus.parquet")
    
    # 5. Generate Feature Dictionaries
    generate_feature_dictionaries()
    
    print("\n--- SWAT PARSING COMPLETE ---")
    print("Files created in output directory:")
    print("  reaches.parquet")
    print("  subbasins.parquet")
    print("  hrus.parquet")
    print("  metadata_reaches.csv")
    print("  metadata_subbasins.csv")
    print("  metadata_hrus.csv")
    print("  feature_dictionary_reaches.csv")
    print("  feature_dictionary_subbasins.csv")
    print("  feature_dictionary_hrus.csv")

if __name__ == "__main__":
    main()
