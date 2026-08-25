import pandas as pd
import numpy as np

class IdentityScaler:
    def fit(self, X):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X):
        return X

class FeatureScaler:
    def __init__(self):
        self.scalers = {}
        
    def add_scaler(self, feature_name, scaler):
        self.scalers[feature_name] = scaler
        
    def transform_df(self, df):
        df_scaled = df.copy()
        for col, scaler in self.scalers.items():
            if col in df_scaled.columns:
                vals = df_scaled[col].values.reshape(-1, 1)
                df_scaled[col] = scaler.transform(vals).ravel()
        return df_scaled
        
    def scale_features_matrix(self, feat_cols, df_pre):
        # Applies scaling to feature columns in the correct order and returns scaled numpy array
        df_pre_scaled = self.transform_df(df_pre[feat_cols])
        # Replace remaining NaNs (if any) with 0.0
        return np.nan_to_num(df_pre_scaled.values, nan=0.0)
