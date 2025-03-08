from __future__ import absolute_import
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df):
    """Enhanced preprocessing with more predictive features"""
    try:
        # Check for NaN values in input data
        na_count = df.isna().sum().sum()
        if na_count > 0:
            logger.warning(f"Found {na_count} NaN values in input data")
        
        # Unit Normalization
        df = df.copy()
        
        # Convert wei to ETH for EVM fees
        df['evm_avg_trans_fee'] = df['evm_avg_trans_fee'] / 1e18
        
        # Convert large numbers to millions
        for col in ['cosmos_total_trans', 'evm_total_trans']:
            df[col] = df[col] / 1e6
            
        # Log transform heavy-tailed features
        for col in ['cosmos_total_gas_used', 'evm_gas_growth']:
            # Handle potential zero or negative values before log transform
            df[f"log_{col}"] = np.log1p(df[col].fillna(0).clip(lower=0))
            
        # Create core features with NaN handling
        # Handle division by zero and NaN values
        df['transaction_mismatch'] = (df['evm_total_trans'] - df['cosmos_total_trans']) / (df['cosmos_total_trans'] + 1)
        df['fee_ratio'] = np.log1p(df['evm_avg_trans_fee'].fillna(0)) - np.log1p(df['cosmos_avg_trans_fee'].fillna(0))
        df['success_rate'] = df['evm_trans_success_rate'].fillna(0)  # Handle NaN with 0
        
        # NEW FEATURE: Gas efficiency ratio
        df['gas_efficiency'] = np.log1p(df['evm_avg_gas_limit'].fillna(0)) / np.log1p(df['evm_total_trans'].fillna(1) + 1)
        
        # NEW FEATURE: Block synchronization
        df['block_sync_ratio'] = (df['evm_new_block'].fillna(0) / (df['cosmos_new_block'].fillna(0) + 1))
        
        # NEW FEATURE: Transaction volume volatility (using rolling window if time-ordered data)
        # If data is time-ordered, uncommenting this section will add volatility metrics
        """
        if len(df) > 5:  # Need enough data points
            df['trans_volatility'] = df['evm_total_trans'].rolling(window=5, min_periods=1).std() / \
                                    df['evm_total_trans'].rolling(window=5, min_periods=1).mean()
            df['fee_volatility'] = df['evm_avg_trans_fee'].rolling(window=5, min_periods=1).std() / \
                                  df['evm_avg_trans_fee'].rolling(window=5, min_periods=1).mean().replace(0, np.nan).fillna(1)
        else:
            logger.warning("Not enough data points for volatility calculation")
            df['trans_volatility'] = 0
            df['fee_volatility'] = 0
        """
        
        # NEW FEATURE: Fee inconsistency (absolute difference from median)
        df['fee_inconsistency'] = np.abs(df['evm_avg_trans_fee'] - df['evm_avg_trans_fee'].median()) / (df['evm_avg_trans_fee'].median() or 1)
        
        # NEW FEATURE: Cross-chain disparity score
        df['cross_chain_disparity'] = np.abs(df['transaction_mismatch']) + np.abs(df['fee_ratio'])
        
        # NEW FEATURE: Gas usage efficiency 
        df['gas_usage_anomaly'] = np.abs(df['log_evm_gas_growth'] - df['log_cosmos_total_gas_used'])
        
        # Select all features including new ones
        features = [
            # Original features
            'transaction_mismatch',
            'fee_ratio',
            'success_rate',
            'log_cosmos_total_gas_used',
            'log_evm_gas_growth',
            'cosmos_new_block',
            'evm_new_block',
            
            # New features
            'gas_efficiency',
            'block_sync_ratio',
            'fee_inconsistency',
            'cross_chain_disparity',
            'gas_usage_anomaly'
            
            # Uncomment if using volatility features
            # 'trans_volatility',
            # 'fee_volatility',
        ]
        
        # Check for NaN values after feature creation
        na_count_after = df[features].isna().sum().sum()
        if na_count_after > 0:
            logger.warning(f"Found {na_count_after} NaN values after feature creation")
            # Fill remaining NaNs with 0 as a safe default
            df[features] = df[features].fillna(0)
        
        # Scale features
        scaler = RobustScaler()
        df[features] = scaler.fit_transform(df[features])
        
        return df[features]
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    try:
        df = pd.read_csv('official_sei_data3.csv')
        processed = preprocess_data(df)
        processed.to_csv('preprocessed_features.csv', index=False)
        logger.info(f"Preprocessing completed successfully with {len(processed.columns)} features")
    except Exception as e:
        logger.exception("Preprocessing failed:")