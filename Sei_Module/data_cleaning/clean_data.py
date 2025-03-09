import pandas as pd
import numpy as np
import warnings
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def clean_and_save_data():
    """Cleans and prepares Sei network data"""
    try:
        # Load data with validation
        try:
            df = pd.read_csv('combined_datas.csv')
            logger.info(f"Loaded data with columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return

        # Validate essential columns
        essential_cols = ['blocks', 'evm_total_trans']
        missing = list(set(essential_cols) - set(df.columns))
        if missing:
            logger.error(f"Missing essential columns: {missing}")
            return

        # Column cleanup
        drop_columns = [
            'timestamp', 'value', 'fee', 'type', 'maxFeePerGas', 'maxPriorityFeePerGas',
            'priorityFee', 'burntFees', 'gasUsedByTransaction', 'nonce', 'data', 'raw',
            'failureReason', 'height', 'to', 'from', 'method', 'status', 'gasPrice',
            'gasLimit', 'actionType', 'evm_transactions', 'evm_statistics', 'cosmos_statistics'
        ]
        df.drop(columns=[c for c in drop_columns if c in df.columns], inplace=True)

        # Enhanced transaction hash handling
        tx_hash_names = ['hash', 'tx_hash', 'transaction_hash', 'evm_transaction_hash']
        for name in tx_hash_names:
            if name in df.columns:
                df.rename(columns={name: 'evm_transaction_hash'}, inplace=True)
                break
        else:
            logger.warning("Creating synthetic transaction IDs")
            df['evm_transaction_hash'] = [f"tx{i:08d}" for i in range(len(df))]

        # NaN handling
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].replace(0, np.nan)

        # Cosmos data fallback
        for col in df.columns:
            if 'cosmos' in col and df[col].isna().all():
                logger.warning(f"Resetting all-NaN column: {col}")
                df[col] = 0

        # Final validation
        critical_cols = ['blocks']
        df.dropna(subset=critical_cols, inplace=True)

        logger.info(f"Final columns: {df.columns.tolist()}")
        df.to_csv('official_sei_data3.csv', index=False)
        logger.info("Data cleaning completed successfully")

    except Exception as e:
        logger.error(f"Cleaning failed: {e}")
        raise

if __name__== "__main__":
    clean_and_save_data()