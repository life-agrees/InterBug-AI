from __future__ import absolute_import
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def detect_bugs(df, threshold_multiplier=1.5):
    """Enhanced bug detection with adjustable thresholds"""
    # Core features that must be present
    core_features = [
        'transaction_mismatch',
        'fee_ratio',
        'success_rate'
    ]
    
    # Additional features that provide more signal if available
    additional_features = [
        'gas_efficiency',
        'block_sync_ratio',
        'fee_inconsistency',
        'cross_chain_disparity',
        'gas_usage_anomaly'
    ]
    
    # Check for core features
    if not all(feat in df.columns for feat in core_features):
        missing = [f for f in core_features if f not in df.columns]
        logger.error(f"Missing required features: {missing}")
        return pd.DataFrame()
    
    # Check which additional features are available
    available_additional = [f for f in additional_features if f in df.columns]
    logger.info(f"Using {len(core_features)} core features and {len(available_additional)} additional features")
    
    # Combined feature list
    used_features = core_features + available_additional
    
    try:
        # Check for NaN values
        na_count = df[used_features].isna().sum().sum()
        if na_count > 0:
            logger.warning(f"Found {na_count} NaN values in input features")
            # Fill NaNs with median values for each column
            for col in used_features:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled NaNs in {col} with median {median_val}")
        
        # Calculate dynamic thresholds for core features with threshold multiplier
        core_thresholds = {
            'transaction_mismatch': df['transaction_mismatch'].abs().quantile(0.80) * threshold_multiplier,
            'fee_ratio': df['fee_ratio'].abs().quantile(0.80) * threshold_multiplier,
            # Careful with success_rate, using a different approach
            'success_rate': max(0, 0.7 - (0.2 * (threshold_multiplier - 1)))
        }
        
        # Avoid zero threshold for transaction_mismatch
        if core_thresholds['transaction_mismatch'] == 0:
            core_thresholds['transaction_mismatch'] = 1e-6
        
        # Calculate dynamic thresholds for additional features if available
        additional_thresholds = {}
        for feat in available_additional:
            additional_thresholds[feat] = df[feat].abs().quantile(0.85) * threshold_multiplier
        
        # Detailed logging of thresholds
        logger.info("Threshold Details:")
        logger.info(f"Threshold Multiplier: {threshold_multiplier}")
        logger.info("Core thresholds: " + ', '.join([f'{k}={v:.4f}' for k, v in core_thresholds.items()]))
        if additional_thresholds:
            logger.info("Additional thresholds: " + ', '.join([f'{k}={v:.4f}' for k, v in additional_thresholds.items()]))
        
        # Create bug conditions with weighted approach
        core_conditions = (
            (df['transaction_mismatch'].abs() > core_thresholds['transaction_mismatch']) |
            (df['fee_ratio'].abs() > core_thresholds['fee_ratio']) |
            (df['success_rate'] < core_thresholds['success_rate'])
        )
        
        # Initialize anomaly score
        df['anomaly_score'] = 0
        
        # Add points to anomaly score for each violated condition
        if core_conditions.any():
            df.loc[df['transaction_mismatch'].abs() > core_thresholds['transaction_mismatch'], 'anomaly_score'] += 2
            df.loc[df['fee_ratio'].abs() > core_thresholds['fee_ratio'], 'anomaly_score'] += 1
            df.loc[df['success_rate'] < core_thresholds['success_rate'], 'anomaly_score'] += 2
        
        # Additional features contribute less but still add signal
        for feat in available_additional:
            df.loc[df[feat].abs() > additional_thresholds[feat], 'anomaly_score'] += 1
        
        # Sort by anomaly score
        df = df.sort_values('anomaly_score', ascending=False)
        
        # Dynamic bug rate targeting
        # Use threshold_multiplier to adjust target bug rate
        target_bug_rate = min(max(0.2 * (2 - threshold_multiplier), 0.1), 0.3)
        target_bug_count = int(len(df) * target_bug_rate)
        
        # Determine minimum anomaly score for bug classification
        min_anomaly_score = max(1, df.iloc[min(target_bug_count, len(df)-1)]['anomaly_score'])
        
        logger.info(f"Target Bug Rate: {target_bug_rate:.2%}")
        logger.info(f"Minimum anomaly score for bug classification: {min_anomaly_score}")
        
        # Classify bugs based on anomaly score
        df['bug_detected'] = np.where(df['anomaly_score'] >= min_anomaly_score, 1, 0)
        
        # Log bug detection summary
        bug_count = df['bug_detected'].sum()
        bug_percentage = bug_count / len(df) * 100
        logger.info(f"Detected {bug_count} bugs in {len(df)} records ({bug_percentage:.2f}%)")
        
        # Ensure bug rate is within acceptable range
        if bug_percentage < 0.1 * 100 or bug_percentage > 0.3 * 100:
            logger.warning("Adjusting bug classification to target rate")
            # Mark top X% as bugs
            df['bug_detected'] = 0
            df.iloc[:target_bug_count, df.columns.get_loc('bug_detected')] = 1
            
            bug_count = df['bug_detected'].sum()
            logger.info(f"Adjusted to {bug_count} bugs in {len(df)} records ({bug_count/len(df)*100:.2f}%)")
        
        # Return results with all used features plus bug detection
        return df[used_features + ['bug_detected', 'anomaly_score']]
        
    except Exception as e:
        logger.error(f"Bug detection failed: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    try:
        # Logging configuration
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Check if preprocessed features exist
        if not os.path.exists('preprocessed_features.csv'):
            logger.error("Run preprocessing first!")
            exit(1)
        
        # Read preprocessed features
        features = pd.read_csv('preprocessed_features.csv')
        
        # Different threshold multiplier scenarios
        threshold_scenarios = [0.5, 1.0, 1.5, 2.0]
        
        for multiplier in threshold_scenarios:
            logger.info(f"\n--- Bug Detection with Threshold Multiplier: {multiplier} ---")
            bugs = detect_bugs(features, threshold_multiplier=multiplier)
            
            if not bugs.empty:
                # Save results for each scenario
                output_file = f'training_data_multiplier_{multiplier}.csv'
                bugs.to_csv(output_file, index=False)
                logger.info(f"Saved results to {output_file}")
                
                # Feature importance analysis
                if 'anomaly_score' in bugs.columns and bugs['anomaly_score'].max() > 0:
                    feature_corr = {}
                    for feat in bugs.columns:
                        if feat not in ['bug_detected', 'anomaly_score']:
                            feature_corr[feat] = bugs[feat].abs().corr(bugs['anomaly_score'])
                    
                    # Sort features by correlation with anomaly score
                    sorted_features = sorted(feature_corr.items(), key=lambda x: x[1], reverse=True)
                    logger.info("Feature importance based on correlation with anomaly score:")
                    for feat, corr in sorted_features:
                        logger.info(f"  {feat}: {corr:.4f}")
            else:
                logger.error("No bugs detected or valid data")
        
    except Exception as e:
        logger.exception("Overall bug detection process failed:")