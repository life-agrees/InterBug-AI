from __future__ import absolute_import
import pandas as pd
import numpy as np
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel


import xgboost as xgb

# Imbalanced learning
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

def train_model():
    try:
        df = pd.read_csv('training_data.csv')
        
        # Debug NaN values
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"NaN count per column:\n{df.isna().sum()}")
        
        # Check class distribution
        class_counts = df['bug_detected'].value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        if len(df) < 5:
            raise ValueError("Insufficient data for training")
        
        # Get all feature columns except 'bug_detected' and 'anomaly_score'
        feature_cols = [col for col in df.columns 
                       if col not in ['bug_detected', 'anomaly_score']]
        
        X = df[feature_cols]
        y = df['bug_detected']
        
        logger.info(f"Training with {len(feature_cols)} features: {', '.join(feature_cols)}")
        
        # Apply imputation to handle NaN values first
        logger.info("Applying imputation to handle missing values")
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Handle class imbalance with SMOTE - improved synthetic sampling
        min_class_count = min(class_counts.values)
        if min_class_count < 10:  # More aggressive resampling threshold
            logger.info(f"Class imbalance detected. Applying SMOTE to balance classes.")
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count-1))
            X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
            
            # Check new class distribution after SMOTE
            resampled_counts = pd.Series(y_resampled).value_counts()
            logger.info(f"New class distribution after SMOTE: {resampled_counts.to_dict()}")
            
            X_imputed, y = X_resampled, y_resampled
        
        # Feature correlation analysis
        logger.info("Analyzing feature correlations")
        X_df = pd.DataFrame(X_imputed, columns=feature_cols)
        correlation = X_df.corr()
        
        # Log highly correlated features
        corr_threshold = 0.8
        high_corr_pairs = []
        for i in range(len(correlation.columns)):
            for j in range(i+1, len(correlation.columns)):
                if abs(correlation.iloc[i, j]) > corr_threshold:
                    high_corr_pairs.append((correlation.columns[i], correlation.columns[j], correlation.iloc[i, j]))
        
        if high_corr_pairs:
            logger.info(f"Highly correlated feature pairs (>{corr_threshold}):")
            for col1, col2, corr in high_corr_pairs:
                logger.info(f"  - {col1} and {col2}: {corr:.4f}")
        
        # Perform feature selection with XGBoost
        logger.info("Performing feature selection with XGBoost")
        base_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        # Fit base model for feature selection
        base_model.fit(X_imputed, y)
        
        # Visualize feature importances before selection
        importances = base_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Log initial feature importances
        logger.info("Initial feature importances:")
        for f in range(X_imputed.shape[1]):
            logger.info(f"{feature_cols[indices[f]]}: {importances[indices[f]]:.4f}")
        
        # Use a custom threshold for feature selection
        selector = SelectFromModel(base_model, threshold='0.5*mean')
        X_selected = selector.fit_transform(X_imputed, y)
        
        # Get selected feature names
        selected_features = []
        for feature_idx in selector.get_support(indices=True):
            selected_features.append(feature_cols[feature_idx])
        
        logger.info(f"Selected {len(selected_features)} important features: {', '.join(selected_features)}")
        
        # Handle train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Use repeated cross-validation for more reliable estimates
        logger.info("Performing repeated stratified cross-validation")
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        
        # Hyperparameter tuning with grid search for XGBoost
        logger.info("Performing hyperparameter optimization")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        base_xgb = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        grid_search = GridSearchCV(
            base_xgb, 
            param_grid, 
            cv=cv, 
            scoring='f1',
            verbose=1,
            n_jobs=-1  # Use all available cores
        )
        
        grid_search.fit(X_selected, y)
        
        # Get best hyperparameters
        best_params = grid_search.best_params_
        logger.info(f"Best hyperparameters: {best_params}")
        
        # Train final model with best hyperparameters
        model = xgb.XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        # Cross validation on final model architecture
        cv_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='f1')
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1 score: {cv_scores.mean():.4f}")
        logger.info(f"CV F1 score std dev: {cv_scores.std():.4f}")
        
        # Train final model on all data
        model.fit(X_selected, y)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Log detailed evaluation
        logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Calculate and log additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        logger.info(f"Specificity (True Negative Rate): {specificity:.4f}")
        
        # Feature importance visualization for final model
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Log feature importances
        logger.info("Feature importances:")
        for _, row in feature_importance.iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        # Save model, imputer, and feature selector together using pickle
        logger.info("Saving model artifacts with pickle...")
        with open("interbug_model.pkl", "wb") as f:
            pickle.dump((model, imputer, selector), f)
        
        # Also save feature names for future reference
        with open("feature_names.pkl", "wb") as f:
            pickle.dump({
                'all_features': feature_cols,
                'selected_features': selected_features
            }, f)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_model()