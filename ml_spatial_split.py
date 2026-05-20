"""
ML Burn Classification with SPATIAL SPLIT
Tests if previous high accuracy was due to spatial autocorrelation
or genuine model performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data with coordinates...")
semey = pd.read_csv('semey_training_v2.csv')
karkaraly = pd.read_csv('karkaraly_training_v2.csv')
combined = pd.read_csv('combined_training_v2.csv')

# Check coordinates exist
print(f"\nSemey columns: {list(semey.columns)}")
print(f"\nSemey: {len(semey)} rows")
print(f"  Longitude range: {semey['longitude'].min():.3f} to {semey['longitude'].max():.3f}")
print(f"  Latitude range: {semey['latitude'].min():.3f} to {semey['latitude'].max():.3f}")

# Feature columns (8 raw bands only - no NBR, no dNBR to avoid data leakage)
features = ['B4_pre', 'B8A_pre', 'B11_pre', 'B12_pre',
            'B4_post', 'B8A_post', 'B11_post', 'B12_post']

print(f"\nFeatures used: {features}")
print(f"Total: {len(features)} features (no NBR, no dNBR - avoiding leakage)")

# ============================================================
# SPATIAL SPLIT FUNCTION
# ============================================================
def spatial_split(df, test_fraction=0.3):
    """
    Split data spatially: divide area into 4 quadrants based on 
    median longitude/latitude. Use 3 quadrants for training, 
    1 quadrant for testing.
    
    This prevents spatial autocorrelation between train/test sets.
    """
    lon_median = df['longitude'].median()
    lat_median = df['latitude'].median()
    
    # Assign quadrant (0,1,2,3) based on position relative to medians
    df = df.copy()
    df['quadrant'] = (
        (df['longitude'] >= lon_median).astype(int) * 2 +
        (df['latitude'] >= lat_median).astype(int)
    )
    
    # Use quadrant 3 (NE corner) as test set
    train = df[df['quadrant'] != 3]
    test = df[df['quadrant'] == 3]
    
    print(f"  Quadrant distribution:")
    print(f"  Q0 (SW): {len(df[df['quadrant']==0])}")
    print(f"  Q1 (NW): {len(df[df['quadrant']==1])}")
    print(f"  Q2 (SE): {len(df[df['quadrant']==2])}")
    print(f"  Q3 (NE - TEST): {len(df[df['quadrant']==3])}")
    
    return train, test

# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9, None],
    'min_samples_leaf': [1, 5, 10]
}

xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}

lgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9, -1],
    'learning_rate': [0.05, 0.1]
}

# ============================================================
# TRAIN AND EVALUATE FUNCTION (SPATIAL SPLIT)
# ============================================================
def train_and_evaluate_spatial(data, scenario_name):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name} (SPATIAL SPLIT)")
    print(f"{'='*60}")
    
    # Spatial split instead of random
    train_df, test_df = spatial_split(data)
    
    X_train = train_df[features]
    y_train = train_df['label']
    X_test = test_df[features]
    y_test = test_df['label']
    
    print(f"\n  Train: {len(X_train)} ({y_train.sum()} burned, {(y_train==0).sum()} unburned)")
    print(f"  Test:  {len(X_test)} ({y_test.sum()} burned, {(y_test==0).sum()} unburned)")
    
    # Check if test set has both classes (needed for meaningful evaluation)
    if y_test.sum() == 0 or (y_test==0).sum() == 0:
        print("  WARNING: Test set has only one class. Skipping.")
        return None
    
    models = {
        'Random Forest': (RandomForestClassifier(random_state=42, n_jobs=-1, 
                                                  class_weight='balanced'), rf_params),
        'XGBoost': (XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss',
                                   scale_pos_weight=max(1, (y_train==0).sum()/max(1,y_train.sum()))),
                    xgb_params),
        'LightGBM': (LGBMClassifier(random_state=42, n_jobs=1, verbose=-1,
                                     class_weight='balanced'), lgb_params)
    }
    
    results = {}
    
    for name, (model, params) in models.items():
        print(f"\n--- Training {name} ---")
        start = time.time()
        
        grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        
        elapsed = (time.time() - start) / 60
        
        y_pred = grid.best_estimator_.predict(X_test)
        
        oa = accuracy_score(y_test, y_pred)
        ua = precision_score(y_test, y_pred, zero_division=0)
        pa = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Best params: {grid.best_params_}")
        print(f"Overall Accuracy:    {oa:.4f}")
        print(f"User's Accuracy:     {ua:.4f}")
        print(f"Producer's Accuracy: {pa:.4f}")
        print(f"F1 Score:            {f1:.4f}")
        print(f"Time: {elapsed:.1f} min")
        print(f"Confusion Matrix:\n{cm}")
        
        if hasattr(grid.best_estimator_, 'feature_importances_'):
            importance = pd.Series(grid.best_estimator_.feature_importances_, 
                                    index=features).sort_values(ascending=False)
            print(f"Top 5 features: {importance.head().to_dict()}")
        
        results[name] = {
            'best_params': grid.best_params_,
            'OA': oa, 'UA': ua, 'PA': pa, 'F1': f1,
            'confusion_matrix': cm.tolist(),
            'time_min': elapsed
        }
    
    return results

# ============================================================
# RUN ALL 3 SCENARIOS
# ============================================================
print("\n" + "="*60)
print("STARTING ML CLASSIFICATION WITH SPATIAL SPLIT")
print("="*60)

all_results = {}

# Scenario 1: Semey only
print("\n>>> SEMEY <<<")
all_results['Semey'] = train_and_evaluate_spatial(semey, "Semey Only")

# Scenario 2: Karkaraly only
print("\n>>> KARKARALY <<<")
all_results['Karkaraly'] = train_and_evaluate_spatial(karkaraly, "Karkaraly Only")

# Scenario 3: Combined
print("\n>>> COMBINED <<<")
all_results['Combined'] = train_and_evaluate_spatial(combined, "Combined (Both Areas)")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY - F1 SCORES (SPATIAL SPLIT)")
print("="*60)
print(f"{'Scenario':<15} {'RF':<10} {'XGBoost':<10} {'LightGBM':<10}")
for scenario, results in all_results.items():
    if results is None:
        print(f"{scenario:<15} (skipped - test set imbalance)")
        continue
    rf = results['Random Forest']['F1']
    xgb = results['XGBoost']['F1']
    lgb = results['LightGBM']['F1']
    print(f"{scenario:<15} {rf:<10.4f} {xgb:<10.4f} {lgb:<10.4f}")

# Save results
import json
with open('ml_results_spatial.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nResults saved to ml_results_spatial.json")

print("\n" + "="*60)
print("INTERPRETATION GUIDE")
print("="*60)
print("If F1 scores stayed 0.95+: Models genuinely learn spectral patterns")
print("If F1 dropped to 0.80-0.95: Previous high was spatial autocorrelation")
print("If F1 dropped below 0.80: Strong spatial dependence, need more data")