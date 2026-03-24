# ── CHUNK 1: Imports ───────────────────────────────────────────────────────
# sklearn  → Random Forest + evaluation metrics
# xgboost  → XGBoost model
# joblib   → saves our trained model to a file
# warnings → keeps output clean

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# ── CHUNK 2: Define Features & Target ─────────────────────────────────────
# Dropping columns that are NOT useful for prediction:
#   engine_id → just an identifier, not a signal
#   cycle     → raw cycle number (we already have cycle_normalized)
#   RUL       → this is what we're PREDICTING, not an input
#
# Everything else = input features for the model

def get_features_and_target(df):

    drop_cols = ['engine_id', 'cycle', 'RUL']
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['RUL']

    print(f" Target        : RUL (Remaining Useful Life)")
    print(f" Features used : {len(feature_cols)}")
    print(f" Dataset size  : {X.shape[0]} rows")

    return X, y, feature_cols


# ── CHUNK 3: Evaluation Function ──────────────────────────────────────────
#   RMSE → average error in cycles (main metric, lower = better)
#   MAE  → similar to RMSE but less sensitive to big errors
#   R²   → how much variance the model explains (1.0 = perfect, 0 = useless)

def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, 125)  # RUL can't be negative

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae  = mean_absolute_error(y_test, predictions)
    r2   = r2_score(y_test, predictions)

    print(f"\n  {'─'*40}")
    print(f"  {model_name} Results:")
    print(f"  {'─'*40}")
    print(f"  RMSE : {rmse:.2f} cycles  ← main metric")
    print(f"  MAE  : {mae:.2f}  cycles")
    print(f"  R²   : {r2:.4f}  (1.0 = perfect)")

    return {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2, 
            "predictions": predictions}
    
    
    # ── CHUNK 4: Train Random Forest ──────────────────────────────────────────
# n_estimators=200  → builds 200 decision trees
# max_depth=10      → each tree can be at most 10 levels deep
#                     (prevents overfitting — memorizing training data)
# n_jobs=-1         → uses all CPU cores to train faster
# random_state=42   → makes results reproducible (same output every run)

def train_random_forest(X_train, y_train):

    print("\n Training Random Forest...")
    
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    print("   ✅ Random Forest trained!")
    
    return rf_model


# ── CHUNK 5: Train XGBoost ────────────────────────────────────────────────
# n_estimators=300    → builds 300 trees sequentially
# learning_rate=0.05  → each tree corrects errors slowly and carefully
#                       (lower = more careful = usually better accuracy)
# max_depth=6         → slightly shallower than RF (XGBoost works better this way)
# subsample=0.8       → each tree uses 80% of data (reduces overfitting)

def train_xgboost(X_train, y_train):

    print("\n Training XGBoost...")
    
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    xgb_model.fit(X_train, y_train)
    print("   ✅ XGBoost trained!")
    
    return xgb_model


# ── CHUNK 6: Feature Importance ───────────────────────────────────────────

def show_feature_importance(model, feature_cols, top_n=10):

    importance = pd.DataFrame({
        'feature':    feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🔍 Top {top_n} Most Important Features (Random Forest):")
    print(f"   {'─'*45}")
    for _, row in importance.head(top_n).iterrows():
        bar = '█' * int(row['importance'] * 200)
        print(f"   {row['feature']:<30} {row['importance']:.4f}  {bar}")

    return importance


# ── CHUNK 7: Run For Specific Dataset ─────────────────────────────────────
if __name__ == "__main__":

    import sys

    # Allow: python train.py FD002
    dataset = sys.argv[1] if len(sys.argv) > 1 else "FD001"

    print(f"\n{'='*50}")
    print(f"  Model Training — {dataset}")
    print(f"{'='*50}\n")

    # Load featured data for this dataset
    input_file = f"data/train_featured_{dataset}.csv"
    print(f" Loading {input_file}...")
    df = pd.read_csv(input_file)

    # Get features and target
    X, y, feature_cols = get_features_and_target(df)

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train split : {X_train.shape[0]} rows")
    print(f"  Test split  : {X_test.shape[0]} rows")

    # Train both models
    print(f"\n{'='*50}")
    print(f"  Model Training & Comparison")
    print(f"{'='*50}")

    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Evaluate both
    print(f"\n{'='*50}")
    print(f"  Evaluation on Test Set")
    print(f"{'='*50}")

    rf_results  = evaluate_model(rf_model,  X_test, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost      ")

    # Pick winner
    print(f"\n{'='*50}")
    print(f"  Winner")
    print(f"{'='*50}")

    if rf_results['RMSE'] < xgb_results['RMSE']:
        best_model = rf_model
        best_name  = "Random Forest"
    else:
        best_model = xgb_model
        best_name  = "XGBoost"

    print(f"\n   Best Model : {best_name}")
    print(f"   Best RMSE  : {min(rf_results['RMSE'], xgb_results['RMSE']):.2f} cycles")

    # Feature importance
    show_feature_importance(rf_model, feature_cols)

    # Save model + features with dataset name
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model,   f"models/best_model_{dataset}.pkl")
    joblib.dump(feature_cols, f"models/feature_cols_{dataset}.pkl")

    # Save comparison
    comparison = pd.DataFrame([
    {"Model": "Random Forest", "RMSE": rf_results["RMSE"],
     "MAE": rf_results["MAE"],  "R2": rf_results["R2"]},
    {"Model": "XGBoost",       "RMSE": xgb_results["RMSE"],
     "MAE": xgb_results["MAE"], "R2": xgb_results["R2"]}
])
    
    
    
# ── CHUNK 8: Save Comparison Results ──────────────────────────────────────
# Saving both metrices for  app.py

comparison = pd.DataFrame([
    {
        "Model": "Random Forest",
        "RMSE" : rf_results["RMSE"],
        "MAE"  : rf_results["MAE"],
        "R2"   : rf_results["R2"]
    },
    {
        "Model": "XGBoost",
        "RMSE" : xgb_results["RMSE"],
        "MAE"  : xgb_results["MAE"],
        "R2"   : xgb_results["R2"]
    }
])

comparison.to_csv(f"models/model_comparison_{dataset}.csv", index=False)
print(f"💾 Saved → models/model_comparison_{dataset}.csv")
