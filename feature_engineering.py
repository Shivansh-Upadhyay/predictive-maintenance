import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── CHUNK 1: Sesnsors ──────────────────────────────────

USEFUL_SENSORS = [
    'sensor_2',  'sensor_3',  'sensor_4',  'sensor_7',
    'sensor_8',  'sensor_9',  'sensor_11', 'sensor_12',
    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
    'sensor_20', 'sensor_21'
]


# ── CHUNK 2 ───────────────────────────────
# FD002 has 6 operating conditions defined by 3 settings
# We use KMeans to automatically group similar conditions together
# Then normalize sensors WITHIN each cluster
#
# WHY? A sensor reading of 520 means something completely different
# at sea level vs high altitude — clustering fixes this so the model
# sees clean, comparable values regardless of operating condition
#
# This is called "condition-based normalization" in industry

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_operating_conditions(df, n_clusters=6):

    print(f"Clustering operating conditions (n={n_clusters})...")

    # Cluster based on the 3 operational settings
    op_cols    = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    op_data    = df[op_cols].values

    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['op_cluster'] = kmeans.fit_predict(op_data)

    print(f"   Found {n_clusters} operating condition clusters")
    print(f"   Cluster distribution:")
    print(df['op_cluster'].value_counts().sort_index().to_string())

    return df, kmeans

# ── CHUNK 2B: ───────────────────────────

def normalize_by_cluster(df):

    print(f"Normalizing sensors within each cluster...")

    for cluster_id in df['op_cluster'].unique():
        mask = df['op_cluster'] == cluster_id

        for sensor in USEFUL_SENSORS:
            cluster_mean = df.loc[mask, sensor].mean()
            cluster_std  = df.loc[mask, sensor].std()

            # Avoid division by zero for constant sensors
            if cluster_std > 0:
                df.loc[mask, sensor] = (
                    (df.loc[mask, sensor] - cluster_mean) / cluster_std
                )
            else:
                df.loc[mask, sensor] = 0.0

    print(f"Sensors normalized across all clusters")
    return df




# ── CHUNK 3: Rolling Window Features ──────────────────────────────────────


def add_rolling_features(df, window=10):

    print(f"⚙️  Adding rolling features (window={window} cycles)...")
    
    for sensor in USEFUL_SENSORS:
        # Rolling mean — the trend
        df[f'{sensor}_mean_{window}'] = (
            df.groupby('engine_id')[sensor]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        # Rolling std — the stability
        df[f'{sensor}_std_{window}'] = (
            df.groupby('engine_id')[sensor]
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )

    print(f" Added {len(USEFUL_SENSORS) * 2} rolling features")
    return df

# ── CHUNK 4: Cycle-Based Features ─────────────────────────────────────────

def add_cycle_features(df):

    print(f"Adding cycle-based features...")

    # Normalize cycle: how far through its life is this engine?
    max_cycles = df.groupby('engine_id')['cycle'].transform('max')
    df['cycle_normalized'] = df['cycle'] / max_cycles

    # Sensor difference from previous cycle (rate of change)
    for sensor in USEFUL_SENSORS:
        df[f'{sensor}_diff'] = (
            df.groupby('engine_id')[sensor]
            .transform(lambda x: x.diff().fillna(0))
        )

    print(f"   Added {1 + len(USEFUL_SENSORS)} cycle features")
    return df


# ── CHUNK 5: Cap RUL Values ────────────────────────────────────────────────

def cap_rul(df, max_rul=125):

    df['RUL'] = df['RUL'].clip(upper=max_rul)
    print(f"  RUL capped at {max_rul} cycles")
    print(f"   New RUL range: {df['RUL'].min()} → {df['RUL'].max()}")
    return df


# ── CHUNK 6: Run Full Pipeline (Updated for FD001 + FD002) ────────────────
if __name__ == "__main__":

    import sys

    # Allow running for specific dataset: python feature_engineering.py FD002
    dataset = sys.argv[1] if len(sys.argv) > 1 else "FD001"

    print(f"\n{'='*50}")
    print(f"  Feature Engineering — {dataset}")
    print(f"{'='*50}\n")

    # Load processed data for this dataset
    input_file  = f"data/train_processed_{dataset}.csv"
    output_file = f"data/train_featured_{dataset}.csv"

    print(f" Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Shape before : {df.shape[0]} rows × {df.shape[1]} columns")

    # FD002 and FD004 need condition clustering + normalization
    if dataset in ["FD002", "FD004"]:
        df, kmeans = cluster_operating_conditions(df)
        df         = normalize_by_cluster(df)
        # Save kmeans for use in app later
        import joblib
        joblib.dump(kmeans, f"models/kmeans_{dataset}.pkl")
        print(f"    Saved → models/kmeans_{dataset}.pkl")

    # Same feature engineering for all datasets
    df = add_rolling_features(df)
    df = add_cycle_features(df)
    df = cap_rul(df)

    print(f"\n   Shape after  : {df.shape[0]} rows × {df.shape[1]} columns")

    df.to_csv(output_file, index=False)
    print(f"\n Saved → {output_file}")
    print(f"\n Feature Engineering Complete for {dataset}!")
    
    
