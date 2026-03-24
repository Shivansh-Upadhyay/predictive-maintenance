import pandas as pd
import numpy as np
import os

# ── CHUNK 2: Define Column Names ───────────────────────────────────────────
COLUMNS = [
    'engine_id', 'cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1',  'sensor_2',  'sensor_3',  'sensor_4',  'sensor_5',
    'sensor_6',  'sensor_7',  'sensor_8',  'sensor_9',  'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# ── CHUNK 3: Data Loader Function ──────────────────────────────────────────
def load_data(data_path="data", dataset="FD001"):
    
    train_path = os.path.join(data_path, f"train_{dataset}.txt")
    test_path  = os.path.join(data_path, f"test_{dataset}.txt")
    rul_path   = os.path.join(data_path, f"RUL_{dataset}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS)
    test_df  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=COLUMNS)
    rul_df   = pd.read_csv(rul_path, header=None, names=["RUL_true"])

    print(f"✅ Train loaded : {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"✅ Test loaded  : {test_df.shape[0]} rows,  {test_df.shape[1]} columns")
    print(f"✅ RUL loaded   : {rul_df.shape[0]} engines")

    return train_df, test_df, rul_df


# ── CHUNK 4: Add RUL Labels ────────────────────────────────────────────────
def add_rul_labels(df):
    
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']

    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)

    print(f"\n RUL Labels Added!")
    print(f"   Max RUL : {df['RUL'].max()} cycles")
    print(f"   Min RUL : {df['RUL'].min()} cycles")
    print(f"   Avg RUL : {df['RUL'].mean():.1f} cycles")

    return df


# ── CHUNK 5: Data Health Check ─────────────────────────────────────────────
def explore_data(df, name="Train"):

    print(f"\n{'='*50}")
    print(f"  {name} Data Overview")
    print(f"{'='*50}")
    print(f"  Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Engines      : {df['engine_id'].nunique()} unique engines")
    print(f"  Cycle range  : {df['cycle'].min()} → {df['cycle'].max()}")
    
    missing = df.isnull().sum().sum()
    print(f"  Missing vals : {'✅ None' if missing == 0 else f'⚠️  {missing} found'}")
    
    print(f"\n  Sample rows:")
    print(df[['engine_id','cycle','sensor_2','sensor_3','sensor_4','RUL']].head(8).to_string(index=False))
    
    
# ── CHUNK 6: Run For Specific Dataset ─────────────────────────────────────
if __name__ == "__main__":

    import sys

    # Allow: python fetch_data.py FD002
    dataset = sys.argv[1] if len(sys.argv) > 1 else "FD001"

    print(f"\n{'='*50}")
    print(f"  Data Loading — {dataset}")
    print(f"{'='*50}\n")

    train_df, test_df, rul_df = load_data(dataset=dataset)
    train_df = add_rul_labels(train_df)
    explore_data(train_df, name=f"Train {dataset}")

    output_file = f"data/train_processed_{dataset}.csv"
    train_df.to_csv(output_file, index=False)
    print(f"\n Saved → {output_file}")
